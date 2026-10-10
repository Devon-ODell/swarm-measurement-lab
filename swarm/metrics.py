#!/usr/bin/env python3
"""Request accounting and time accounting for one swarm run.

Requests become a first-class, enforceable budget, provider retries are visible and
bounded, and a run's wall time is reported in mutually exclusive buckets rather
than a single "Waiting" state. Everything here is a small, file-backed accumulator
so a SIGKILLed turn still leaves an exact count behind, and every number a report
shows can be traced to a row.

Requests
    flint.py writes one line to a request sidecar before every HTTP request attempt
    (including retries), so the daemon never has to parse human log lines. swarmd
    consumes the sidecar after the turn ends and banks the count per role/model.

Time
    A worker publishes a structured state code on every transition. `BucketClock`
    accumulates seconds per state; the total is a run's wall time by construction,
    so "Waiting" can no longer swallow an allowance hold.

None of this makes a network call or touches the user's checkout.
"""
import json
import threading
import time
from pathlib import Path

# State codes a worker publishes. They are mutually exclusive and exhaustive for
# the time a worker is alive; `idle` means no task is claimed.
IDLE = "idle"
ACTIVE_MODEL = "active_model"        # a model request is in flight
TESTING = "testing"                  # a supervisor gate is running
REVIEWING = "reviewing"              # a reviewer/adversary turn is running
PROVIDER_BACKOFF = "provider_backoff"
ALLOWANCE_BLOCKED = "allowance_blocked"
DEPENDENCY_BLOCKED = "dependency_blocked"
OPERATOR_DRAINING = "operator_draining"

WALL_BUCKETS = (IDLE, ACTIVE_MODEL, TESTING, REVIEWING, PROVIDER_BACKOFF,
                ALLOWANCE_BLOCKED, DEPENDENCY_BLOCKED, OPERATOR_DRAINING)


def _now():
    return time.time()


class RequestLog:
    """Append-only sidecar: one JSON line per request attempt.

    `flint.py` writes to the path in FLINT_REQUEST_FILE with `append()` before every
    HTTP attempt (streams and retries count alike, because all of them hit the API).
    The daemon reads the whole file after a turn and banks the total into its own
    counters; the file is the durable source of truth if the daemon dies first.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, model="", role="", attempt=1, retry_of=None,
               error_class=None, paid=False, ts=None):
        row = {"t": ts if ts is not None else _now(),
               "model": model, "role": role, "attempt": int(attempt),
               "retry_of": retry_of, "error_class": error_class, "paid": bool(paid)}
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass  # a count line must never kill a turn

    def read(self):
        """(rows, total). Total counts every line; corrupt lines are skipped, not fatal."""
        try:
            lines = self.path.read_text().splitlines()
        except OSError:
            return [], 0
        rows, total = [], 0
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows.append(row)
                total += 1
            except ValueError:
                continue
        return rows, total

    def clear(self):
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


class BucketClock:
    """Wall-time buckets. Mutations are cheap and transitions are exact."""

    def __init__(self):
        self._lock = threading.Lock()
        self._buckets = {k: 0.0 for k in WALL_BUCKETS}
        self._state = None
        self._since = None

    def enter(self, state):
        if state not in WALL_BUCKETS:
            raise ValueError(f"unknown state {state!r}")
        now = _now()
        with self._lock:
            if self._state is not None and self._state != state:
                self._buckets[self._state] += now - self._since
            self._state, self._since = state, now

    def poke(self):
        """Freeze the current bucket's elapsed time without changing state."""
        now = _now()
        with self._lock:
            if self._state is not None:
                self._buckets[self._state] += now - self._since
                self._since = now

    def snapshot(self):
        with self._lock:
            out = dict(self._buckets)
            if self._state is not None:
                out[self._state] += _now() - self._since
            return out

    def total(self):
        return sum(self.snapshot().values())

    def finished_summary(self):
        """(buckets, wall_seconds) with the current state already banked."""
        snap = self.snapshot()
        return snap, snap["idle"] + sum(v for k, v in snap.items() if k != "idle")


class Reconcile:
    """Provider-vs-local request reconciliation.

    At run start we record `provider_before` (OpenRouter's own daily free-request
    counter) and `provider_after` at run stop; the difference is attributed against
    the locally observed HTTP attempts. The difference is shown explicitly — never
    silently assigned to the swarm.
    """

    @staticmethod
    def summary(local_rows, provider_before=None, provider_after=None, paid_requests=0):
        local = sum(1 for r in local_rows if r.get("paid") is not True)
        paid_local = sum(1 for r in local_rows if r.get("paid") is True)
        provider_delta = None
        if isinstance(provider_before, int) and isinstance(provider_after, int):
            provider_delta = max(0, provider_after - provider_before)
        other = None
        if provider_delta is not None:
            other = max(0, provider_delta - local)
        return {
            "provider_delta": provider_delta,
            "locally_observed": local,
            "paid_requests": paid_local + paid_requests,
            "other_unattributed": other,
        }

    @staticmethod
    def format_line(s):
        return (f"provider delta {s.get('provider_delta', 0) if s.get('provider_delta') is not None else '?'}"
                f"                  locally observed HTTP attempts {s.get('locally_observed', 0)}"
                f" other/unattributed {s.get('other_unattributed') if s.get('other_unattributed') is not None else '?'}"
                f" paid requests {s.get('paid_requests', 0)}")


def appreciate_seconds(seconds):
    """Compact human duration: 4h00m, 1h48m, 23m, 45s."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s" if s else f"{m}m"
    return f"{s}s"