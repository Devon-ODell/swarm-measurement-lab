#!/usr/bin/env python3
"""Reinforcement without weight updates: the swarm learns which agents to trust.

Free API models cannot be fine-tuned, so the swarm reinforces what it controls:

- Bandits. The implementer model, the planner model and the planner persona
  are drawn by Thompson sampling over Beta posteriors. Rewards shift later
  draws toward what works; a discount lets a model that got worse lose ground.
- Playbook. Accepted work leaves a one-line lesson, rejected work a pitfall.
  Both are shown to later agents and credited with the reward of every task
  they were shown to, so advice that helps rises and noise sinks.
- Breakthroughs. Work far above the running standard is pinned, reported,
  reinforced twice and followed up with priority.

External gates come first: tests, adversarial review and landing on trunk.
The judge's impact/creativity/quality scores only rank work that already
passed them, so a flattering judge cannot promote broken code.
"""
import fcntl
import json
import math
import random
import re
import statistics
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

PRIOR = 1.0
DISCOUNT = 0.97          # per pull of an arm: evidence half-life ≈ 23 pulls
COOLDOWN_BASE = 1800     # seconds a model rests after provider failure; doubles per strike
COOLDOWN_MAX = 12 * 3600 # also: strikes older than this are forgotten
BUSY_COOLDOWN = 120      # rate-limited upstream ("busy"): a short rest, doubling per busy spell
BUSY_MAX = 1800

# Failure stages and their reward. None means "not the implementer's doing": no update.
# Faulty code earns nothing; how hard it hurts is set by PENALTY_WEIGHT below.
STAGE_REWARD = {
    "agent_timeout": None, # wall-clock exhaustion is not a proven code defect
    "no_change": 0.0,       # produced nothing
    "model_error": 0.0,     # malformed or empty responses
    "weakened_tests": 0.0,  # tried to cheat the gate
    "tests_failed": 0.0,    # shipped code that fails the tests
    "rejected": 0.0,        # green, but the reviewer proved a defect
    "conflict": 0.3,        # accepted, but lost the race to land on trunk
    "review_error": None,
    "baseline": None,
    "refused": None,
}
FAULTY = ("weakened_tests", "tests_failed", "rejected")
CRIME = {"weakened_tests": "weakened existing tests to fake a pass",
         "tests_failed": "shipped code that fails the tests",
         "rejected": "shipped a defect the reviewer proved"}

# How much evidence one outcome adds to a model's posterior. Punishment outweighs
# praise: one faulty submission costs more than two good ones earn, so a model
# that ships broken code is drawn far less until it earns its way back.
REWARD_WEIGHT = 2.0          # a landed task counts as two pulls at its reward
BREAKTHROUGH_WEIGHT = 3.0    # plus three more at full reward
PENALTY_WEIGHT = {
    "weakened_tests": 6.0,   # cheating is the worst offence
    "tests_failed": 4.0,
    "rejected": 4.0,
    "model_error": 2.0,
    "no_change": 2.0,
    "conflict": 0.5,         # not the implementer's fault
}
REPAIR_COST = 0.1            # reward lost per repair round a landed task needed
WEIGHTS = {"impact": 0.35, "creativity": 0.35, "quality": 0.30}
STOP = set("the and for with that this from into when then than have has are was were will "
           "would should could can not but all any add adds added make makes use uses using "
           "new test tests support".split())


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def tokens(text):
    return {w for w in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if w not in STOP}


def similarity(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def novelty(text, prior_texts):
    """1 − max Jaccard similarity to work already accepted."""
    if not tokens(text):
        return 0.0
    return round(1.0 - max((similarity(text, p) for p in prior_texts), default=0.0), 3)


def reward(stage, scores=None, nov=0.0, repairs=0, goal_weight=1.0):
    """Reward in [0, 1], or None when the outcome says nothing about the implementer.

    Any accepted change earns at least 0.4, above every failure. The judge's
    scores spread accepted work over 0.4–1.0; each repair round the change
    needed first costs 0.1 (never below 0.4); novelty adds up to 0.1.

    `goal_weight` scales what the change *earned* above that floor, by where the goal ranks
    what it served. Landing something is always worth 0.4 — the floor is the invariant above,
    and a landed change must never be worth less than a failure — but landing the fourth
    priority is worth much less than landing the first."""
    if stage != "accepted":
        return STAGE_REWARD.get(stage)
    earned = (0.6 * sum(w * clamp(scores.get(k, 5), 0, 10) for k, w in WEIGHTS.items()) / 10
              if scores else 0.2)
    base = 0.4 + earned * clamp(goal_weight, 0.0, 1.0)
    base = max(0.4, base - REPAIR_COST * max(0, repairs or 0))
    return round(min(1.0, base + 0.1 * clamp(nov, 0.0, 1.0)), 4)


def weight(stage):
    """Evidence weight of an outcome for the model that produced it."""
    return REWARD_WEIGHT if stage == "accepted" else PENALTY_WEIGHT.get(stage, 1.0)


def _last_error(text):
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    for l in reversed(lines):
        if re.search(r"error|fail|assert|exception|panic", l, re.I):
            return l[:240]
    return lines[-1][:240] if lines else ""


def defect(stage, artifact=None, fallback=""):
    """One line naming what was wrong, taken from supervisor evidence where possible."""
    if stage == "weakened_tests":
        return "removed assertions from existing tests to get a green run"
    d = Path(artifact) if artifact else None
    if d and d.is_dir():
        if stage == "rejected":
            order = {"blocker": 0, "major": 1, "minor": 2}
            for p in sorted(d.glob("review-*.json"), reverse=True):
                try:
                    r = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                if r.get("verdict") != "request_changes":
                    continue
                findings = sorted((f for f in r.get("findings", []) if isinstance(f, dict)),
                                  key=lambda f: order.get(f.get("severity"), 3))
                if findings:
                    f = findings[0]
                    return (f"{f.get('path')}:{f.get('line')} {f.get('issue', '')}"
                            f" (repro: {f.get('verification', '')})")[:300]
                failed = [c for c in r.get("checks", []) if isinstance(c, dict) and not c.get("passed")]
                if failed:
                    return f"failed {failed[0].get('criterion')}: {failed[0].get('evidence', '')}"[:300]
                return str(r.get("summary", ""))[:300]
        if stage == "tests_failed":
            for p in sorted(d.glob("gate-*.json"), reverse=True):
                try:
                    rows = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                bad = [r for r in rows if isinstance(r, dict) and not r.get("passed")]
                if bad:
                    return f"`{bad[0].get('command')}` failed: {_last_error(bad[0].get('tail', ''))}"[:300]
    return _last_error(fallback) or CRIME.get(stage, stage)


def exhibit(diff, stage, max_lines=24, max_chars=1600):
    """The offending lines of a diff: what was added, or for weakened tests what was removed."""
    want = "-" if stage == "weakened_tests" else "+"
    keep, path = [], None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            path = line.split(" b/", 1)[-1]
        elif line.startswith(("--- ", "+++ ", "index ", "@@", "new file", "deleted file")):
            continue
        elif line.startswith(want):
            if path:
                keep.append(f"# {path}")
                path = None
            keep.append(line)
            if len(keep) >= max_lines:
                keep.append("# …")
                break
    return "\n".join(keep)[:max_chars]


def is_breakthrough(r, scores, history, min_history=8):
    """A judged breakthrough with a top score, or a reward far above accepted work so far."""
    scores = scores or {}
    if scores.get("breakthrough") and max(scores.get("impact", 0), scores.get("creativity", 0)) >= 8:
        return True
    if len(history) >= min_history:
        mu, sd = statistics.fmean(history), statistics.pstdev(history)
        return r >= 0.85 and r >= mu + 2 * sd
    return False


def parse_scores(text):
    """Judge output → scores dict, or None if it is not the requested JSON."""
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        d = json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    out = {}
    for k in WEIGHTS:
        try:
            out[k] = clamp(float(d.get(k)), 0.0, 10.0)
        except (TypeError, ValueError):
            return None
    out["breakthrough"] = d.get("breakthrough") is True
    try:                                   # which of the goal's priorities this served, or 0
        out["goal_item"] = max(0, int(d.get("goal_item", 0)))
    except (TypeError, ValueError):
        out["goal_item"] = 0
    out["why"] = str(d.get("why", ""))[:300]
    out["lesson"] = str(d.get("lesson", ""))[:300]
    out["follow_ups"] = [
        {"title": f["title"].strip()[:120], "detail": str(f.get("detail", ""))[:800]}
        for f in (d.get("follow_ups") or [])
        if isinstance(f, dict) and isinstance(f.get("title"), str) and f["title"].strip()][:2]
    return out


def json_array(text):
    """The JSON array in a model reply, or None if there is none. Tries the first-to-last
    bracket span, then every '[' in turn, so prose with brackets or markdown links before the
    array does not hide it. Prefers a non-empty array of objects; an empty one means "nothing"."""
    s, e = text.find("["), text.rfind("]")
    if s < 0 or e <= s:
        return None
    try:
        v = json.loads(text[s:e + 1])
        if isinstance(v, list):
            return v
    except json.JSONDecodeError:
        pass
    decoder, empty = json.JSONDecoder(), None
    for m in re.finditer(r"\[", text):
        try:
            v, _ = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return v
        if v == [] and empty is None:
            empty = v
    return empty


class Ledger:
    """Everything the swarm has learned about one target repo, in one JSON file."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @contextmanager
    def txn(self, write=True):
        with self._lock, open(self.path.with_suffix(".lock"), "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                d = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                d = {}
            for k, v in (("arms", {}), ("lessons", []), ("accepted", []), ("cooldown", {}),
                         ("shame", [])):
                d.setdefault(k, v)
            yield d
            if write:
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(d, indent=1))
                tmp.replace(self.path)

    def snapshot(self):
        with self.txn(write=False) as d:
            return json.loads(json.dumps(d))

    # ------------------------------------------------------------ bandits

    def pick(self, role, arms, exclude=(), rng=random):
        """Thompson sample among arms that are not excluded or cooling down."""
        now = time.time()
        with self.txn(write=False) as d:
            table, cool = d["arms"].get(role, {}), d["cooldown"]
        live = [a for a in arms if a not in exclude and cool.get(a, {}).get("until", 0) <= now]
        if not live:
            return None
        return max(live, key=lambda a: rng.betavariate(table.get(a, {}).get("a", PRIOR),
                                                        table.get(a, {}).get("b", PRIOR)))

    def update(self, role, arm, r, weight=1.0):
        with self.txn() as d:
            s = d["arms"].setdefault(role, {}).setdefault(
                arm, {"a": PRIOR, "b": PRIOR, "n": 0, "total": 0.0})
            s["a"] = PRIOR + DISCOUNT * (s["a"] - PRIOR) + weight * r
            s["b"] = PRIOR + DISCOUNT * (s["b"] - PRIOR) + weight * (1 - r)
            s["n"] += 1
            s["total"] += r
            s["last"] = time.time()

    def cool(self, model, reason="", busy=False, permanent=False):
        """Rest a model whose provider is failing. Not a quality judgement.

        busy: rate-limited upstream, which clears in minutes; it rests the model briefly and
        leaves its outage strikes alone. permanent: this key cannot use the model at all, so it
        rests for the maximum instead of being redrawn every half hour. Failures older than
        COOLDOWN_MAX (e.g. from an earlier run, since the ledger persists) are forgotten rather
        than escalating a new rest."""
        now = time.time()
        with self.txn() as d:
            c = d["cooldown"].setdefault(model, {"strikes": 0})
            if now - c.get("last", c.get("until", 0)) > COOLDOWN_MAX:
                c["strikes"], c["busy"] = 0, 0
            c["last"] = now
            if permanent:
                c["permanent"] = True
                rest = COOLDOWN_MAX
            elif busy:
                c["busy"] = c.get("busy", 0) + 1
                rest = min(BUSY_MAX, BUSY_COOLDOWN * 2 ** (c["busy"] - 1))
            else:
                c["strikes"] = c.get("strikes", 0) + 1
                rest = min(COOLDOWN_MAX, COOLDOWN_BASE * 2 ** (c["strikes"] - 1))
            c["until"] = max(c.get("until", 0), now + rest)
            c["reason"] = reason[:200]
            return c["until"]

    def resting(self):
        """Models still resting, soonest back first: [(model, until, reason)]."""
        now = time.time()
        with self.txn(write=False) as d:
            rows = [(m, c.get("until", 0), c.get("reason", "")) for m, c in d["cooldown"].items()]
        return sorted((r for r in rows if r[1] > now), key=lambda r: r[1])

    def warm(self, model):
        with self.txn(write=False) as d:
            if model not in d["cooldown"]:
                return
        with self.txn() as d:
            d["cooldown"].pop(model, None)

    def leaderboard(self):
        with self.txn(write=False) as d:
            faults = {}
            for s in d["shame"]:
                faults[s["model"]] = faults.get(s["model"], 0) + 1
            rows = [{"role": role, "arm": arm, "pulls": s["n"],
                     "mean": round(s["total"] / max(1, s["n"]), 3),
                     "posterior": round(s["a"] / (s["a"] + s["b"]), 3),
                     "faults": faults.get(arm, 0) if role == "implementer" else 0}
                    for role, table in d["arms"].items() for arm, s in table.items()]
        return sorted(rows, key=lambda x: (x["role"], -x["posterior"]))

    # ------------------------------------------------------------ the rafters

    def hang(self, model, task, stage, what, code="", artifact=""):
        """Put faulty code on the wall, under its author's name, for every later agent to see."""
        with self.txn() as d:
            sid = f"S{uuid.uuid4().hex[:8]}"
            d["shame"].append({"id": sid, "kind": "shame", "t": time.time(), "model": model,
                               "task": task.get("id"), "title": task.get("title", ""),
                               "stage": stage, "text": " ".join(str(what).split())[:300],
                               "exhibit": code, "artifact": artifact, "n": 0, "mean": 0.0,
                               "pinned": False, "created": time.time()})
            d["shame"] = d["shame"][-200:]
            return sid

    def wall(self, k=3, window=12):
        """The worst of the recent exhibits: cheating first, then the newest."""
        with self.txn(write=False) as d:
            recent = d["shame"][-window:]
        return sorted(recent, key=lambda s: (PENALTY_WEIGHT.get(s["stage"], 1.0), s["t"]),
                      reverse=True)[:k]

    # ------------------------------------------------------------ playbook

    def add_lesson(self, text, kind, source, score, pinned=False):
        text = " ".join(str(text).split())[:300]
        if len(text) < 12:
            return None
        with self.txn() as d:
            for l in d["lessons"]:
                if l["kind"] == kind and similarity(l["text"], text) > 0.6:
                    l["pinned"] = l["pinned"] or pinned
                    l["seen"] = l.get("seen", 1) + 1
                    return l["id"]
            lid = f"L{uuid.uuid4().hex[:8]}"
            d["lessons"].append({"id": lid, "text": text, "kind": kind, "source": source,
                                 "mean": score, "n": 0, "pinned": pinned, "created": time.time()})
            if len(d["lessons"]) > 300:
                keep = sorted(d["lessons"], key=lambda l: (l["pinned"], _rank(l)), reverse=True)[:250]
                d["lessons"] = keep
            return lid

    def playbook(self, k_lessons=6, k_pitfalls=4, max_pinned=3, k_shame=3):
        """(lessons, pitfalls). Pitfalls end with the wall of shame (kind "shame"), so
        every prompt that shows the playbook also shows the faulty code on the rafters."""
        with self.txn(write=False) as d:
            ls = [l for l in d["lessons"] if l["kind"] == "lesson"]
            ps = [l for l in d["lessons"] if l["kind"] == "pitfall"]
        pinned = sorted((l for l in ls if l["pinned"]), key=_rank, reverse=True)[:max_pinned]
        rest = sorted((l for l in ls if l not in pinned), key=_rank, reverse=True)
        lessons = (pinned + rest)[:max(k_lessons, len(pinned))]
        pitfalls = sorted(ps, key=lambda l: (_rank(l), l["created"]), reverse=True)[:k_pitfalls]
        return lessons, pitfalls + self.wall(k_shame)

    def credit(self, ids, r):
        """Each lesson's mean reward, counting its origin score as one observation."""
        if not ids or r is None:
            return
        ids = set(ids)
        with self.txn() as d:
            for l in d["lessons"]:
                if l["id"] in ids:
                    l["n"] += 1
                    l["mean"] += (r - l["mean"]) / (l["n"] + 1)

    # ------------------------------------------------------------ history

    def record_accepted(self, entry):
        with self.txn() as d:
            d["accepted"].append({**entry, "t": time.time()})
            d["accepted"] = d["accepted"][-400:]

    def accepted_texts(self, n=200):
        with self.txn(write=False) as d:
            return [f"{a['title']} {a.get('detail', '')}" for a in d["accepted"][-n:]]

    def accepted_rewards(self, n=50):
        with self.txn(write=False) as d:
            return [a["reward"] for a in d["accepted"][-n:]]

    def breakthroughs(self, n=5):
        with self.txn(write=False) as d:
            return [a for a in d["accepted"] if a.get("breakthrough")][-n:]


def _rank(lesson):
    # Optimism for advice that has rarely been tried, so new lessons get a hearing.
    return lesson["mean"] + 0.3 / math.sqrt(1 + lesson["n"])


def exhibit_files(text):
    """The files an exhibit's diff is about, from the `# path` lines exhibit() writes."""
    return {m.group(1) for m in re.finditer(r"^# (\S+)$", text or "", re.M)}


def format_playbook(lessons, pitfalls, paths=(), max_diff=600):
    """The playbook for one task's prompt.

    `paths` are the files this task touches. The wall of shame was 4.1 KB of diffs from
    unrelated tasks in every prompt, so the names stay — that is the deterrent — and at most one
    diff is shown, only when it is about a file this task is going to touch."""
    shame = [p for p in pitfalls if p.get("kind") == "shame"]
    pitfalls = [p for p in pitfalls if p.get("kind") != "shame"]
    paths = set(paths or ())
    relevant = next((s for s in shame
                     if s.get("exhibit") and paths & exhibit_files(s["exhibit"])), None)
    out = []
    if shame:
        out.append("HUNG FROM THE RAFTERS — faulty code agents shipped in this repository, kept "
                   "under their names as a warning to everyone. Study it. Do not be the next exhibit:")
        for s in shame:
            out.append(f"- `{s['model']}` {CRIME.get(s['stage'], s['stage'])} on '{s['title']}': {s['text']}")
            if s is relevant:
                diff = s["exhibit"][:max_diff]
                out.append("  ```diff\n" + "\n".join("  " + l for l in diff.splitlines()) + "\n  ```")
    if lessons:
        out.append("PLAYBOOK — lessons this swarm earned in this repository:")
        out += [f"- {'★ ' if l['pinned'] else ''}{l['text']}" for l in lessons]
    if pitfalls:
        out.append("PITFALLS — recent reasons work here was rejected:")
        out += [f"- {l['text']}" for l in pitfalls]
    return "\n".join(out)
