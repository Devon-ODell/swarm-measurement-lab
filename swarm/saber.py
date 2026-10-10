#!/usr/bin/env python3
"""SABER — the three-axis measurement loop for the swarm itself.

QUICKNESS, ACCURACY, SPEND: every run through this script measures all three
on the same task battery, through the same flint harness the swarm uses, and
appends the result to a SQLite time series so "is the harness getting faster /
more accurate / cheaper" is a trend line instead of a feeling.

Uses swarm/bench_swarm.py's battery and runner directly (the same tasks, the
same verifiers, the same FLINT_CHARGE_FILE spend capture) so there is exactly
one definition of "well-specified task" in this project.

Usage:
    python3 swarm/saber.py --models a,b,c [--tasks 1,2,3] [--rounds 12] [--retain]
    python3 swarm/saber.py --trend [--runs 10] [--json]

    --trend   print the last N runs from the history database (no model calls)
    --models  comma-separated model slugs (the ones the swarm actually routes to)
    --tasks   comma-separated 1..N tasks from the bench battery
    --dry     print the battery and exit without spending
"""
import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_swarm  # noqa: E402  (battery + run_one, the shared definitions)

DB = Path(__file__).resolve().parent / "saber_history.sqlite"
NOW = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")  # noqa: E731

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    models      TEXT NOT NULL,
    tasks       TEXT NOT NULL,
    rounds      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    run_id  TEXT NOT NULL,
    model   TEXT NOT NULL,
    task    TEXT NOT NULL,
    ok      INTEGER NOT NULL,
    secs    REAL NOT NULL,
    charges REAL,            -- NULL when spend could not be measured (e.g. timeout)
    UNIQUE(run_id, model, task)
);
"""

HELP = """\n--- three-axis cheat sheet -------------------------------------------------
  ACCURACY  = ok / total per model per run
  QUICKNESS = mean wall-seconds per task (includes 429 backoff; that is the
              latency a user actually feels)
  SPEND     = sum(charges) per run; NULL-aware (a timeout has no spend)
  Trend     = same numbers rolled up per model across the last --runs runs
--------------------------------------------------------------------------------"""


def connect(db: Path = DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.executescript(SCHEMA)
    return conn


def record_run(rows, models, tasks, rounds):
    """Append one benchmark run as a time series. Returns the run_id."""
    conn = connect()
    run_id = f"{int(time.time() * 1000)}"
    started = NOW()
    try:
        conn.execute(
            "INSERT INTO runs (run_id, started_at, models, tasks, rounds) VALUES (?,?,?,?,?)",
            (run_id, started, ",".join(models), ",".join(tasks), rounds),
        )
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO results (run_id, model, task, ok, secs, charges)"
                " VALUES (?,?,?,?,?,?)",
                (run_id, r["model"], r["task"], 1 if r["ok"] else 0,
                 r["secs"], r["charges"]),
            )
        conn.commit()
        return run_id
    finally:
        conn.close()


def fmt_num(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def trend(runs=10, as_json=False):
    """Roll up the last `runs` runs per model: accuracy, quickness, spend."""
    conn = connect()
    try:
        run_ids = [r[0] for r in conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT ?", (runs,))]
        if not run_ids:
            print("no saber history yet — run `saber.py --models ...` first")
            return 1
        qmarks = ",".join("?" * len(run_ids))
        rows = conn.execute(
            f"SELECT model, COUNT(*) n, SUM(ok) acc, AVG(secs) secs, SUM(charges) spend"
            f" FROM results WHERE run_id IN ({qmarks}) GROUP BY model ORDER BY acc DESC",
            run_ids,
        ).fetchall()
    finally:
        conn.close()

    if as_json:
        print(json.dumps([
            {"model": m, "n": n, "accuracy": a / n, "mean_secs": s, "spend": sp}
            for m, n, a, s, sp in rows
        ], indent=2))
        return 0

    print(f"\nsaber trend — last {len(run_ids)} run(s), "
          f"{NOW()} UTC  (axes: QUICKNESS / ACCURACY / SPEND)")
    print(f"{'model':<46} {'n':>4} {'acc':>7} {'mean_s':>7} {'spend':>9}")
    print("-" * 80)
    for m, n, acc, secs, spend in rows:
        print(f"{m:<46} {n:>4} {fmt_num(100.0 * acc / n):>6}% "
              f"{fmt_num(secs):>7} {fmt_num(spend, 3):>9}")
    print(HELP)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", help="comma-separated model slugs")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--retain", action="store_true")
    ap.add_argument("--trend", action="store_true")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.trend:
        return trend(a.runs, a.json)

    if not a.models:
        ap.error("--models is required unless you want --trend")
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    tasks = (bench_swarm.TASKS if not a.tasks
             else [bench_swarm.TASKS[int(i) - 1] for i in a.tasks.split(",")])
    if a.dry:
        print(f"saber dry run — {len(tasks)} task(s) x {len(models)} model(s)")
        for i, t in enumerate(tasks, 1):
            print(f"  {i}. {t['name']}: {t['detail']}")
        print("models:", ", ".join(models))
        return 0

    results = []
    for t in tasks:
        for m in models:
            print(f"== {m} / {t['name']} ==", flush=True)
            r = bench_swarm.run_one(m, t, rounds=a.rounds, timeout=a.timeout,
                                    retain=a.retain)
            results.append(r)
            ok = "PASS" if r["ok"] else "FAIL"
            charged = r["charges"] if r["charges"] is not None else float("nan")
            print(f"  {ok}  {r['secs']}s  ${charged:.4f}")

    run_id = record_run(results, models, [t["name"] for t in tasks], a.rounds)
    print(f"\nsaber: {len(results)} result(s) recorded as run {run_id} -> {DB}")
    return trend(1)


if __name__ == "__main__":
    sys.exit(main())