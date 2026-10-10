"""Tracking + aggregate reporting for consensus-of-agents runs.

Reads the append-only JSONL at consensus/runs/runs.jsonl written by
consensus_runner.py and renders:
  * per-task + overall accuracy for the panel (majority) and per panelist
  * per-task + overall cost (USD) and wall-time
  * the frontier (openai/gpt-5.5) raw answers when captured
  * a Markdown table block the paper can paste directly
The log is never rewritten: `report` only reads.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "consensus" / "runs" / "runs.jsonl"
KEPT = ROOT / "consensus" / "runs" / "kept"


def load(runs_path=RUNS):
    recs = []
    if not runs_path.exists():
        return recs
    for line in runs_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"skip malformed line: {line[:80]!r}", file=sys.stderr)
    return recs


def aggregate(recs):
    """Return dict of per-task and overall stats from raw records.

    If a task appears more than once (re-runs), the LAST record for that task
    wins, so the aggregate never double-counts a task. Records carry a `suite`
    tag (code | translation); dedup is per (suite, task)."""
    # collapse to one record per (suite, task) (last wins)
    latest = {}
    for rec in recs:
        latest[(rec.get("suite", "code"), rec["task"])] = rec
    recs = list(latest.values())
    tasks = defaultdict(lambda: {"suite": "code", "panel_ok": 0, "n": 0, "panel_secs": [],
                                 "panel_cost": [], "agents": [],
                                 "frontier": None, "frontier_cost": 0.0,
                                 "frontier_ok": None, "frontier_secs": None})
    for rec in recs:
        t = rec["task"]
        d = tasks[t]
        d["suite"] = rec.get("suite", "code")
        d["n"] = rec.get("n", len(rec.get("models", [])))
        d["panel_ok"] += int(rec.get("majority_ok", False))
        d["panel_secs"].append(sum(a.get("secs") or 0 for a in rec.get("agents", [])))
        d["panel_cost"].append(sum(a.get("charges") or 0 for a in rec.get("agents", [])))
        d["agents"].append(rec.get("agents", []))
        fr = rec.get("raw_frontier")
        if fr:
            d["frontier_cost"] += fr.get("charges") or 0
            d["frontier_ok"] = fr.get("ok")
            d["frontier_secs"] = fr.get("secs")
    overall = {
        "n_tasks": len(tasks),
        "panel_pass": sum(d["panel_ok"] for d in tasks.values()),
        "panel_n": len(tasks),
        "panel_secs_mean": statistics.mean([sum(v["panel_secs"]) / max(1, len(v["panel_secs"]))
                                            for v in tasks.values()]) if tasks else 0.0,
        "panel_cost_mean": statistics.mean([sum(v["panel_cost"]) / max(1, len(v["panel_cost"]))
                                            for v in tasks.values()]) if tasks else 0.0,
        "frontier_ok": sum(1 for v in tasks.values() if v["frontier_ok"]),
        "frontier_total": sum(1 for v in tasks.values() if v["frontier_ok"] is not None),
        "frontier_cost_mean": statistics.mean([v["frontier_cost"] for v in tasks.values()
                                               if v["frontier_cost"] > 0]) if tasks else 0.0,
        "frontier_secs_mean": statistics.mean([v["frontier_secs"] for v in tasks.values()
                                               if v["frontier_secs"] is not None]) if tasks else 0.0,
    }
    return dict(tasks), overall


def render_markdown(recs):
    tasks, ov = aggregate(recs)
    lines = []
    lines.append("| task | suite | panel | frontier | panel $ | panel s | frontier $ |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, d in sorted(tasks.items()):
        panel_r = "pass" if d["panel_ok"] else "fail"
        fr_ok = "pass" if d["frontier_ok"] else ("fail" if d["frontier_ok"] is False else "—")
        lines.append(
            f"| {name} | {d['suite']} | {panel_r} | {fr_ok} | "
            f"${sum(d['panel_cost']):.4f} | {sum(d['panel_secs']):.0f}s | "
            f"${d['frontier_cost']:.4f} |")
    lines.append(f"| **overall** | — | **{ov['panel_pass']}/{ov['panel_n']}** | "
                 f"**{ov['frontier_ok']}/{ov['frontier_total']}** | "
                 f"**${ov['panel_cost_mean']:.4f}/task** | "
                 f"**{ov['panel_secs_mean']:.0f}s/task** | "
                 f"**${ov['frontier_cost_mean']:.4f}/task** |")
    return "\n".join(lines)


def render_per_panelist(recs):
    by = defaultdict(lambda: {"ok": 0, "n": 0, "secs": [], "cost": []})
    for rec in recs:
        for a in rec.get("agents", []):
            m = a["model"]
            by[m]["n"] += 1
            by[m]["ok"] += int(bool(a.get("ok")))
            if a.get("secs") is not None:
                by[m]["secs"].append(a["secs"])
            if isinstance(a.get("charges"), (int, float)):
                by[m]["cost"].append(a["charges"])
    lines = ["| panelist | acc | n | mean s | mean $ |", "|---|---|---|---|---|"]
    for m, d in sorted(by.items(), key=lambda kv: -kv[1]["ok"]):
        lines.append(f"| {m.split('/')[-1]} | {d['ok']}/{d['n']} | {d['n']} | "
                     f"{statistics.mean(d['secs']):.1f} | "
                     f"${statistics.mean(d['cost']):.4f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="dump full aggregate as JSON")
    ap.add_argument("--pretty", action="store_true", help="pretty per-task detail")
    a = ap.parse_args()
    recs = load()
    if not recs:
        print("no runs yet — run consensus_runner.py first", file=sys.stderr)
        return
    tasks, ov = aggregate(recs)
    if a.json:
        print(json.dumps({"overall": ov, "tasks": {k: {"panel_ok": v["panel_ok"], "n": v["n"],
                                                       "panel_cost": sum(v["panel_cost"]),
                                                       "panel_secs": sum(v["panel_secs"]),
                                                       "frontier_cost": v["frontier_cost"],
                                                       "frontier_ok": v["frontier_ok"],
                                                       "frontier_secs": v["frontier_secs"]}
                                                for k, v in tasks.items()}}, indent=2))
        return
    print("== overall ==")
    print(f"panel: {ov['panel_pass']}/{ov['panel_n']} tasks pass via majority")
    print(f"panel cost: ${ov['panel_cost_mean']:.4f}/task, wall {ov['panel_secs_mean']:.0f}s/task")
    print(f"frontier (raw): {ov['frontier_ok']}/{ov['frontier_total']} pass, "
          f"${ov['frontier_cost_mean']:.4f}/task, {ov['frontier_secs_mean']:.0f}s/task")
    print("\n== per panelist ==")
    print(render_per_panelist(recs))
    print("\n== markdown table ==")
    print(render_markdown(recs))
    if a.pretty:
        print("\n== per-task detail ==")
        for name, d in sorted(tasks.items()):
            print(f"  {name}: panel {d['panel_ok']}/{d['n']}, "
                  f"$ {sum(d['panel_cost']):.4f}, {sum(d['panel_secs']):.0f}s; "
                  f"frontier ok={d['frontier_ok']}, ${d['frontier_cost']:.4f}")


if __name__ == "__main__":
    main()