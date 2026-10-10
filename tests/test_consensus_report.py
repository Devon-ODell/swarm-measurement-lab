"""Tests for the consensus-of-agents tracking/report logic."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consensus"))
import report


def _rec(task, agents, frontier_ok=None, suite="code"):
    return {
        "task": task, "n": 3, "majority": sum(1 for a in agents if a["ok"]),
        "majority_ok": sum(1 for a in agents if a["ok"]) >= 2,
        "suites": suite and [suite],
        "suite": suite,
        "models": [a["model"] for a in agents],
        "agents": agents,
        "raw_frontier": {"ok": frontier_ok, "secs": 5, "charges": 0.04}
        if frontier_ok is not None else None,
    }


def _agent(model, ok, secs=10, costs=0.0):
    return {"model": model, "ok": ok, "secs": secs, "charges": costs, "rc": 0}


def test_aggregate_dedup_last_wins(tmp_path):
    p = tmp_path / "runs.jsonl"
    r1 = _rec("a", [_agent("m1", True), _agent("m2", True), _agent("m3", False)], True)
    r2 = _rec("a", [_agent("m1", False), _agent("m2", False), _agent("m3", False)], False)
    p.write_text("\n".join(json.dumps(x) for x in [r1, r2]) + "\n")
    recs = report.load(p)
    tasks, _ = report.aggregate(recs)
    assert len(tasks) == 1           # collapsed by task, last wins
    assert tasks["a"]["panel_ok"] == 0  # r2 (FAIL) overrides r1 (PASS)


def test_aggregate_counts_majority():
    recs = [
        _rec("x", [_agent("m1", True), _agent("m2", True), _agent("m3", False)], False),
        _rec("y", [_agent("m1", True), _agent("m2", False), _agent("m3", False)], True),
    ]
    tasks, ov = report.aggregate(recs)
    assert ov["panel_pass"] == 1
    assert ov["panel_n"] == 2
    assert tasks["x"]["panel_ok"] == 1
    assert tasks["y"]["panel_ok"] == 0
    assert ov["frontier_ok"] == 1       # x fails, y passes frontier
    assert ov["frontier_total"] == 2


def test_render_markdown_no_dup():
    recs = [_rec("x", [_agent("m1", True), _agent("m2", True), _agent("m3", True)], True),
            _rec("x", [_agent("m1", True), _agent("m2", True), _agent("m3", True)], True)]
    md = report.render_markdown(recs)
    assert md.count("| x |") == 1       # one row per task even with dup records


def test_per_panelist_accuracy():
    recs = [_rec("x", [_agent("m1", True, 5), _agent("m1", False, 7)], True)]
    by = report.render_per_panelist(recs)
    assert "m1" in by
    assert "1/2" in by                  # m1 passes 1 of 2 agent-runs