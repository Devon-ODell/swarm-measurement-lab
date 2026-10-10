"""```text
consensus/lab/lab.py — the self-improving benchmark lab.

Purpose
-------
Turns the consensus-of-agents benchmark into a REPEATABLE, COMPARABLE scientific
loop (the "recursive testing" the owner asked for):

    hypothesis → experiment → measurement → analysis → next hypothesis

Every run is an EPOCH: a named batch (default: auto timestamp) over a chosen
suite (code / translation / both), with a chosen panel strategy, logged to
consensus/lab/runs/<epoch>.jsonl. Epochs are append-only and self-contained
(each line carries family/epoch/variant + per-agent inference trace), so any
two epochs can be diffed later as an A/B study with a paired per-task table.

The three main tests (the owner's "three tests"):
  T1 REPEATABILITY  — same epoch config twice (or N seeds) must produce the
                      same pass/fail per (task, agent) (deterministic seeds,
                      fresh worktrees, same battery).
  T2 ACCURACY       — a variant (weighted votes, debate round, better prompt,
                      different panel) must improve per-task pass rate vs the
                      recorded baseline epoch with no accuracy regression.
  T3 LATENCY/COST   — a variant must cut wall-time or cost per task without
                      dropping accuracy (measured deltas, not vibes).

Inference points tracked per agent call (the "inference points" the owner
asked to keep counting):
  * wall seconds (whole flint session and per flint run when the session is
    observed live), requests/rounds actually made, tools called, study reads,
  * prompt/completion tokens and USD when flint's charge file exists,
  * flint exit code (fast-fail codes 3..10) and their meanings,
  * rate-limit / provider-error / retry markers parsed from stderr,
  * whether the run timed out or was killed,
  * outcome of the deterministic verifier (the ground truth).

The lab NEVER edits flint or the harness config to "improve" a number. An
improvement is a *config/prompt/variant change supplied by the scientist*; the
lab measures it and reports the delta honestly, including regressions.

Usage (see lab/README.md for the full protocol):
    .venv/bin/python consensus/lab/lab.py --suite code --epoch my-first --dry
    .venv/bin/python consensus/lab/lab.py --suite code --epoch my-first \
        --variant weighted --seeds 3 --rounds 12 --timeout 600
    .venv/bin/python consensus/lab/lab.py --suite translation \
        --variant majority --seeds 1
    .venv/bin/python consensus/lab/lab.py diff --base <epoch> --head <epoch>
    .venv/bin/python consensus/lab/lab.py stats [--epoch E]
    .venv/bin/python consensus/lab/lab.py traces --epoch E --plain
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "swarm"))

import bench_swarm  # noqa: E402  (task battery + verifiers + FLINT runner)
from translation_tasks import TRANSLATION_TASKS  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

RUNS = ROOT / "consensus" / "lab" / "runs"

# --------------------------------------------------------------------- config
# Panel: three small free models, well separated by vendor/family so their
# errors are as independent as a free tier allows (same default as the paper).
PANEL = [
    "qwen/qwen3.8-27b:free",
    "cohere/north-mini-code:free",
    "poolside/laguna-xs-2.1:free",
]
FRONTIER = "openrouter:openai/gpt-5.5"  # force the OpenRouter backend (bare openai/ -> api.openai.com)

# flint/OpenRouter fast-fail exit codes -> human meaning (mirrors swarm/bridge.py)
EXIT_MEANING = {
    3: "daily free-request cap reached",
    4: "credit or account limit",
    5: "ran out of steps",
    6: "budget pause",
    7: "provider unavailable",
    8: "model busy (rate-limited upstream)",
    9: "model not available to this API key",
    10: "paid allowance exhausted",
}

# markers parsed from flint stderr; counted per agent call
_RATE_LIMIT = re.compile(r"rate[- ]limit|per-minute|pacing|retry", re.I)
_PROVIDER_ERR = re.compile(r"provider unavailable|upstream|5\d\d|429|temporarily", re.I)
_TOOL_LINE = re.compile(r"^tool: (\S+)", re.M)
_ROUND_LINE = re.compile(r"^round (\d+)/", re.M)
_STUDY_LINE = re.compile(r"^tool: study$", re.M)


# ----------------------------------------------------------------- helpers
def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def suites():
    return {"code": bench_swarm.TASKS, "translation": TRANSLATION_TASKS}


def default_epoch():
    return time.strftime("epoch-%Y%m%d-%H%M%S")


def _fresh_workdir(prefix="lab-work-"):
    return Path(tempfile.mkdtemp(prefix=prefix))


def _env_for(rounds):
    return dict(os.environ, FLINT_MAX_STEPS=str(rounds))


# ------------------------------------------------------------ trace parsing
def parse_trace(stderr, stdout="", wall=None, charge_file=None, rc=None,
                timeouts=None, timed_out=False):
    """Turn one flint session's captured streams + charge file into a compact
    inference-trace dict. Every field is a measured or parsed point; nothing
    here invents numbers."""
    stderr = stderr or ""
    trace = {
        "wall": round(wall, 1) if wall is not None else None,
        "timed_out": bool(timed_out),
        "timeouts": timeouts or [],
        "rc": rc,
        "exit_meaning": EXIT_MEANING.get(rc) if rc is not None else None,
        "rounds": len(_ROUND_LINE.findall(stderr)),
        "tools": sorted(set(_TOOL_LINE.findall(stderr))),
        "tool_calls": len(_TOOL_LINE.findall(stderr)),
        "study_calls": len(_STUDY_LINE.findall(stderr)),
        "rate_limited": bool(_RATE_LIMIT.search(stderr)),
        "provider_error": bool(_PROVIDER_ERR.search(stderr)),
        "tokens": None,
        "usd": None,
    }
    if charge_file and Path(charge_file).exists():
        try:
            rows = [json.loads(l) for l in Path(charge_file).read_text().splitlines() if l.strip()]
            trace["usd"] = round(sum(r.get("usd") or 0 for r in rows), 6)
        except (OSError, ValueError):
            pass  # a broken charge file must never fail the lab
    return trace


# ------------------------------------------------------------- flint runners
def run_agent_flint(task, model, rounds=12, timeout=600, retain=False,
                    env_extra=None, timeout_grace=30):
    """One complete independent answer via flint in a fresh worktree.

    Mirrors bench_swarm.run_one (same flint args, same FLINT_CHARGE_FILE +
    FLINT_MAX_STEPS env, same verifier) and captures a full inference trace
    from the captured stdout/stderr + charge file."""
    work = _fresh_workdir()
    try:
        for name, body in task["seeded"].items():
            p = work / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
        charge = work.parent / f"charges-{int(time.time()*1000)}.jsonl"
        env = dict(os.environ, FLINT_CHARGE_FILE=str(charge), FLINT_MAX_STEPS=str(rounds))
        if env_extra:
            env.update(env_extra)
        t0 = time.time()
        timed_out = False
        timeouts = []
        try:
            r = subprocess.run(
                [*bench_swarm.FLINT, "-p", task["prompt"], "-m", model, "--yolo", "-C", str(work)],
                capture_output=True, text=True, timeout=timeout + timeout_grace, env=env)
            out, err, rc = r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            out, err, rc = e.stdout or "", e.stderr or "", None
            timed_out, timeouts = True, [timeout]
        wall = time.time() - t0
        ok, detail = task["verify"](work), ""
        kept = None
        if not ok and retain:
            kept = RUNS / "kept" / f"{model.replace('/', '_')}__{task['name']}"
            kept.parent.mkdir(parents=True, exist_ok=True)
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(work, kept)
        trace = parse_trace(err, out, wall=wall, charge_file=charge, rc=rc,
                            timeouts=timeouts, timed_out=timed_out)
        return {
            "model": model, "ok": ok, "detail": detail or ("TIMEOUT" if timed_out else ""),
            "rc": rc, "secs": round(wall, 1), "out_tail": out.strip()[-200:],
            "err_tail": err.strip()[-200:], "charges": trace.get("usd") or 0.0,
            "trace": trace, "kept": str(kept) if kept else None,
        }
    except Exception as e:  # noqa: BLE001 — report, never crash the batch
        return {"model": model, "ok": False, "detail": f"{type(e).__name__}: {str(e)[:200]}",
                "rc": None, "secs": round(time.time() - t0, 1), "out_tail": "",
                "err_tail": "", "charges": 0.0, "trace": {"wall": round(time.time() - t0, 1),
                "timed_out": False, "timeouts": [], "rc": None, "rounds": 0, "tools": [],
                "tool_calls": 0, "study_calls": 0, "rate_limited": False,
                "provider_error": True, "tokens": None, "usd": 0.0}, "kept": None}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_frontier_flint(task, rounds=12, timeout=600, retain=False):
    """Frontier baseline through the identical flint worktree agent."""
    return run_agent_flint(task, FRONTIER, rounds=rounds, timeout=timeout, retain=retain)


# ------------------------------------------------------------ panel variants
def panel_result(agents, variant="majority", weights=None):
    """Derive the panel's verdict + payoffs for any variant.

    Variants:
      majority — majority vote of independent ok signals (paper default).
      weighted — weighted vote by per-model historical accuracy (weights given
                 or uniform); weight must be >=0, an agent with 0 is only
                 counted if unanimous.
      top1     — single best expected model (cheapest accurate subset): pick
                 the first agent that passes, else the first agent (fallback
                 reproduces "ask one model" latency).
      debate   — majority vote, but a "debate round" flag is carried for the
                 runner to schedule (the lab runner runs the debate loop).

    Returns (decided_ok, votes, n, variant, weights).
    """
    votes = [bool(a.get("ok")) for a in agents]
    n = len(agents)
    if variant == "top1":
        decided = bool(agents[0].get("ok")) if agents else False
        return decided, int(decided), 1, variant, None
    if variant == "weighted":
        w = weights or [1.0] * n
        if len(w) != n:
            w = [1.0] * n
        total = sum(w[i] for i in range(n) if votes[i])
        weight_sum = sum(w)
        decided = (total >= weight_sum / 2) if weight_sum else False
        return decided, int(decided), n, variant, w
    # majority (and debate-before-majority share the same vote rule)
    decided = sum(votes) >= (n + 1) // 2
    return decided, int(decided), n, variant, None


def run_debate_round(task, agents, rounds=8, timeout=300):
    """Proposed v2: after independent answers, each agent sees the others'
    answers (fresh copies only — no shared context) and revises. Kept as a
    pure measurement: the *revised* verdict is the debate outcome."""
    summaries = []
    for a in agents:
        if a.get("ok"):
            summaries.append(f"- {a['model']} passed the verifier.")
        else:
            tail = (a.get("err_tail") or "").strip().splitlines()[-2:]
            summaries.append(f"- {a['model']} failed: {'; '.join(tail)[:200]}")
    debate_prompt = (
        "You are the REVISER in a consensus panel. Your peers produced these "
        "independent results on the same task:\n" + "\n".join(summaries) +
        "\n\nRe-read the task, look for the mistake your peers (and possibly you) "
        "made, and produce the definitive correct answer now. Reply with: DONE."
    )
    revised = []
    for a in agents:
        work = _fresh_workdir()
        try:
            for name, body in task["seeded"].items():
                p = work / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(body)
            charge = work.parent / f"charges-{int(time.time()*1000)}.jsonl"
            env = dict(os.environ, FLINT_CHARGE_FILE=str(charge), FLINT_MAX_STEPS=str(rounds))
            t0 = time.time()
            try:
                r = subprocess.run([*bench_swarm.FLINT, "-p", debate_prompt, "-m", a["model"],
                                    "--yolo", "-C", str(work)],
                                   capture_output=True, text=True, timeout=timeout, env=env)
                rc = r.returncode
            except subprocess.TimeoutExpired:
                rc = None
            wall = time.time() - t0
            ok = bool(task["verify"](work))
            revised.append({"model": a["model"], "ok": ok, "rc": rc,
                            "secs": round(wall, 1), "detail": "" if ok else "debate revision FAIL",
                            "charges": 0.0, "trace": parse_trace("", "", wall=wall)})
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return revised


# ------------------------------------------------------------ epoch machinery
def task_specs(suite, tasks):
    pool = suites()[suite]
    return [pool[i - 1] for i in tasks] if tasks else pool


def run_epoch(suite, epoch, variant="majority", panel=None, seeds=1, rounds=12,
              timeout=600, retain=False, dry=False, debate=False, tasks=None,
              env_extra=None):
    """Run one epoch: N seeds x M models x T tasks, each with a full trace,
    logged append-only under consensus/lab/runs/<epoch>.jsonl. Returns the
    list of records written (or, in dry mode, the plan list)."""
    panel = panel or PANEL
    specs = task_specs(suite, tasks)
    rows = []
    runs_dir = RUNS
    path = runs_dir / f"{epoch}.jsonl"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if not dry:
        marker = {"epoch": epoch, "kind": "epoch-start", "ts": _now(),
                  "suite": suite, "variant": variant, "panel": panel,
                  "seeds": seeds, "rounds": rounds, "timeout": timeout,
                  "frontier": FRONTIER}
        with open(path, "a") as f:
            f.write(json.dumps(marker) + "\n")
    for si in range(1, seeds + 1):
        for ti, task in enumerate(specs, 1):
            agents = []
            for model in panel:
                if dry:
                    agents.append({"model": model, "ok": None, "detail": "dry",
                                   "rc": None, "secs": None, "out_tail": "", "err_tail": "",
                                   "charges": None, "trace": {}, "kept": None})
                    continue
                r = run_agent_flint(task, model, rounds=rounds, timeout=timeout,
                                    retain=retain, env_extra=env_extra)
                agents.append(r)
                print(f"  [{suite}/{ti}/{len(specs)} seed{si} {model.split('/')[-1]}] "
                      f"{'PASS' if r['ok'] else 'FAIL'} {r['secs']}s ${r['charges'] or 0:.4f} "
                      f"rc={r['rc']} rounds={r['trace'].get('rounds')}", flush=True)
            if not dry and debate and variant in ("majority", "debate"):
                # debate round: revise only when the first-pass majority FAILED
                # (that is the case where an extra round can change the verdict)
                first_ok = panel_result(agents, "majority")[0]
                if not first_ok:
                    revised = run_debate_round(task, agents, rounds=rounds, timeout=timeout)
                    agents = [{"**revised**": True, **a} for a in revised]
            frontier = None
            if not dry:
                frontier = run_frontier_flint(task, rounds=rounds, timeout=timeout, retain=retain)
            decided, votes, n_used, variant_used, weights = \
                panel_result(agents, variant) if not dry else (None, 0, len(panel), variant, None)
            rec = {
                "epoch": epoch, "ts": _now(), "family": f"{suite}__{variant}",
                "suite": suite, "task": task["name"], "seed": si,
                "variant": variant_used, "weights": weights,
                "n": len(panel), "votes": votes, "majority_ok": decided,
                "agents": agents, "frontier": frontier,
            }
            rows.append(rec)
            if not dry:
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  [{suite}/{ti}/{len(specs)} seed{si}] panel "
                      f"{votes}/{len(panel)} -> {'PASS' if decided else 'FAIL'} "
                      f"({variant_used})", flush=True)
    if dry:
        return rows
    with open(path, "a") as f:
        f.write(json.dumps({"epoch": epoch, "kind": "epoch-end", "ts": _now(),
                            "rows": len(rows)}) + "\n")
    return rows


# ----------------------------------------------------------- analysis/report
def load_epoch(epoch, meta=False):
    """Read one epoch file as a list of records (in order)."""
    path = RUNS / f"{epoch}.jsonl"
    if not path.exists():
        return [], None
    recs = []
    meta_rec = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("kind") in ("epoch-start", "epoch-end"):
            meta_rec = d
            continue
        recs.append(d)
    return (recs, meta_rec) if meta else (recs, None)


def epoch_list():
    if not RUNS.exists():
        return []
    return sorted(p.stem for p in RUNS.glob("*.jsonl"))


def summarize(recs, variant=None):
    """Accuracy, latency, cost, and trace aggregates over records.
    De-duplicates (seed, task) per (epoch, suite, variant) keeping the LAST
    record, then aggregates per task and overall."""
    latest = {}
    for r in recs:
        key = (r.get("suite"), r["task"], r.get("seed", 1), r.get("variant"))
        latest[key] = r
    recs = list(latest.values())
    if variant:
        recs = [r for r in recs if r.get("variant") == variant]
    if not recs:
        return {"n_tasks": 0}
    per_task = {}
    trace_agg = defaultdict(list)
    frontier = {"ok": 0, "n": 0, "secs": [], "usd": []}
    agents_agg = defaultdict(lambda: {"ok": 0, "n": 0, "secs": [], "usd": []})
    for r in recs:
        t = r["task"]
        per_task.setdefault(t, {"n_seeds": 0, "panel_pass": 0, "panel_secs": [], "panel_usd": [],
                                "frontier_ok": [], "frontier_secs": [], "agents": []})
        d = per_task[t]
        d["n_seeds"] += 1
        d["panel_pass"] += int(bool(r.get("majority_ok")))
        sum_secs = sum(a.get("secs") or 0 for a in r.get("agents", []))
        sum_usd = sum(a.get("charges") or 0 for a in r.get("agents", []))
        d["panel_secs"].append(sum_secs)
        d["panel_usd"].append(sum_usd)
        fr = r.get("frontier")
        if fr:
            d["frontier_ok"].append(int(bool(fr.get("ok"))))
            d["frontier_secs"].append(fr.get("secs") or 0)
        for a in r.get("agents", []):
            m = a.get("model")
            if not m:
                continue
            g = agents_agg[m]
            g["n"] += 1
            g["ok"] += int(bool(a.get("ok")))
            if a.get("secs") is not None:
                g["secs"].append(a["secs"])
            if isinstance(a.get("charges"), (int, float)):
                g["usd"].append(a["charges"])
            tr = a.get("trace") or {}
            for k in ("rounds", "tool_calls", "study_calls", "rate_limited",
                      "provider_error", "timed_out"):
                if tr.get(k) is not None:
                    trace_agg[k].append(tr[k])
    tasks = list(per_task.keys())
    n_passes = sum(1 for d in per_task.values() if d["panel_pass"] > 0)
    n_total = len(per_task)
    # weighted per-task pass rate across seeds: pass if panel passed in the
    # majority of seeds (or any seed when seeds==1), which mirrors the paper.
    weighted_pass = sum(1 for d in per_task.values()
                        if d["panel_pass"] / max(1, d["n_seeds"]) >= 0.5)
    overall_secs = [s for d in per_task.values() for s in d["panel_secs"]]
    overall_usd = [s for d in per_task.values() for s in d["panel_usd"]]
    return {
        "n_tasks": n_total,
        "n_records": len(recs),
        "n_seeds": max(r.get("seed", 1) for r in recs) if recs else 1,
        "tasks_pass": weighted_pass,
        "tasks_pass_majority_seed": n_passes,
        "panel_pass_rate": weighted_pass / max(1, n_total),
        "panel_secs_mean": statistics.mean(overall_secs) if overall_secs else 0.0,
        "panel_secs_sum": sum(overall_secs),
        "panel_usd_mean": statistics.mean(overall_usd) if overall_usd else 0.0,
        "panel_usd_sum": sum(overall_usd),
        "frontier_ok": sum(1 for d in per_task.values() if d["frontier_ok"] and d["frontier_ok"][-1]),
        "frontier_total": sum(1 for d in per_task.values() if d["frontier_ok"]),
        "frontier_secs_mean": statistics.mean(
            [s for d in per_task.values() for s in d["frontier_secs"]]) if any(
            d["frontier_secs"] for d in per_task.values()) else 0.0,
        "frontier_usd_sum": sum(a.get("charges") or 0 for r in recs
                                for a in ([r["frontier"]] if r.get("frontier") else [])),
        "agents": {m: {"ok": g["ok"], "n": g["n"], "secs_mean": statistics.mean(g["secs"]) if g["secs"] else 0.0,
                       "usd_mean": statistics.mean(g["usd"]) if g["usd"] else 0.0}
                   for m, g in sorted(agents_agg.items())},
        "trace": {k: (statistics.mean(v) if k not in ("rate_limited", "provider_error", "timed_out")
                      else sum(bool(x) for x in v)) for k, v in trace_agg.items()},
    }


def diff_epochs(base, head, suite=None, variant=None):
    """A/B study: per-task paired table + overall deltas between two epochs.
    Accuracy delta is per-task pass-rate over seeds (ignores seed ordering);
    latency/cost deltas are mean-per-task. Returns rows + overall dict."""
    b_recs, b_meta = load_epoch(base, meta=True)
    h_recs, h_meta = load_epoch(head, meta=True)
    if not b_recs or not h_recs:
        raise SystemExit(f"epoch {base if not b_recs else head} has no records")
    # collapse base (last wins per task+seed) to build the per-task baseline
    b_latest, h_latest = {}, {}
    for r in b_recs:
        if suite and r.get("suite") != suite:
            continue
        if variant and r.get("variant") != variant:
            continue
        b_latest[(r.get("suite"), r["task"], r.get("seed", 1))] = r
    for r in h_recs:
        if suite and r.get("suite") != suite:
            continue
        if variant and r.get("variant") != variant:
            continue
        h_latest[(r.get("suite"), r["task"], r.get("seed", 1))] = r
    all_keys = sorted(set(b_latest) | set(h_latest))
    rows = []
    by_task = defaultdict(lambda: {"b_pass": 0, "h_pass": 0, "n": 0,
                                   "b_secs": [], "h_secs": [], "b_usd": [], "h_usd": []})
    for k in all_keys:
        suite_k, task, seed = k
        b, h = b_latest.get(k), h_latest.get(k)
        d = by_task[(suite_k, task)]
        d["n"] += 1
        if b:
            d["b_pass"] += int(bool(b.get("majority_ok")))
            d["b_secs"].append(sum(a.get("secs") or 0 for a in b.get("agents", [])))
            d["b_usd"].append(sum(a.get("charges") or 0 for a in b.get("agents", [])))
        if h:
            d["h_pass"] += int(bool(h.get("majority_ok")))
            d["h_secs"].append(sum(a.get("secs") or 0 for a in h.get("agents", [])))
            d["h_usd"].append(sum(a.get("charges") or 0 for a in h.get("agents", [])))
    for (suite_k, task), d in sorted(by_task.items()):
        n = max(1, d["n"])
        rows.append({
            "suite": suite_k, "task": task,
            "base_pass": d["b_pass"], "head_pass": d["h_pass"], "seeds": d["n"],
            "base_pass_rate": d["b_pass"] / n, "head_pass_rate": d["h_pass"] / n,
            "base_secs": statistics.mean(d["b_secs"]) if d["b_secs"] else 0.0,
            "head_secs": statistics.mean(d["h_secs"]) if d["h_secs"] else 0.0,
            "base_usd": statistics.mean(d["b_usd"]) if d["b_usd"] else 0.0,
            "head_usd": statistics.mean(d["h_usd"]) if d["h_usd"] else 0.0,
        })
    b_pass = sum(1 for r in rows if r["base_pass_rate"] >= 0.5)
    h_pass = sum(1 for r in rows if r["head_pass_rate"] >= 0.5)
    b_secs = statistics.mean([r["base_secs"] for r in rows]) if rows else 0.0
    h_secs = statistics.mean([r["head_secs"] for r in rows]) if rows else 0.0
    b_usd = statistics.mean([r["base_usd"] for r in rows]) if rows else 0.0
    h_usd = statistics.mean([r["head_usd"] for r in rows]) if rows else 0.0
    overall = {
        "base_pass": b_pass, "head_pass": h_pass, "n_tasks": len(rows),
        "acc_delta": h_pass - b_pass,
        "base_secs": b_secs, "head_secs": h_secs, "secs_delta": h_secs - b_secs,
        "base_usd": b_usd, "head_usd": h_usd, "usd_delta": h_usd - b_usd,
        "regressed_tasks": [r["task"] for r in rows if r["head_pass_rate"] < r["base_pass_rate"]],
        "improved_tasks": [r["task"] for r in rows if r["head_pass_rate"] > r["base_pass_rate"]],
    }
    return rows, overall


def render_diff(rows, overall, head_meta=None):
    lines = ["| task | suite | base pass | head pass | base s | head s | base $ | head $ |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['task']} | {r['suite']} | {r['base_pass']}/{r['seeds']} | "
                     f"{r['head_pass']}/{r['seeds']} | {r['base_secs']:.0f} | {r['head_secs']:.0f} | "
                     f"${r['base_usd']:.4f} | ${r['head_usd']:.4f} |")
    d = overall
    lines.append(f"| **overall** | — | **{d['base_pass']}/{d['n_tasks']}** | "
                 f"**{d['head_pass']}/{d['n_tasks']}** | **{d['base_secs']:.0f}** | "
                 f"**{d['head_secs']:.0f}** | **${d['base_usd']:.4f}** | **${d['head_usd']:.4f}** |")
    lines.append("")
    lines.append(f"accuracy delta: {d['acc_delta']:+d} tasks  ({d['head_pass']} vs {d['base_pass']})")
    lines.append(f"latency delta:  {d['secs_delta']:+.1f}s/task  ({d['head_secs']:.0f} vs {d['base_secs']:.0f})")
    lines.append(f"cost delta:     ${d['usd_delta']:+.4f}/task (${d['head_usd']:.4f} vs ${d['base_usd']:.4f})")
    if d["regressed_tasks"]:
        lines.append(f"REGRESSIONS: {', '.join(d['regressed_tasks'])}")
    if d["improved_tasks"]:
        lines.append(f"improvements:  {', '.join(d['improved_tasks'])}")
    return "\n".join(lines)


def render_summary(s):
    lines = [f"{s.get('n_tasks', 0)} tasks × {s.get('n_seeds', 1)} seed(s), "
             f"{s['n_records']} records"]
    lines.append(f"panel pass rate: {s.get('panel_pass_rate', 0):.0%} "
                 f"({s.get('tasks_pass')}/{s.get('n_tasks', 0)} tasks over ≥50% seeds)")
    lines.append(f"panel cost:  ${s.get('panel_usd_mean', 0):.4f}/task total "
                 f"${s.get('panel_usd_sum', 0):.4f}")
    lines.append(f"panel wall:  {s.get('panel_secs_mean', 0):.1f}s/task total "
                 f"{s.get('panel_secs_sum', 0):.0f}s")
    lines.append(f"frontier:    {s.get('frontier_ok', 0)}/{s.get('frontier_total', 0)} pass, "
                 f"{s.get('frontier_secs_mean', 0):.1f}s/task, "
                 f"${s.get('frontier_usd_sum', 0):.4f} total")
    lines.append("per panelist:")
    for m, g in (s.get("agents") or {}).items():
        lines.append(f"  {m:<40} {g['ok']}/{g['n']}  {g['secs_mean']:.1f}s  ${g['usd_mean']:.4f}")
    if s.get("trace"):
        tr = s["trace"]
        lines.append("trace aggregates (avg per agent call unless noted):")
        for k in ("rounds", "tool_calls", "study_calls"):
            if k in tr:
                lines.append(f"  {k}: {tr[k]:.1f}")
        for k in ("rate_limited", "provider_error", "timed_out"):
            if k in tr:
                lines.append(f"  {k}: {tr[k]} calls")
    return "\n".join(lines)


# ------------------------------------------------------------------- CLI
def cmd_run(a):
    load_dotenv(ROOT / ".env")
    epoch = a.epoch or default_epoch()
    if a.dry:
        plan = run_epoch(a.suite, epoch, variant=a.variant, panel=a.panel or PANEL,
                         seeds=a.seeds, rounds=a.rounds, timeout=a.timeout,
                         retain=a.retain, dry=True, debate=a.debate, tasks=a.tasks)
        print(f"dry-run plan for epoch {epoch}:")
        for r in plan:
            print(f"  {r['task']}  models: {', '.join(x['model'] for x in r['agents'])}")
        print("no network calls made")
        return 0
    print(f"lab run: epoch={epoch} suite={a.suite} variant={a.variant} "
          f"seeds={a.seeds} panel={a.panel or PANEL} rounds={a.rounds} "
          f"timeout={a.timeout} debate={a.debate}", flush=True)
    rows = run_epoch(a.suite, epoch, variant=a.variant, panel=a.panel, seeds=a.seeds,
                     rounds=a.rounds, timeout=a.timeout, retain=a.retain,
                     debate=a.debate, tasks=a.tasks)
    print(f"\nlogged {len(rows)} records -> {RUNS / (epoch + '.jsonl')}\n")
    print(render_summary(summarize(rows)))
    return 0


def cmd_diff(a):
    rows, overall = diff_epochs(a.base, a.head, suite=a.suite, variant=a.variant)
    print(render_diff(rows, overall))
    return 0


def cmd_stats(a):
    if a.epoch:
        recs, meta = load_epoch(a.epoch, meta=True)
        if not recs:
            print(f"no records in epoch {a.epoch}", file=sys.stderr)
            return 1
        print(render_summary(summarize(recs)))
        return 0
    print("epochs on disk:")
    for ep in epoch_list():
        recs, meta = load_epoch(ep, meta=True)
        if not recs:
            print(f"  {ep}  (no records)")
            continue
        s = summarize(recs)
        print(f"  {ep:<28} suite={meta.get('suite') if meta else '?'} "
              f"variant={meta.get('variant') if meta else '?'} "
              f"pass={s['tasks_pass']}/{s['n_tasks']} "
              f"{s['panel_secs_mean']:.0f}s/task ${s['panel_usd_mean']:.4f}/task")
    return 0


def cmd_traces(a):
    recs, meta = load_epoch(a.epoch, meta=True)
    if not recs:
        print(f"no records in epoch {a.epoch}", file=sys.stderr)
        return 1
    if a.plain:
        for r in recs:
            for ag in r.get("agents", []):
                tr = ag.get("trace") or {}
                print(f"{r['task']}\t{ag['model'].split('/')[-1]}\t"
                      f"ok={bool(ag.get('ok'))}\twall={tr.get('wall')}\t"
                      f"rounds={tr.get('rounds')}\ttools={tr.get('tool_calls')}\t"
                      f"study={tr.get('study_calls')}\trl={tr.get('rate_limited')}\t"
                      f"perr={tr.get('provider_error')}\trc={tr.get('rc')}\t"
                      f"usd={tr.get('usd')}\ttimed_out={tr.get('timed_out')}")
        return 0
    by = defaultdict(list)
    for r in recs:
        for ag in r.get("agents", []):
            by[ag["model"]].append(ag.get("trace") or {})
    for m, trs in sorted(by.items()):
        ok = sum(1 for r in recs for a in r.get("agents", []) if a.get("model") == m and a.get("ok"))
        n = len(trs)
        print(f"\n{m}  ({ok}/{n} pass)")
        for k in ("wall", "rounds", "tool_calls", "study_calls"):
            vals = [t.get(k) for t in trs if t.get(k) is not None]
            if vals:
                print(f"  {k}: mean {statistics.mean(vals):.1f}  "
                      f"max {max(vals)}  min {min(vals)}")
        print(f"  rate_limited calls: {sum(1 for t in trs if t.get('rate_limited'))} / {n}")
        print(f"  provider_error calls: {sum(1 for t in trs if t.get('provider_error'))} / {n}")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="lab", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one epoch (batch) of the benchmark")
    r.add_argument("--suite", choices=("code", "translation"), default="code")
    r.add_argument("--epoch", default=None, help="epoch name (default: auto timestamp)")
    r.add_argument("--variant", choices=("majority", "weighted", "top1", "debate"),
                   default="majority")
    r.add_argument("--panel", default=None, help="comma-separated panel models")
    r.add_argument("--seeds", type=int, default=1)
    r.add_argument("--rounds", type=int, default=12)
    r.add_argument("--timeout", type=int, default=600)
    r.add_argument("--tasks", default=None, help="comma-separated 1-based task indices")
    r.add_argument("--retain", action="store_true", help="keep failed worktrees for inspection")
    r.add_argument("--debate", action="store_true",
                   help="run a debate/revision round when first-pass majority fails")
    r.add_argument("--dry", action="store_true")
    r.set_defaults(fn=cmd_run)
    d = sub.add_parser("diff", help="A/B study between two epochs")
    d.add_argument("--base", required=True)
    d.add_argument("--head", required=True)
    d.add_argument("--suite", choices=("code", "translation"), default=None)
    d.add_argument("--variant", choices=("majority", "weighted", "top1", "debate"), default=None)
    d.set_defaults(fn=cmd_diff)
    s = sub.add_parser("stats", help="summary of one or all epochs")
    s.add_argument("--epoch", default=None)
    s.set_defaults(fn=cmd_stats)
    t = sub.add_parser("traces", help="per-agent inference traces of an epoch")
    t.add_argument("--epoch", required=True)
    t.add_argument("--plain", action="store_true", help="one line per agent call")
    t.set_defaults(fn=cmd_traces)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
