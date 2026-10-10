"""Consensus-of-agents benchmark harness.

Runs a panel of N small models on each task from swarm/bench_swarm.py, with
each model writing a *complete independent answer* to its own worktree, then
the task's real verifier checks each worktree and the panel outcome is the
MAJORITY VOTE of the independent pass/fail signals.

The comparable baseline is a single frontier model (openai/gpt-5.5) run
through the identical harness (same verifier, same `-yolo` flint path, same
worktree capture). Every run is appended to an append-only JSONL log under
consensus/runs/ so the published numbers are reproducible from raw data.

Design notes
------------
* "Consensus" here is honest: each agent works alone (no shared context), so
  the vote is over independent draws. This is the structure that makes the
  "3 small heads beat 1 big head" claim testable.
* Effort is not equalized: the panel pays for 3 sessions vs 1 for the
  baseline. The paper reports cost and wall-time separately so readers can
  judge whether the accuracy gain is worth the extra spend.
* All model calls go straight to OpenRouter's OpenAI-compatible
  /v1/chat/completions (the same endpoint flint uses). We use raw urllib so
  the frontier model does not depend on the flint SDK quirks that broke
  gpt-5.5 through flint.Agent.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # flint.py lives at the repo root
sys.path.insert(0, str(ROOT / "swarm"))
import bench_swarm  # noqa: E402  (task battery + verifiers)
import flint  # noqa: E402  (reuse _ssl_context + OPENROUTER default)
from translation_tasks import TRANSLATION_TASKS  # noqa: E402

OPENROUTER = getattr(flint, "OPENROUTER", "https://openrouter.ai/api/v1")
RUNS_DIR = ROOT / "consensus" / "runs"

# Panel: three small free models, well separated by vendor/family so their
# errors are as independent as a free tier allows.
PANEL = [
    "qwen/qwen3.8-27b:free",
    "cohere/north-mini-code:free",
    "poolside/laguna-xs-2.1:free",
]
FRONTIER = "openrouter:openai/gpt-5.5"  # force the OpenRouter backend: the bare `openai/` prefix would route to api.openai.com


def _ssl():
    return flint._ssl_context() or ssl._create_unverified_context()


def _call(model, messages, max_tokens=1200, timeout=120):
    """Raw OpenAI-compatible chat completion via OpenRouter. Returns (text, usage, wall)."""
    t0 = time.time()
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        OPENROUTER + "/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl()) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{model}: HTTP {e.code}: {e.read()[:300]}") from e
    wall = time.time() - t0
    usage = data.get("usage") or {}
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    # price from usage vs the catalog's per-Mtok rates (fall back to flint's rounded)
    prompt_tok = usage.get("prompt_tokens", 0)
    compl_tok = usage.get("completion_tokens", 0)
    return text, {"prompt_tokens": prompt_tok, "completion_tokens": compl_tok,
                  "usd": _price(model, prompt_tok, compl_tok)}, wall


def _price(model, ptok, ctok):
    """USD, from the live catalog when reachable; else a tiny constant fallback."""
    model = model.split(":", 1)[-1] if ":" in model else model  # strip backend: prefix
    try:
        r = urllib.request.Request(flint.OPENROUTER + "/models",
                                   headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY', '')}"})
        with urllib.request.urlopen(r, timeout=20, context=_ssl()) as resp:
            catalog = json.loads(resp.read()).get("data", [])
        for m in catalog:
            if m.get("id") == model:
                p = m.get("pricing") or {}
                return (ptok / 1e6) * float(p.get("prompt", 0)) + (ctok / 1e6) * float(p.get("completion", 0))
    except Exception:
        pass
    # last-resort estimate so cost is never "free" for frontier calls
    p_rate = {"openai/gpt-5.5": 0.000005}.get(model, 0.0)
    return (ptok / 1e6) * p_rate


def _run_agent(task, model, workdir, rounds=12, timeout=600):
    """One complete independent answer for `task` via flint in its own worktree.

    This is the *same* path bench_swarm.run_one uses (flint -p/-m/--yolo/-C),
    so a frontier model run here is directly comparable to a panelist: same
    verifier, same charge capture, same step budget."""
    charge = workdir.parent / f"charges-{int(time.time()*1000)}.jsonl"
    env = dict(os.environ, FLINT_CHARGE_FILE=str(charge), FLINT_MAX_STEPS=str(rounds))
    t0 = time.time()
    try:
        r = subprocess.run([*bench_swarm.FLINT, "-p", task["prompt"], "-m", model,
                            "--yolo", "-C", str(workdir)],
                           capture_output=True, text=True, timeout=timeout, env=env)
        wall = time.time() - t0
        ok, detail = task["verify"](workdir), ""
        if not ok:
            detail = "(verified FAIL)"
            # keep failed worktrees for inspection under consensus/runs/kept/
            kept = RUNS_DIR / "kept" / model.replace("/", "_") / task["name"]
            kept.parent.mkdir(parents=True, exist_ok=True)
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(workdir, kept)
        return {"model": model, "ok": ok, "detail": detail, "secs": round(wall, 1),
                "rc": r.returncode,
                "charges": sum(json.loads(l)["usd"] for l in charge.read_text().splitlines())
                           if charge.exists() else 0.0,
                "out_tail": r.stdout.strip()[-200:], "err_tail": r.stderr.strip()[-200:]}
    except subprocess.TimeoutExpired:
        return {"model": model, "ok": False, "detail": f"TIMEOUT {timeout}s",
                "secs": timeout, "rc": None, "charges": None,
                "out_tail": "", "err_tail": ""}


def _run_frontier(task, model, rounds=12, timeout=600):
    """Run the single frontier model through the identical flint worktree agent, so
    its score is directly comparable to the panel (same verifier + charge capture)."""
    work = Path(tempfile.mkdtemp(prefix="consensus-frontier-"))
    for name, body in task["seeded"].items():
        p = work / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    out = _run_agent(task, model, work, rounds=rounds, timeout=timeout)
    shutil.rmtree(work, ignore_errors=True)
    return out


def _run_raw_answer(task, model):
    """Single-shot answer (no tools) for `task` via raw OpenRouter, for the paper's
    cheap frontier baseline comparison. Returns dict with answer text + cost."""
    messages = [{"role": "system", "content": "You are a coding agent. Do exactly what the task asks."},
                {"role": "user", "content": task["prompt"]}]
    try:
        text, usage, wall = _call(model, messages)
        return {"model": model, "ok": None, "detail": "", "secs": round(wall, 1),
                "rc": None, "charges": usage["usd"], "text": text, "usage": usage}
    except Exception as e:
        return {"model": model, "ok": None, "detail": f"{type(e).__name__}: {str(e)[:200]}",
                "secs": None, "rc": None, "charges": None, "text": "", "usage": {}}


def run_consensus(task, panel=None, rounds=12, timeout=600, raw_frontier=False):
    """Run one task: each panelist writes its own answer, verifier checks each,
    outcome = majority vote of independent ok signals. The frontier baseline
    runs through the identical flint worktree agent when raw_frontier=True."""
    panel = panel or PANEL
    rows, workdirs = [], []
    for model in panel:
        work = Path(tempfile.mkdtemp(prefix="consensus-"))
        for name, body in task["seeded"].items():
            p = work / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
        workdirs.append(work)
        rows.append(_run_agent(task, model, work))
    oks = [r["ok"] for r in rows]
    majority = sum(bool(o) for o in oks) >= (len(panel) + 1) // 2
    frontier = _run_frontier(task, FRONTIER) if raw_frontier else None
    for w in workdirs:
        shutil.rmtree(w, ignore_errors=True)
    return {"task": task["name"], "models": panel, "agents": rows,
            "majority_ok": majority, "majority": int(sum(bool(o) for o in oks)),
            "n": len(panel), "raw_frontier": frontier}


def log_run(rec, runs_dir=RUNS_DIR):
    """Append one benchmark record to the append-only JSONL log."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "runs.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None, help="comma-separated 1-based task indices")
    ap.add_argument("--suite", choices=("code", "translation"), default="code",
                    help="task battery: code (bench_swarm) or translation (translation_tasks)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--frontier-raw", dest="raw_frontier", action="store_true",
                    help="also capture the frontier (openrouter:openai/gpt-5.5) via the flint worktree agent per task")
    ap.add_argument("--frontier-only", action="store_true",
                    help="only run the frontier baseline over the tasks (panel results already logged)")
    ap.add_argument("--models", default=",".join(PANEL),
                    help="comma-separated panel model ids")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    load_dotenv(ROOT / ".env")
    tasks = TRANSLATION_TASKS if a.suite == "translation" else bench_swarm.TASKS
    if a.tasks:
        tasks = [tasks[int(i) - 1] for i in a.tasks.split(",")]
    panel = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.dry:
        print("panel:", panel)
        print("frontier:", FRONTIER)
        for i, t in enumerate(tasks, 1):
            print(f"  {i}. {t['name']}: {t['detail']}")
        return
    print(f"panel ({len(panel)}): {panel}", flush=True)
    ran = 0
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['name']} …", flush=True)
        if a.frontier_only:
            rec = {"task": task["name"], "models": panel, "agents": [],
                   "majority_ok": None, "majority": 0, "n": len(panel),
                   "suite": a.suite,
                   "raw_frontier": _run_frontier(task, FRONTIER, rounds=a.rounds,
                                                   timeout=a.timeout)}
        else:
            rec = run_consensus(task, panel=panel, rounds=a.rounds,
                                timeout=a.timeout, raw_frontier=a.raw_frontier)
            rec["suite"] = a.suite
        rec["ts"] = time.time()
        rec["rounds"] = a.rounds
        path = log_run(rec)
        print(f"   majority {rec['majority']}/{rec['n']} -> "
              f"{'PASS' if rec['majority_ok'] else 'FAIL'}  agents: "
              + ", ".join(f"{r['model'].split('/')[-1]}={'PASS' if r['ok'] else 'FAIL'}"
                          f"({r['secs']}s,${r['charges'] or 0:.4f})" for r in rec["agents"]),
              flush=True)
        ran += 1
    print(f"logged {ran} runs -> {RUNS_DIR / 'runs.jsonl'}")


if __name__ == "__main__":
    main()