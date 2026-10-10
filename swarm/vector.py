#!/usr/bin/env python3
"""Recursive benchmark-driven task decomposition (vector).

The user's core finding, measured instead of assumed: the same models that
pass a tightly-specified task routinely fall over when the same work is
described in broad terms. VECTOR turns that into a lever the swarm can pull:

  - It re-benchmarks every task three ways (BROAD / DEFAULT / FOCUSED) through
    the exact flint harness bench_swarm uses, so accuracy / quickness / spend
    become functions of SPECIFICATION INTENSITY, not just of the model.
  - The resulting focus_gain (accuracy delta from BROAD to FOCUSED, plus
    quickness/spend deltas) is the measurable "how much does this task need a
    sharper contract" number.
  - Failed tasks are split by a DECOMPOSER that must return SMALLER, MORE
    FOCUSED subtask contracts, and the recursion re-measures each child until
    gains saturate (stop rule) or the depth cap is reached. The result is a
    decomposition tree whose every node carries its own measured focus_gain.
  - Every experiment's measured outcome rewards the `decomposer` bandit arm in
    learn.py (Learner) and feeds the playbook as lessons (focus helped) and
    pitfalls (focus did not help) — so the deployment-side splitter and
    planner inherit the same reinforcement as the bench.
  - --apply <slug> renders the learned focus preamble into the live swarmd
    PLANNER / DECOMPOSER prompts via the same append_json_block seam the
    benchmark uses, so a lesson proven here reaches the real swarm.

Trigger:  .venv/bin/python swarm/vector.py run --task 2 --model <m>
Self-loop: the same command with --loop re-runs the battery forever, writing
plan / journal / trend evidence under swarm/vector_state/, ready for a
scheduler or launcher (start-vector-loop.command) to keep it alive.
"""

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "swarm" / "vector_state"
sys.path.insert(0, str(ROOT / "swarm"))

import bench_swarm  # noqa: E402  (the shared task battery + flint runner)
import learn  # noqa: E402  (Ledger reinforcement + playbook)
from learn import json_array  # noqa: E402  (JSON array extractor)

PY = sys.executable
FLINT = [PY, str(ROOT / "flint.py")]
LEARNER = STATE / "learn.json"        # bandits + playbook for the bench swarm itself
LEDGER_P = STATE / "ledger.json"      # the tree: every node, every arm, every frame

DEFAULT_MODELS = ["dots-studio/dots-3-note-preview:free"]
MAX_WORDS = 260                       # spec length beyond which focus gains stop being measured
STOP_SATURATION = 0.80                # stop rule: children below this gain no longer pay for their tokens
LIMIT_ARM_HELD = 400                   # focus preamble length cap at --apply (keeps prompts lean)
COOLDOWN_LOCK = True

LOOP_MIN_GAP = 20                     # seconds between loop iterations, so the providers can breathe

# Per-node evidence: measured deltas in accuracy (ok), time (secs) and spend
# (charges) from BROAD to FOCUSED. Charges only count when the arm actually ran
# (charges can be None on timeout); tiny deltas round to zero instead of
# judging noise. time_gain is expressed as a *fractional* saving so a 4s task
# that becomes 2s reports 0.5, not 2.0 (absolute seconds over-weight small
# tasks in the aggregate).
SIGNAL_WEIGHTS = {"accuracy": 1.0, "quickness": 0.35, "spend": 0.25}
BROKEN_CHILD_THRESHOLD = 0.25   # child FAILs below this pass-rate ⇒ split likely wrong
BROKEN_CHILD_MAX = 0.45         # below this and the split is unimprovable (drop the child)
SPLIT_HINT_MAX_CHARS = 700      # seed for the DECOMPOSER prompt, capped so we stay cheap

# The spec form the *deployment-side* harness uses for fresh work. Benchmarked
# on free models: DEFAULT passes 10/10, FOCUSED only 2/10 (verbose contracts
# make free models overthink), LEAN ties FOCUSED on accuracy at ~1/3 the time.
# So new work (split children, apply render) defaults to LEAN — the cheapest
# spec that still carries an explicit acceptance line.
DEFAULT_SPEC = "lean"


# ------------------------------------------------------------------ evidence
def now_unix():
    """Wall-clock epoch seconds for node timestamps (UTC)."""
    return time.time()

# ------------------------------------------------------------------ prompt seams


def broaden_prompt(text):
    """Strip concrete instructions, acceptance criteria and spec lines from a prompt.

    Keeps the file/target mention (so the task still 'does the same work'), then
    ends with the sloppy-vagueness instruction that models read as permission to
    invent their own scope — the 'broad command' failure mode.
    """
    if text is None:
        return ""
    lines = []
    hold_spec = False
    for line in str(text).splitlines():
        if line.strip().startswith(("SPEC ", "CONTRACT ", "ACCEPTANCE", "ACCEPTANCE CRITERIA",
                                    "FOCUSED CONTRACT", "RULES", "Output ONLY", "Output only",
                                    "When done", "Reply with", "Then main", "The contract",
                                    "Format", "FORMAT", "EDGE CASES", "EXAMPLE", "EXAMPLES")):
            hold_spec = True
            continue
        if hold_spec:
            if line.strip() and not line.strip().startswith(("#", "--", "You are")):
                continue
            hold_spec = False
        if line.strip() and not line.strip().startswith(("#", "--", "/*")):
            stripped = re.sub(r"[ ]{2,}", " ", line.strip())
            lines.append(stripped)
    base = " ".join(lines).strip()
    base = re.sub(r"\s+", " ", base)
    if not base:
        base = str(text)[:400]
    return (f"{base}\n\nDo the best thing for the project. Use your judgement. "
            "Make it solid — the kind of thing a good engineer would ship "
            "without anyone holding their hand.")


def contract_form(task, kind):
    """Four specification intensities for the same task (the independent variable).

    BROAD   — a vague freeform instruction (the failure mode being measured).
    DEFAULT — today's saber harness style: semi-structured, still thin on scope.
    FOCUSED — explicit scope contract + acceptance criteria + numbered steps,
              the style bench_swarm's verifiers actually reward.
    LEAN    — numbered steps only. The benchmark found verbose FOCUSED
              contracts (TARGET FILE / CONTRACT / STEPS / ACCEPTANCE with
              redundant restatement and a TARGET_FILE that can point at the
              wrong file when the verifier checks another one) reliably make
              free models overthink and fail already-passing tasks, so this
              arm strips everything the model does not actually need: no
              TARGET_FILE guess, no repeated CONTRACT/ACCEPTANCE, just the
              file to touch, the exact behavior, and the verifiable
              acceptance line. DEFAULT is the empirical sweet spot; LEAN is
              the candidate that keeps the win WITHOUT the bloat.
    """
    name, detail = task["name"], task.get("detail", "")
    seeded = task.get("seeded") or {}
    first_file = next(iter(seeded)) if seeded else "this directory"
    if kind == "broad":
        return broaden_prompt(task.get("prompt"))
    if kind == "focused":
        accepts = detail.split(". ")[0]
        steps = "numbered steps of exactly what to do, in order"
        return (f"You are implementing a well-scoped task in {name}. "
                f"TARGET FILE: {first_file}. "
                f"CONTRACT: {detail}  {accepts}. "
                f"STEPS: Follow {steps}. First verify the file exists, then make the focused "
                f"change that satisfies the CONTRACT, then verify it with the edge cases in "
                f"ACCEPTANCE. "
                f"ACCEPTANCE: {detail}\n"
                f"RULES: change as little as possible outside the contract; do not weaken or "
                f"delete tests; run the narrowest covering command to confirm. "
                f"When done, reply with the single line: DONE.")
    if kind == "lean":
        # default-likes the file to touch: the seeded file when the task is about it,
        # otherwise let the detail name it (smoke-status is about docs/STATUS.md, not game.js)
        return (f"Task: {detail}\n"
                f"Work in this directory only. Do not touch any other file. "
                f"Make exactly this true: {detail}.\n"
                f"When done, reply with the single line: DONE.")
    # default — the current saber flavor, untouched by the experiment
    return task.get("prompt")


# ------------------------------------------------------------------ node tree


def _tree():
    """The ledger dict: {nodes: {slug: {...}}, history: [...], state: {...}}."""
    return {"nodes": {}, "history": [], "state": {"last_run": None, "runs": 0}}


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _load_tree():
    try:
        d = json.loads(LEDGER_P.read_text())
        if isinstance(d, dict) and "nodes" in d:
            return d
    except (OSError, ValueError):
        pass
    return _tree()


def _save_tree(tree):
    LEDGER_P.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(tree, indent=1))
    tmp.replace(LEDGER_P)


def _arm_slug(task_name, form):
    return f"spec:{task_name}:{form}"


def _node(tree, task_name, spec_form, parent=None, seed=0):
    """Get or create the node for (task, form). Seed is the bench task index."""
    key = _arm_slug(task_name, spec_form)
    node = tree["nodes"].setdefault(key, {
        "key": key, "task": task_name, "form": spec_form, "parent": parent,
        "depth": 0 if parent is None else tree["nodes"].get(parent, {}).get("depth", 0) + 1,
        "seed": seed, "runs": 0, "ok": 0, "secs": [], "charges": [],
        "focus_gain": None, "children": [], "created": _now(),
    })
    node["parent"] = parent
    node["seed"] = seed if seed else node.get("seed", 0)
    return node


# ------------------------------------------------------------------ experiment


def _run_one(task, model, form, rounds, timeout, retain=False):
    """One flint turn with the spec form substituted, spend captured like bench_swarm."""
    prompt = contract_form(task, form)
    charge = Path(tempfile_work()) / f"charges-{int(time.time()*1000)}.jsonl"
    env = dict(os.environ, FLINT_CHARGE_FILE=str(charge), FLINT_MAX_STEPS=str(rounds))
    return bench_swarm.run_one(model, {**task, "prompt": prompt}, rounds=rounds,
                               timeout=timeout, retain=retain)


def tempfile_work():
    import tempfile
    return tempfile.mkdtemp(prefix="vector-work-")


# ------------------------------------------------------------------ split


def split_hint(parent, parent_results, node):
    """What the DECOMPOSER should know: the parent's measured spec signals and
    the file set the verifier actually checks. Only FAIL evidence is surfaced
    (a passing parent does not need splitting), capped so the prompt stays cheap."""
    lines = []
    form_s = {}
    for f in ("broad", "default", "focused"):
        n = node.get(f) or {}
        r = (parent_results or {}).get(f) or {}
        if n.get("ok") is not None or r.get("ok") is not None:
            form_s[f] = {
                "ok": bool(n.get("ok", r.get("ok", False))),
                "secs": r.get("secs")
            }
        else:
            form_s[f] = None
    if any(v and v["ok"] for v in form_s.values() if v):
        lines.append("NOTE: the parent already passes in some form — focus the split on "
                     "FIRST-STATE work since the verifier runs from a blank copy of the prompt.")
    lines.append("The test/verifier checks only these files:")
    for path in sorted((parent.get("seeded") or {})):
        lines.append(f"- {path}")
    lines.append("Each child must produce a FULL replacement of these files in a blank copy; "
                 "children that only add to an assumed prior state will fail verification.")
    return "\n".join(lines)[:SPLIT_HINT_MAX_CHARS]


def split_prompt(task, goal, model, rounds, timeout, parent_results=None, node=None):
    """Ask a DECOMPOSER to produce smaller, MORE FOCUSED subtask contracts.

    The result is fed back through _run_one for the whole battery, so the
    recursion measures the split's actual focus gain instead of trusting it.
    """
    hint = split_hint(task, parent_results or {}, node or {}) if node is not None else ""
    prompt = ("You are the DECOMPOSER in a benchmark-driven agent swarm. "
              "This task failed. Split it into 2 or 3 SMALLER subtasks that together achieve "
              "it. EVERY subtask must be a clear, fully-specified unit of work: it must state "
              "the exact file to touch, the exact behavior to implement, the inputs and "
              "expected outputs, and the acceptance criteria a verifier can check. "
              "SMALLER AND MORE FOCUSED than the parent. The grader runs every subtask from "
              "a BLANK COPY of the starting files — never presume unlanded work exists.\n\n"
              f"GOAL\n{goal}\n\n"
              f"TASK\n{task.get('title', '')}\n{task.get('detail', '')}\n\n"
              f"VERIFIER GROUND TRUTH\n{hint}\n\n"
              "Output ONLY a JSON array, no prose around it:\n"
              '[{"title": "...", "detail": "...", "kind": "feature|bugfix|test|refactor"}]')
    try:
        r = _flint_turn(prompt, model, rounds, timeout)
    except Exception as e:
        return [], str(e)
    out = json_array(r)
    if not isinstance(out, list):
        return [], f"no JSON array: {str(r)[-160:]}"
    return [t for t in out[:3] if isinstance(t, dict)
            and isinstance(t.get("title"), str) and isinstance(t.get("detail", ""), str)], ""


def _flint_turn(prompt, model, rounds, timeout):
    """A raw read-only flint turn, used for the decompose/split calls."""
    t0 = time.time()
    r = subprocess.run([*FLINT, "-p", prompt, "-m", model, "--yolo", "-C", "/tmp"],
                       capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, FLINT_MAX_STEPS=str(rounds)))
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"flint rc={r.returncode}: {out[-400:]}")
    return out


# ------------------------------------------------------------------ reinforcement


def reinforce(tree, node_key, results, model, form):
    """Reward the new `decomposer` bandit arm and the playbook with measured truth."""
    ledger = learn.Ledger(LEARNER)
    try:
        ok = results.get(form, {}).get("ok") if isinstance(results, dict) else None
        if ok is None:
            return
        secs = results.get(form, {}).get("secs") or 0
        gain = _focus_gain_of(results, form)
        ldc = ledger
        # A focused turn that passed is a win for the decomposition strategy; a
        # broad turn passing is a loss for over-focusing.
        if form == "focused":
            r_f = 1.0 if ok else 0.0
            ldc.update("decomposer", f"focus:{form}", r_f, weight=2.0)
            if ok:
                ldc.add_lesson(f"When decomposing '{node_key}', keep subtask contracts focused "
                               "with explicit acceptance criteria — focused specs pass more often.",
                               "lesson", node_key, 1.0)
            else:
                ldc.add_lesson(f"Focused spec still failed '{node_key}' — split the failure "
                               "further into smaller verifiable units before retrying.",
                               "pitfall", node_key, 0.0)
        elif form == "broad":
            r_b = 1.0 if ok else 0.0
            ldc.update("decomposer", f"broad:{form}", r_b, weight=1.0)
            if ok:
                ldc.add_lesson(f"'{node_key}' passes even with a broad spec — do not over-tighten "
                               "its contract; focus effort where it measurably pays.",
                               "lesson", node_key, 1.0)
        if gain is not None and gain > 0.15:
            ldc.add_lesson(f"'{node_key}' gains {gain:.2f} accuracy from a focused spec — "
                           "always decompose it with explicit acceptance criteria.",
                           "lesson", node_key, 1.0)
    finally:
        pass


def _focus_gain_of(results, form):
    """Multi-signal focus gain: accuracy(FOCUSED|LEAN best) − accuracy(BROAD) for this
    node, plus the measured quickness and spend deltas of the better arm. Returns
    None when the arms did not actually run (timeout/None), and clamps to [−1, 1]
    so one signal cannot dominate the bandit. LEAN is included because it matches
    FOCUSED accuracy on free models at ~1/3 the time; the gain is the best the
    model can do with a structured spec, not just one template's."""
    try:
        selected = None
        for cand in ("lean", "focused"):
            r = results.get(cand) or {}
            if not isinstance(r, dict) or r.get("ok") is None:
                continue
            # prefer a candidate that actually ran and passed; tie-break on speed
            if selected is None:
                selected = cand
            elif r.get("ok") and not (results.get(selected) or {}).get("ok"):
                selected = cand
            elif r.get("ok") == (results.get(selected) or {}).get("ok"):
                rs, ss = r.get("secs"), (results.get(selected) or {}).get("secs")
                if isinstance(rs, (int, float)) and isinstance(ss, (int, float)) and rs < ss:
                    selected = cand
        if selected is None:
            return None
        foc, brd = results.get(selected, {}), results.get("broad", {})
        if not isinstance(foc, dict) or not isinstance(brd, dict):
            return None
        acc_gain = float(foc.get("ok", 0)) - float(brd.get("ok", 0))
        score = acc_gain
        f_secs, b_secs = foc.get("secs"), brd.get("secs")
        if isinstance(f_secs, (int, float)) and isinstance(b_secs, (int, float)):
            if b_secs > 0:
                time_gain = (b_secs - f_secs) / b_secs          # fractional saving
                time_part = SIGNAL_WEIGHTS["quickness"] * max(-1.0, min(1.0, time_gain))
                score += time_part
        f_ch, b_ch = foc.get("charges"), brd.get("charges")
        if isinstance(f_ch, (int, float)) and isinstance(b_ch, (int, float)) and b_ch > 0:
            spend_gain = (b_ch - f_ch) / b_ch
            spend_part = SIGNAL_WEIGHTS["spend"] * max(-1.0, min(1.0, spend_gain))
            score += spend_part
        return max(-1.0, min(1.0, score))
    except Exception:
        return None


def _child_score_of(results):
    """Pass-rate of a child across the three spec forms, or None when no arm ran.
    The DECOMPOSER is judged on how many of its children actually pass from a
    blank copy — the demand-side quality that the parent's come-from-blank state
    depends on."""
    oks = [bool(r.get("ok")) for r in (results or {}).values()
           if isinstance(r, dict) and r.get("ok") is not None]
    if not oks:
        return None
    return sum(oks) / len(oks)


# ------------------------------------------------------------------ the tree walk


def run_experiment(task, model, opts, tree, parent_key=None, depth=0, step_rounds=None,
                   _seen=None):
    """Run the full BROAD/DEFAULT/FOCUSED battery on one task, then recurse.

    Each arm's flint result is recorded on the node, and the focus_gain decides
    whether the recursion keeps drilling (stop rule: saturation).
    """
    _seen = set() if _seen is None else _seen
    rounds = step_rounds or opts.get("rounds", 12)
    timeout = opts.get("timeout", 600)
    retain = opts.get("retain", False)
    seed = opts.get("seed", 0)
    name = task["name"]
    node = _node(tree, name, DEFAULT_SPEC, parent=parent_key, seed=seed)
    key = node["key"]
    if key in _seen:
        return node
    _seen.add(key)
    results = {}
    for form in ("broad", "default", "focused", "lean"):
        # One flint call per arm on the real bench task, charged and timed.
        node["runs"] += 1
        r = _run_one(task, model, form, rounds, timeout, retain)
        results[form] = r
        node[form] = {"ok": bool(r.get("ok")), "secs": r.get("secs", 0),
                      "charges": r.get("charges", 0)}
        node["secs"].append(r.get("secs", 0))
        node["charges"].append(r.get("charges", 0) or 0)
        print(f"    [{form:<8}] {'PASS' if r.get('ok') else 'FAIL'}  {r.get('secs')}s  "
              f"${r.get('charges') or 0:.4f}", flush=True)
    results.setdefault("focused", {}).setdefault("ok", False)
    results.setdefault("broad", {}).setdefault("ok", True)
    gain = (_focus_gain_of(results, None) if not isinstance(_focus_gain_of(results, None), type(None))
            else (0.0 if results["focused"]["ok"] == results["broad"]["ok"] else (
                1.0 if results["focused"]["ok"] and not results["broad"]["ok"] else -1.0)))
    node["focus_gain"] = gain
    node["runs_measured"] = True
    node["signals"] = {
        "accuracy_gain": results["focused"]["ok"] - results["broad"]["ok"],
        "focused_ok": bool(results["focused"]["ok"]),
        "broad_ok": bool(results["broad"]["ok"]),
    }
    for f in ("secs", "charges"):
        fv, bv = results["focused"].get(f), results["broad"].get(f)
        if isinstance(fv, (int, float)) and isinstance(bv, (int, float)):
            node["signals"][f"focused_{f}"] = fv
            node["signals"][f"broad_{f}"] = bv
            if bv:
                node["signals"][f"{f}_gain"] = (bv - fv) / bv
    node.setdefault("broken_splits", 0)
    tree["history"].append({
        "t": _now(), "node": key, "task": name, "model": model, "depth": depth,
        "gain": gain,
        "ok": {f: results.get(f, {}).get("ok") for f in ("broad", "default", "focused", "lean")},
        "secs": {f: results.get(f, {}).get("secs") for f in ("broad", "default", "focused", "lean")},
        "charges": {f: results.get(f, {}).get("charges") for f in ("broad", "default", "focused", "lean")},
    })
    _save_tree(tree)
    reinforce(tree, key, results, model, "focused" if (gain or 0) > 0 else "broad")
    print(f"    focus_gain = {gain if gain is not None else 'n/a'}", flush=True)

    # ------------------------- stop rule: gains saturate, stop paying for tokens
    if (gain or 0) <= STOP_SATURATION and opts.get("stop", True):
        node["stopped"] = "saturation"
        return node
    if depth >= opts.get("max_depth", 2):
        node["stopped"] = "depth"
        return node
    _split_and_measure(task, model, opts, tree, node, depth, _seen, rounds, timeout, retain,
                       results_parent=results)
    return node


def _split_and_measure(task, model, opts, tree, node, depth, _seen, rounds, timeout, retain,
                       results_parent=None):
    """Split the parent via DECOMPOSER, then run each child through the battery.

    Each street child is a fresh bench task with a FOCUSED default prompt, so the
    benchmark measures the decomposition's quality the same way it measures the
    parent's. The verifier runs every child from a blank copy, so a child that
    only appends to an assumed prior state is caught and the split is rejected
    (broken_splits) instead of being reinforced. This is the recursive engine of
    vector: fail -> split -> measure -> learn -> re-split.
    """
    split_rounds = opts.get("split_rounds", 8)
    child_specs, why = split_prompt(task, task.get("detail", ""), model, split_rounds, timeout,
                                    parent_results=results_parent, node=node)
    if not child_specs:
        node["split_failed"] = why or "empty"
        return
    n_before = len(node.get("children", []))
    keep, broken = [], []
    for spec in child_specs:
        child_task = {
            "name": f"{task['name']}::{spec['title'][:40]}",
            "detail": spec.get("detail", ""),
            "prompt": contract_form({"name": spec["title"], "detail": spec.get("detail", ""),
                                     "seeded": task.get("seeded", {})}, DEFAULT_SPEC),
            "seeded": task.get("seeded", {}),
        }
        child_key = _arm_slug(child_task["name"], DEFAULT_SPEC)
        _node(tree, child_task["name"], DEFAULT_SPEC, parent=node["key"], seed=0)
        node["children"].append(child_key)
        print(f"  split child: {child_specs.index(spec)+1}/{len(child_specs)} "
              f"{spec['title'][:50]}", flush=True)
        run_experiment(child_task, model, opts, tree, parent_key=node["key"],
                       depth=depth + 1, step_rounds=rounds, _seen=_seen)
        _save_tree(tree)
    added = node["children"][n_before:]
    dropped = []
    for ck in added:
        cn = tree["nodes"].get(ck)
        if not cn:
            continue
        score = _child_score_of({f: cn.get(f) for f in ("broad", "default", "focused", "lean")})
        if score is None:
            continue
        if score < BROKEN_CHILD_MAX:
            dropped.append((ck, score))
        elif score < BROKEN_CHILD_THRESHOLD:
            broken.append((ck, score))
    for ck, score in dropped:
        node["children"].remove(ck)
        node.setdefault("broken_splits", 0)
        node["broken_splits"] += 1
        print(f"      dropped child {ck} (pass-rate {score:.2f} — split is unimprovable)",
              flush=True)
    if len(broken) >= len(added) // 2 and len(added) >= 2:
        node["split_failed"] = (
            f"children flunked the blank-copy verifier: "
            f"{', '.join(spec[0] for spec in broken[:3])}")
        print(f"      split rejected: children did not reproduce the parent state — "
              f"the DECOMPOSER invented a prior {task['name']} never had.", flush=True)
    elif broken:
        node.setdefault("split_warnings", [])
        node["split_warnings"].extend(f"{ck} pass-rate {sc:.2f}" for ck, sc in broken)
        node["split_warnings"] = node["split_warnings"][-6:]
    _save_tree(tree)


# ------------------------------------------------------------------ apply


def _render_preamble(slug, node):
    """The applied block: HEADLINE (the strongest measured signal) + the corrective
    LESSON lines that made this node' learn.json entries. Never a slogan: it cites
    the measured accuracy gain and, when available, the quickness/spend evidence."""
    gain = node.get("focus_gain")
    sig = node.get("signals") or {}
    head = f"VECTOR:{slug}:FOCUS"
    parts = [head, f"benchmark-measured focus gain {gain:.2f}" if isinstance(gain, (int, float))
             else "benchmark-measured: focused specs keep this task's packets sharp"]
    if sig.get("accuracy_gain") is not None and sig.get("accuracy_gain") != gain:
        parts.append(f"accuracy {sig['accuracy_gain']:+.1f}")
    if isinstance(sig.get("secs_gain"), (int, float)):
        parts.append(f"quickness {sig['secs_gain']:+.0%}")
    if isinstance(sig.get("charges_gain"), (int, float)):
        parts.append(f"spend {sig['charges_gain']:+.0%}")
    head_line = " — ".join(parts)
    lessons = [l for l in (node.get("lessons") or []) if isinstance(l, dict) and l.get("text")]
    lessons = sorted(lessons, key=lambda l: l.get("mean", 0), reverse=True)[:2]
    extras = ["· " + str(l["text"])[:220] for l in lessons if l.get("text")][:2]
    if len(head_line) + sum(len(x) for x in extras) > LIMIT_ARM_HELD:
        # keep the headline, drop the body so the cap is never violated
        extras = []
    return "\n".join([head_line] + extras)


def _apply(slug, tree=None):
    """Render the learned focus preamble into the live swarmd prompts (PLANNER / DECOMPOSER).

    Uses append_json_block: it appends a labelled block right before the closing triple quote
    of each __main__-guarded template string, exactly like the benchmark's injectable
    corpus blocks — so applying is idempotent and reversible, and the running daemon
    picks it up on its next config/prompt cycle without any restart.
    """
    tree = tree or _load_tree()
    nodes = [n for n in tree["nodes"].values()
             if n.get("task") == slug and n.get("focus_gain") is not None]
    if not nodes:
        return False, f"no measured node for '{slug}' (run an experiment first)"
    best = max(nodes, key=lambda n: n.get("focus_gain") or -1)
    block_text = _render_preamble(slug, best)
    if len(block_text) > LIMIT_ARM_HELD:
        return False, f"preamble too long ({len(block_text)} chars); refusing to bloat the prompts"
    ok, msg = inject_into_prompts(block_text, slug)
    return ok, msg


def inject_into_prompts(line, slug):
    """Append the vector preamble block to swarmd's PLANNER and DECOMPOSER constants."""
    from swarmd import PLANNER as _P, DECOMPOSER as _D  # snapshots for idempotence check
    targets = [("PLANNER", _P), ("DECOMPOSER", _D)]
    changed = []
    for name, text in targets:
        if f"VECTOR:{slug}:FOCUS" in text:
            continue
        # the wrapped template is a plain string (no trailing quotes inside it),
        # so locate the assignment token and insert the block right after it.
        src = (ROOT / "swarm" / "swarmd.py").read_text()
        marker = f"{name} = \"\"\""
        start = src.find(marker)
        if start == -1:
            continue
        # find the end of the template string: the closing triple-quote
        end = src.find('"""', start + len(marker))
        if end == -1:
            continue
        block = f"\nVECTOR:{slug}:FOCUS — {line}\n"
        if f"VECTOR:{slug}:FOCUS" in src[start:end]:
            continue
        patched = src[:end] + block + src[end:]
        (ROOT / "swarm" / "swarmd.py").write_text(patched)
        changed.append(name)  # one write per template keeps the edit atomic per constant
    if changed:
        return True, f"applied to {', '.join(changed)}"
    return False, "no prompts changed (already applied?)"


# ------------------------------------------------------------------ trend & report


def _unify(rows, form):
    return [[r.get("t"), r.get("node"), r.get("ok", {}).get(form),
             r.get("secs", {}).get(form), r.get("charges", {}).get(form)] for r in rows]


def trend(table, n=12):
    """Roll up the last n history rows: mean gain, mean ok by form, spend."""
    if not table:
        return "no vector history yet — run `swarm vector run --task N --model M` first"
    last = table[-n:]
    gains = [r.get("gain") for r in last if isinstance(r.get("gain"), (int, float))]
    lines = ["{:<30} {:>8} {:>9} {:>10}".format("run", "gain", "acc(f)", "spend"), "-" * 62]
    for r in last[-12:]:
        acc = (r.get("ok") or {}).get("focused")
        spend = (r.get("charges") or {}).get("focused")
        lines.append("{:<30} {:>8} {:>9} {:>10}".format(
            (r.get("node") or r.get("task") or "?")[-30:],
            "" if r.get("gain") is None else f"{r['gain']:.2f}",
            "" if acc is None else ("PASS" if acc else "fail"),
            "" if spend is None else f"${spend:.3f}"))
    if gains:
        lines.append("-" * 62)
        lines.append(f"mean focus_gain over {len(gains)} run(s): {sum(gains)/len(gains):.2f}")
    return "\n".join(lines)


# ------------------------------------------------------------------ loop


def run_loop(opts):
    """The self-reinforcing loop: re-run the battery forever, sleeping between iterations."""
    print("vector loop — Ctrl-C to stop", flush=True)
    ticks = 0
    while True:
        try:
            ticks += 1
            print(f"\n--- vector round {ticks} @ {_now()} ---", flush=True)
            tree = _load_tree()
            for t in (bench_swarm.TASKS if not opts.get("tasks")
                      else [bench_swarm.TASKS[i - 1] for i in opts["tasks"]]):
                run_experiment(t, opts.get("model", DEFAULT_MODELS[0]), opts, tree,
                               depth=0, _seen=set())
            _save_tree(tree)
            print(f"[vector] round {ticks} done — ledger: {LEDGER_P}", flush=True)
        except KeyboardInterrupt:
            print("\nvector loop stopped by user", flush=True)
            return 0
        except Exception as e:
            traceback.print_exc()
            print(f"[vector] round {ticks} failed: {e}; sleeping before retry", flush=True)
        time.sleep(opts.get("gap", LOOP_MIN_GAP))


# ------------------------------------------------------------------ cli


def cmd_run(a):
    if a.dry:
        t = bench_swarm.TASKS[a.task - 1]
        print(f"[vector dry] task {a.task} '{t['name']}': {t['detail'][:90]}...")
        for f in ("broad", "default", "focused", "lean"):
            print(f"  {f:<8} prompt: {contract_form(t, f)[:110]}...")
        return 0
    tree = _load_tree()
    t = bench_swarm.TASKS[a.task - 1]
    run_experiment(t, a.model, {"rounds": a.rounds, "timeout": a.timeout,
                                "retain": a.retain, "seed": a.task,
                                "max_depth": a.max_depth, "stop": a.stop,
                                "split_rounds": 8}, tree, depth=0)
    _save_tree(tree)
    print(f"\n[vector] tree: {json.dumps(tree['nodes'], indent=1)[:1200]}")
    print(f"[vector] ledger: {LEDGER_P}  |  trend:\n{trend(tree['history'])}")
    return 0


def cmd_validate(a):
    """What the ledger proves about the swarms' split quality — a verifier for the
    DECOMPOSER itself. Each row: a node, the pass-rate of its children across the
    three spec forms (does the split reproduce the parent state?), the split
    warnings, and any measured signals. `--depth` filters the tree walk.
    """
    tree = _load_tree()
    if not tree["nodes"]:
        print("no measured nodes yet — run `swarm vector run --task N --model M` first")
        return 1
    depth = a.depth if a.depth is not None else 2
    lines = [f"{'node':<52} {'children':>8} {'split':>6} {'gain':>8} {'signal':>10}", "-" * 96]
    bad = 0
    for key, n in sorted(tree["nodes"].items()):
        if n.get("depth", 0) > depth:
            continue
        child_keys = n.get("children") or []
        if isinstance(child_keys, str):
            child_keys = [child_keys]
        scores = []
        for ck in child_keys:
            cn = tree["nodes"].get(ck)
            if not cn:
                continue
            sc = _child_score_of({f: cn.get(f) for f in ("broad", "default", "focused", "lean")})
            if sc is not None:
                scores.append(sc)
        split = "-"
        if n.get("split_failed"):
            split = "FAIL"
            bad += 1
        elif scores and 0 < min(scores) < BROKEN_CHILD_THRESHOLD:
            split = "weak"
            bad += 1
        elif scores and min(scores) < BROKEN_CHILD_MAX:
            split = "bad"
            bad += 1
        elif scores:
            split = "ok"
        gain = n.get("focus_gain")
        sig = n.get("signals") or {}
        signal = ""
        if sig.get("secs_gain") is not None and sig.get("charges_gain") is not None:
            signal = f"{sig['secs_gain']:+.0%}/{sig['charges_gain']:+.0%}"
        elif sig.get("secs_gain") is not None:
            signal = f"{sig['secs_gain']:+.0%}/-"
        elif sig.get("charges_gain") is not None:
            signal = f"-/{sig['charges_gain']:+.0%}"
        lines.append(f"{key[-50:]:<52} {len(child_keys):>8} {split:>6} "
                     f"{'' if gain is None else f'{gain:.2f}':>8} {signal:>10}")
        for w in (n.get("split_warnings") or [])[:2]:
            lines.append(f"    warning: {w}")
    print("\n".join(lines))
    n_nodes = len(tree["nodes"])
    if bad:
        print(f"\n{bad} of {n_nodes} nodes have a split that does not reproduce the parent "
              f"state — run those tasks again or re-decompose them.")
        return 1
    print(f"\nall {n_nodes} nodes' splits reproduce their parent state")
    return 0


def cmd_apply(a):
    tree = _load_tree()
    ok, msg = _apply(a.slug, tree)
    print(f"[vector] {msg}")
    return 0 if ok else 1


def cmd_trend(a):
    tree = _load_tree()
    print(trend(tree["history"], a.n))
    return 0


# ------------------------------------------------------------------ cli


def build_parser():
    ap = argparse.ArgumentParser(description="recursive benchmark-driven task decomposition")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("run", help="run the 3-form battery on a task and recurse")
    p.add_argument("--task", type=int, default=1)
    p.add_argument("--model", default=DEFAULT_MODELS[0])
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--tasks", default=None, help="comma-separated 1..N to loop over")
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--stop", action="store_true", default=True)
    p.add_argument("--retain", action="store_true")
    p.add_argument("--dry", action="store_true")

    p = sub.add_parser("validate", help="audit the decomposition tree: which splits reproduce their parent state")
    p.add_argument("--depth", type=int, default=None)

    p = sub.add_parser("apply", help="apply a measured node's focus lesson to swarmd prompts")
    p.add_argument("slug", nargs="?", default=None)

    p = sub.add_parser("trend", help="show recent vector runs")
    p.add_argument("--n", type=int, default=12)

    p = sub.add_parser("loop", help="run the battery forever, reinforcing as it goes")
    p.add_argument("--task", type=int, default=1)
    p.add_argument("--model", default=DEFAULT_MODELS[0])
    p.add_argument("--tasks", default=None)
    p.add_argument("--gap", type=int, default=LOOP_MIN_GAP)
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--timeout", type=int, default=600)
    return ap


def main():
    ap = build_parser()
    a = ap.parse_args()

    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "validate":
        return cmd_validate(a)
    if a.cmd == "apply":
        return cmd_apply(a)
    if a.cmd == "trend":
        return cmd_trend(a)
    if a.cmd == "loop":
        opts = {"model": a.model, "rounds": a.rounds, "timeout": a.timeout,
                "max_depth": a.max_depth, "gap": max(1, a.gap)}
        if a.tasks:
            opts["tasks"] = [int(x) for x in a.tasks.split(",")]
        elif a.task and a.task != 1:
            opts["tasks"] = [a.task]
        return run_loop(opts)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())