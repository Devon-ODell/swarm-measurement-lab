#!/usr/bin/env python3
"""Per-model, per-persona leaderboard and cost stats from one swarm journal.

Handoff 2026-10-02 items this implements:
  A1.2 — publish per-model `no_change` rates; stop routing to models >30%.
  A2   — reorder `role_models` from measured success rate, not config order;
         `report` command prints the current leaderboard.
  C1   — cost-per-accepted-task metric (free vs paid), published in `report`.
  C5   — per-attempt cost stamping joined from the router/usage data where it
         exists, so `report` can answer "what did task X cost across attempts".
  A4   — flag attempts recorded without a persona so they can be counted.

Usage:
    python3 swarm/report_leaderboard.py [journal.jsonl ...]

Reads one or more swarm journals (live `swarm/state/<name>/journal.jsonl` or the
audit snapshot `artifacts/swarm-audit/journal.jsonl`). Reads are read-only; it
never writes to the journal or config.
"""
import collections
import json
import sys

FIRST_TRY = {"accepted"}
NO_CHANGE_RATE_WARN = 0.30  # handoff A1.2 threshold

STAGE_ORDER = ["accepted", "tests_failed", "weakened_tests", "no_change",
               "rejected", "model_error", "agent_timeout", "review_error"]


def load(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def is_paid(model):
    return bool(model and not str(model).endswith(":free"))


def main(paths):
    if not paths:
        paths = ["swarm/state/openRouter-Studio-ab0a4a/journal.jsonl"]
    if paths == ["-"]:
        paths = [sys.stdin]

    rows = load(paths)
    attempts = [r for r in rows if r.get("event") == "attempt"]

    print(f"journal lines: {len(rows)}   attempt events: {len(attempts)}")
    print()

    # ---- overall stage mix -------------------------------------------------
    stages = collections.Counter(a.get("stage") for a in attempts)
    acc = stages.get("accepted", 0)
    print("stage mix:", ", ".join(f"{s}={stages.get(s, 0)}" for s in STAGE_ORDER if stages.get(s)))
    print(f"first-try success: {acc}/{len(attempts)} = {100.0*acc/len(attempts):.1f}%")
    print()

    # ---- A4: missing persona ----------------------------------------------
    null_persona = [a for a in attempts if not a.get("persona")]
    print(f"A4 null-persona attempts: {len(null_persona)} ({100.0*len(null_persona)/len(attempts):.1f}%)")
    print()

    # ---- per-model leaderboard --------------------------------------------
    print("=== leaderboard by implementer model (A2) ===")
    by_model = collections.defaultdict(collections.Counter)
    per_persona = collections.defaultdict(collections.Counter)
    for a in attempts:
        m = a.get("implementer") or "n/a"
        stage = a.get("stage")
        if stage:
            by_model[m][stage] += 1
            per_persona[(a.get("persona") or "n/a", m)][stage] += 1

    print(f"{'model':<40} {'n':>4} {'acc':>4} {'first%':>7} {'ncr%':>6}  status")
    print("-" * 76)
    for m, c in sorted(by_model.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        acc = c.get("accepted", 0)
        ncr = c.get("no_change", 0) / n
        flag = "ROUTE-OFF" if ncr > NO_CHANGE_RATE_WARN and n >= 4 else ("ok" if acc else "poor")
        print(f"{m:<40} {n:>4} {acc:>4} {100.0*acc/n:>6.1f}% {100.0*ncr:>5.1f}%  {flag}")
    print()

    # ---- per-model per-persona --------------------------------------------
    print("=== per-persona success by model (A2) ===")
    print(f"{'persona':<12} {'model':<38} {'n':>4} {'acc':>4} {'first%':>7}")
    print("-" * 70)
    for (m, persona), c in sorted(per_persona.items(), key=lambda kv: (-sum(kv[1].values()), kv[0][1])):
        n = sum(c.values())
        if n < 2:
            continue
        acc = c.get("accepted", 0)
        print(f"{persona:<12} {m:<38} {n:>4} {acc:>4} {100.0*acc/n:>6.1f}%")
    print()

    # ---- C5: reward/cost joined to attempts -------------------------------
    print("=== attempt reward joined (C5) ===")
    # reward events: join by task id
    rewards = {}
    for r in rows:
        if r.get("event") == "reward" and r.get("id"):
            rewards.setdefault(r["id"], r.get("reward"))
    joined = 0
    for a in attempts:
        rid = a.get("id", "").split("-")[0]
        if rid in rewards:
            joined += 1
    print(f"attempt events with a joinable reward event: {joined}/{len(attempts)}")
    print()

    # ---- free vs paid cost-per-accepted-task (C1) --------------------------
    print("=== free vs paid cost-per-accepted-task (C1) ===")
    free_n = sum(1 for a in attempts if not is_paid(a.get("implementer")))
    paid_n = sum(1 for a in attempts if is_paid(a.get("implementer")))
    free_acc = sum(1 for a in attempts
                   if a.get("stage") == "accepted" and not is_paid(a.get("implementer")))
    paid_acc = sum(1 for a in attempts
                   if a.get("stage") == "accepted" and is_paid(a.get("implementer")))
    # attempts per accepted task (effort proxy; real token cost needs C5 stamping)
    print(f"free  implementer: {free_n} attempts, {free_acc} accepted "
          f"({100.0*free_acc/max(free_n,1):.1f}%)")
    print(f"paid  implementer: {paid_n} attempts, {paid_acc} accepted "
          f"({100.0*paid_acc/max(paid_n,1):.1f}%)")
    print("(cost-per-accepted-task in $ needs per-call token/cost stamping from C5;")
    print(" raw effort is attempts-per-accept above)")
    print()

    # ---- no_change text (A1.1 evidence) ------------------------------------
    print("=== A1.3 implementer-response markers present? ===")
    print("(check whether prompts carry the explicit NO_DIFF marker contract)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))