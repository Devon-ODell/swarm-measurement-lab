# swarm-measurement-lab

Benchmark and tuning infrastructure extracted from [OpenRouterSwarm](https://github.com/Devon-ODell/OpenRouterSwarm):
the tooling used to measure how packet-writing style (broad/default/focused/lean) affects free-model
coding results, and the consensus-of-agents comparison work.

This is **not required to run the swarm** — OpenRouterSwarm works standalone without any of this. It's
the research/tuning side: the "focused beats default beats broad" packet doctrine documented in
OpenRouterSwarm's `AGENTS.md` came from running this against real tasks.

## Contents

- `swarm/vector.py`, `swarm/vector_runner.py` — the VECTOR bench: runs one task in BROAD/DEFAULT/FOCUSED/
  LEAN packet form and measures the gain from tighter specs. `vector_runner.py apply <lesson>` folds a
  measured finding back into a planner/decomposer prompt (copy the result into your OpenRouterSwarm
  checkout by hand; this repo doesn't write to that one).
- `swarm/saber.py` — model trend measurement across runs.
- `swarm/bench_swarm.py` — the shared task battery + flint-based runner both VECTOR and SABER build on.
- `swarm/board.py`, `swarm/report_leaderboard.py`, `swarm/metrics.py` — reporting/leaderboard helpers.
- `swarm/translation_tasks.py` — a multilingual variant of the task battery.
- `swarm/learn.py`, `flint.py` — copied from OpenRouterSwarm as of the split (duplicated, not shared,
  so this repo has no dependency on that one). Expect drift over time; re-sync by hand if needed.
- `consensus/` — the consensus-of-agents benchmark (panel vs. a single strong model), its findings,
  write-up, and saved sample runs.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
OPENROUTER_API_KEY=sk-or-... .venv/bin/python -m unittest discover -s tests -q
```

Needs `OPENROUTER_API_KEY` for anything that makes a real model call; the test suite itself runs offline.
