#!/usr/bin/env python3
"""VECTOR — recursive benchmark-driven task decomposition (command entry).

Wraps swarm/vector.py so the harness, the ledger and the bandit ledger all live
under one import surface. Run:

    swarm vector run --task 2 --model dots-studio/dots-3-note-preview:free
    swarm vector trend
    swarm vector apply smoke-status
    swarm vector loop --tasks 1,2,3

The loop re-runs the battery forever, writes plan/journal/trend evidence under
swarm/vector_state/, and reinforces the `decomposer` bandit arm in learn.json
plus the playbook — so decomposition gets better as it measures itself.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "swarm"))

import vector as _vector  # noqa: E402


def main():
    return _vector.main()


if __name__ == "__main__":
    sys.exit(main())