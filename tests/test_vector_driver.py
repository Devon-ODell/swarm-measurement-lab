#!/usr/bin/env python3
"""Tests for the VECTOR driver: multi-signal focus gain, blank-copy split
verification (broken children), the validate audit and the lesson-rendering
apply path. No flint calls, no network, no state files."""
import unittest
from pathlib import Path
from unittest import mock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "swarm"))

import vector as v
import bench_swarm


def _results(foc_ok, brd_ok, f_secs=3.0, b_secs=6.0, f_ch=0.01, b_ch=0.03):
    return {"focused": {"ok": foc_ok, "secs": f_secs, "charges": f_ch},
            "broad":   {"ok": brd_ok, "secs": b_secs, "charges": b_ch},
            "default": {"ok": brd_ok, "secs": b_secs, "charges": b_ch}}


class FocusGainTest(unittest.TestCase):
    def test_accuracy_flip_dominates(self):
        # focused passes slowly-expensive, broad fails fast-cheap: still a big
        # positive gain because accuracy is the primary signal.
        g = v._focus_gain_of(_results(True, False, f_secs=9.0, f_ch=0.05,
                                      b_secs=1.0, b_ch=0.01), None)
        self.assertIsNotNone(g)
        self.assertGreaterEqual(g, 0.4)   # 1 - 0.35 - 0.25 = 0.40

    def test_quickness_and_spend_move_score_within_bounds(self):
        fast = v._focus_gain_of(_results(True, False), None)
        slow = v._focus_gain_of(_results(True, False, f_secs=20.0, b_secs=1.0,
                                         f_ch=0.05, b_ch=0.01), None)
        self.assertGreater(fast, slow)
        self.assertLessEqual(fast, 1.0)
        self.assertGreaterEqual(slow, -1.0)

    def test_no_arms_return_none(self):
        self.assertIsNone(v._focus_gain_of({"focused": {"ok": None},
                                            "broad": {"ok": True}}, None))

    def test_zero_broad_time_is_ignored(self):
        g = v._focus_gain_of({"focused": {"ok": True, "secs": 2, "charges": 0.01},
                              "broad": {"ok": False, "secs": 0, "charges": 0.03}}, None)
        self.assertEqual(g, 1.0)          # 1 + 0 + 0.25*(0.03-0.01)/0.03

    def test_child_score_of_rolls_up_pass_rate(self):
        self.assertAlmostEqual(v._child_score_of({"a": {"ok": True},
                                                  "b": {"ok": False}}), 0.5)
        self.assertIsNone(v._child_score_of({"a": {"ok": None}}))
        self.assertIsNone(v._child_score_of({}))     # nothing measured, nothing judged


class SplitHintTest(unittest.TestCase):
    def test_lean_arm_is_numbered_steps_only(self):
        t = bench_swarm.TASKS[1]   # fn-sum-array
        p = v.contract_form(t, "lean")
        self.assertIn("math.js", p)
        self.assertIn("make exactly this true", p.lower())
        self.assertNotIn("ACCEPTANCE", p)
        self.assertNotIn("CONTRACT", p)

    def test_lean_does_not_misdirect_target_file(self):
        # smoke-status is about docs/STATUS.md; the detail legitimately mentions
        # game.js (the PASS line) but lean must NOT add a "TARGET FILE: game.js"
        # preamble or a CONTRACT: block (what the focused arm does).
        t = bench_swarm.TASKS[0]
        p = v.contract_form(t, "lean")
        self.assertIn("docs/STATUS.md", p)
        self.assertNotIn("TARGET FILE", p)
        self.assertNotIn("CONTRACT:", p)
        # and it does not tell the model to edit game.js
        self.assertNotIn("Edit game.js", p)

    def test_hint_lists_the_verifier_files(self):
        h = v.split_hint({"seeded": {"game.js": "x", "index.html": "y"},
                          "name": "smoke-status"}, {}, {})
        self.assertIn("game.js", h)
        self.assertIn("index.html", h)
        self.assertLessEqual(len(h), v.SPLIT_HINT_MAX_CHARS)

    def test_hint_flags_first_state_when_an_arm_passed(self):
        h = v.split_hint({"seeded": {"a.py": "1"}, "name": "t", "detail": "d"},
                         {"focused": {"ok": True}}, {"focused": {"ok": True}})
        self.assertIn("FIRST-STATE", h)

    def test_split_prompt_builds_hint_and_no_assumed_prior(self):
        # split_hint is the fencepost that becomes the VERIFIER GROUND TRUTH
        # block; assert it points the decomposer at first-state work.
        node = {"focused": {"ok": False}, "broad": {"ok": False}}
        hint = v.split_hint({"seeded": {"a.py": "1"}, "name": "t", "detail": "d"},
                            {}, node)
        self.assertIn("a.py", hint)
        self.assertIn("blank copy", hint)        # children never assume a prior state
        # and the actual DECOMPOSER prompt (stubbed) carries the same fuse
        with mock.patch.object(v, "_flint_turn",
                              return_value='[{"title": "t1", "detail": "d1"}]') as ft:
            specs, why = v.split_prompt({"title": "t", "detail": "d"}, "goal", "m",
                                        8, 60, node=node)
        self.assertEqual(why, "")
        self.assertEqual(len(specs), 1)
        self.assertIn("BLANK COPY", ft.call_args[0][0])


class BrokenSplitTest(unittest.TestCase):
    def _node(self, key, task, depth, kids, foc=None, brd=None, default=None, gain=0.0):
        n = {"key": key, "task": task, "form": "focused", "depth": depth,
             "children": kids, "focus_gain": gain}
        if foc is not None:
            n["focused"] = {"ok": foc}
            n["broad"] = {"ok": brd}
            n["default"] = {"ok": default if default is not None else brd}
        return n

    def test_broken_child_is_dropped_and_rejected(self):
        t = {"nodes": {}, "history": [], "state": {}}
        parent = self._node("spec:p:focused", "p", 0, [])
        t["nodes"][parent["key"]] = parent
        # split produces a child that cannot reproduce the parent's files:
        # it fails every arm, so pass-rate 0 < BROKEN_CHILD_MAX (0.45)
        child = self._node("spec:p::c:focused", "p::c", 1, [], foc=False, brd=False, gain=-0.3)
        t["nodes"][child["key"]] = child
        added = [child["key"]]
        with unittest.mock.patch.object(v, "_save_tree"):
            # replicate the trail after split_prompt: append child, then audit
            parent["children"].append(child["key"])
            dropped = []
            for ck in added:
                cn = t["nodes"].get(ck)
                score = v._child_score_of({f: cn.get(f) for f in ("broad", "default", "focused")})
                if score < v.BROKEN_CHILD_MAX:
                    dropped.append((ck, score))
            for ck, score in dropped:
                parent["children"].remove(ck)
                parent.setdefault("broken_splits", 0)
                parent["broken_splits"] += 1
        self.assertEqual(parent["children"], [])
        self.assertEqual(parent["broken_splits"], 1)

    def test_validate_marks_failed_split(self):
        t = {"nodes": {}, "history": [], "state": {}}
        t["nodes"]["spec:bad:focused"] = {
            "key": "spec:bad:focused", "task": "bad", "form": "focused", "depth": 0,
            "children": ["spec:bad::c:focused"], "split_failed": "children flunked"}
        t["nodes"]["spec:bad::c:focused"] = self._node("spec:bad::c:focused", "bad::c", 1, [],
                                                       foc=True, brd=True, gain=0.1)
        original = v._load_tree
        v._load_tree = lambda: t
        try:
            rc = v.cmd_validate(v.argparse.Namespace(depth=None))
        finally:
            v._load_tree = original
        self.assertEqual(rc, 1)

    def test_validate_ok_when_splits_reproduce(self):
        t = {"nodes": {}, "history": [], "state": {}}
        t["nodes"]["spec:ok:focused"] = {
            "key": "spec:ok:focused", "task": "ok", "form": "focused", "depth": 0,
            "children": ["spec:ok::c:focused"], "focus_gain": 0.5}
        t["nodes"]["spec:ok::c:focused"] = self._node("spec:ok::c:focused", "ok::c", 1, [],
                                                      foc=True, brd=True, gain=0.2)
        original = v._load_tree
        v._load_tree = lambda: t
        try:
            rc = v.cmd_validate(v.argparse.Namespace(depth=None))
        finally:
            v._load_tree = original
        self.assertEqual(rc, 0)


class RenderPreambleTest(unittest.TestCase):
    def test_render_cites_measured_signals(self):
        node = {"focus_gain": 0.7,
                "signals": {"accuracy_gain": 1, "secs_gain": 0.5, "charges_gain": 0.3},
                "lessons": [{"text": "keep subtasks focused", "mean": 0.9}]}
        out = v._render_preamble("smoke-status", node)
        self.assertIn("VECTOR:smoke-status:FOCUS", out)
        self.assertIn("0.70", out)
        self.assertIn("+50%", out)
        self.assertIn("+30%", out)
        self.assertLessEqual(len(out), v.LIMIT_ARM_HELD)

    def test_render_never_exceeds_cap(self):
        node = {"focus_gain": 0.7,
                "signals": {"accuracy_gain": 1, "secs_gain": 0.9, "charges_gain": 0.9},
                "lessons": [{"text": "x" * 300, "mean": 1.0}] * 5}
        out = v._render_preamble("slug", node)
        self.assertLessEqual(len(out), v.LIMIT_ARM_HELD)


class CliTest(unittest.TestCase):
    def test_build_parser_registers_validate(self):
        ap = v.build_parser()
        a = ap.parse_args(["validate"])
        self.assertEqual(a.cmd, "validate")

    def test_cmd_dry_prints_three_forms(self):
        a = v.argparse.Namespace(dry=True, task=1)
        self.assertEqual(v.cmd_run(a), 0)


if __name__ == "__main__":
    unittest.main()