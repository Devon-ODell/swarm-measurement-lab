"""Offline tests for the translation task battery (no network)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "swarm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consensus"))
import report
from translation_tasks import TRANSLATION_TASKS, translate_tasks

# canonical correct translations (from the battery's expected answers)
CORRECT = {
    "tr-en-es": "Hola, ¿cómo estás?",
    "tr-en-fr": "Le temps est agréable aujourd'hui.",
    "tr-en-de": "Guten Morgen, mein Freund.",
    "tr-en-pt": "Muito obrigado.",
    "tr-en-zh": "我喜欢编程。",
    "tr-en-ja": "また明日。",
    "tr-en-hi": "मैं आज बहुत खुश हूँ।",
    "tr-en-ar": "السلام عليكم",
    "tr-en-it": "Dov'è la stazione dei treni?",
    "tr-en-ru": "Удачи на экзамене.",
}
WRONG = {
    "tr-en-es": "Hello, how are you?",
    "tr-en-fr": "The weather is nice today.",
    "tr-en-zh": "我喜欢编码。",   # wrong char
    "tr-en-ja": "明日見て",        # chinese-style, not japanese
    "tr-en-ar": "سلام",           # truncated
}


def _run_verify(task, content):
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / next(iter(task["seeded"]))).write_text(content, encoding="utf-8")
    return task["verify"](d)


@pytest.mark.parametrize("name,want", CORRECT.items())
def test_translation_verifier_accepts_correct(name, want):
    task = next(t for t in TRANSLATION_TASKS if t["name"] == name)
    assert _run_verify(task, want)


@pytest.mark.parametrize("name,bad", WRONG.items())
def test_translation_verifier_rejects_wrong(name, bad):
    task = next(t for t in TRANSLATION_TASKS if t["name"] == name)
    assert not _run_verify(task, bad)


def test_translation_tasks_have_shape():
    for t in TRANSLATION_TASKS:
        assert set(t) >= {"name", "seeded", "prompt", "verify", "detail"}
        assert isinstance(t["prompt"], str)
        assert callable(t["verify"])
    # 10 languages
    assert len(TRANSLATION_TASKS) == 10
    langs = {t["name"].split("-")[2] for t in TRANSLATION_TASKS}
    assert langs == {"es", "fr", "de", "pt", "zh", "ja", "hi", "ar", "it", "ru"}


def test_report_separates_suites(tmp_path):
    """Records tagged code vs translation must not collide in aggregate."""
    p = tmp_path / "runs.jsonl"
    import json
    recs = [
        {"task": "code-x", "suite": "code", "n": 3, "majority": 2, "majority_ok": True,
         "models": ["m"], "agents": [{"model": "m", "ok": True, "secs": 1, "charges": 0}],
         "raw_frontier": {"ok": True, "secs": 1, "charges": 0.01}},
        {"task": "tr-x", "suite": "translation", "n": 3, "majority": 1, "majority_ok": False,
         "models": ["m"], "agents": [{"model": "m", "ok": False, "secs": 1, "charges": 0}],
         "raw_frontier": {"ok": False, "secs": 1, "charges": 0.02}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    tasks, ov = report.aggregate(report.load(p))
    assert len(tasks) == 2                    # both suites kept
    assert ov["panel_pass"] == 1              # one pass (code), one fail (translation)
    assert ov["panel_n"] == 2