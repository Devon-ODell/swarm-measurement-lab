"""Translation task battery for the consensus-of-agents benchmark.

Shares the exact task shape of swarm/bench_swarm.py TASKS:
  {"name", "seeded", "prompt", "verify", "detail"}

Each task asks the agent to translate a short piece of English text into another
language, writing a single file (a .txt or a language file). The verifier is
DETERMINISTIC: it compares the produced content against the expected translation
(exact, or whitespace-normalized for CJK/CRLF robustness). No LLM is used to
judge — this keeps the scoring honest and reproducible.

We deliberately use well-known translations so the "ground truth" is
uncontroversial and a human can read the battery in seconds.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# helper: whitespace-normalized exact match (handles CRLF, trailing spaces,
# and  full-width/CJK spacing differences that plague naive string equality)
# ---------------------------------------------------------------------------


def _norm(s):
    return " ".join(s.split())


def _txt_eq(path, want):
    return path.exists() and _norm(path.read_text().strip()) == _norm(want.strip())


TRANSLATION_TASKS = [
    {
        "name": "tr-en-es",
        "seeded": {"mensaje.txt": "Hello, how are you?\n"},
        "prompt": (
            "Edit mensaje.txt: translate its content to Spanish. Write ONLY the "
            "translated sentence, nothing else. Correct answer: 'Hola, ¿cómo estás?'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "mensaje.txt", "Hola, ¿cómo estás?"),
        "detail": "EN→ES: 'Hello, how are you?' -> 'Hola, ¿cómo estás?'",
    },
    {
        "name": "tr-en-fr",
        "seeded": {"message.txt": "The weather is nice today.\n"},
        "prompt": (
            "Edit message.txt: translate its content to French. Write ONLY the "
            "translated sentence, nothing else. Correct answer: "
            "'Le temps est agréable aujourd'hui.' or 'Il fait beau aujourd'hui.'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "message.txt", "Le temps est agréable aujourd'hui.")
        or _txt_eq(cwd / "message.txt", "Il fait beau aujourd'hui."),
        "detail": "EN→FR: 'The weather is nice today.' -> ('Le temps est agréable aujourd'hui.' | 'Il fait beau aujourd'hui.')",
    },
    {
        "name": "tr-en-de",
        "seeded": {"nachricht.txt": "Good morning, my friend.\n"},
        "prompt": (
            "Edit nachricht.txt: translate its content to German. Write ONLY the "
            "translated sentence, nothing else. Correct answer: 'Guten Morgen, mein Freund.'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "nachricht.txt", "Guten Morgen, mein Freund."),
        "detail": "EN→DE: 'Good morning, my friend.' -> 'Guten Morgen, mein Freund.'",
    },
    {
        "name": "tr-en-pt",
        "seeded": {"mensagem.txt": "Thank you very much.\n"},
        "prompt": (
            "Edit mensagem.txt: translate its content to Portuguese. Write ONLY the "
            "translated sentence, nothing else. Correct answer: 'Muito obrigado.'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "mensagem.txt", "Muito obrigado."),
        "detail": "EN→PT: 'Thank you very much.' -> 'Muito obrigado.'",
    },
    {
        "name": "tr-en-zh",
        "seeded": {"xiao_xi.txt": "I love programming.\n"},
        "prompt": (
            "Edit xiao_xi.txt: translate its content to Simplified Chinese. Write ONLY "
            "the translated sentence, nothing else. Correct answer: '我喜欢编程。'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "xiao_xi.txt", "我喜欢编程。"),
        "detail": "EN→ZH: 'I love programming.' -> '我喜欢编程。'",
    },
    {
        "name": "tr-en-ja",
        "seeded": {"messeeji.txt": "See you tomorrow.\n"},
        "prompt": (
            "Edit messeeji.txt: translate its content to Japanese. Write ONLY the "
            "translated sentence, nothing else. Correct answer: 'また明日。' or "
            "'また明日ね。'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "messeeji.txt", "また明日。")
        or _txt_eq(cwd / "messeeji.txt", "また明日ね。"),
        "detail": "EN→JA: 'See you tomorrow.' -> 'また明日。'",
    },
    {
        "name": "tr-en-hi",
        "seeded": {"sandesh.txt": "I am very happy today.\n"},
        "prompt": (
            "Edit sandesh.txt: translate its content to Hindi. Write ONLY the "
            "translated sentence, nothing else. Correct answer: 'मैं आज बहुत खुश हूँ।'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "sandesh.txt", "मैं आज बहुत खुश हूँ।"),
        "detail": "EN→HI: 'I am very happy today.' -> 'मैं आज बहुत खुश हूँ।'",
    },
    {
        "name": "tr-en-ar",
        "seeded": {"risala.txt": "Peace be upon you.\n"},
        "prompt": (
            "Edit risala.txt: translate its content to Arabic. Write ONLY the "
            "translated sentence, nothing else. Correct answer: 'السلام عليكم' "
            "(the standard greeting; punctuation optional)."
        ),
        "verify": lambda cwd: _txt_eq(cwd / "risala.txt", "السلام عليكم")
        or _txt_eq(cwd / "risala.txt", "السلام عليكم."),
        "detail": "EN→AR: 'Peace be upon you.' -> 'السلام عليكم'",
    },
    {
        "name": "tr-en-it",
        "seeded": {"messaggio.txt": "Where is the train station?\n"},
        "prompt": (
            "Edit messaggio.txt: translate its content to Italian. Write ONLY the "
            "translated sentence, nothing else. Correct answer: "
            "'Dov'è la stazione dei treni?' or 'Dov'è la stazione ferroviaria?'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "messaggio.txt", "Dov'è la stazione dei treni?")
        or _txt_eq(cwd / "messaggio.txt", "Dov'è la stazione ferroviaria?"),
        "detail": "EN→IT: 'Where is the train station?' -> 'Dov'è la stazione dei treni?'",
    },
    {
        "name": "tr-en-ru",
        "seeded": {"soobshchenie.txt": "Good luck with your exam.\n"},
        "prompt": (
            "Edit soobshchenie.txt: translate its content to Russian. Write ONLY the "
            "translated sentence, nothing else. Correct answer: 'Удачи на экзамене.'"
        ),
        "verify": lambda cwd: _txt_eq(cwd / "soobshchenie.txt", "Удачи на экзамене."),
        "detail": "EN→RU: 'Good luck with your exam.' -> 'Удачи на экзамене.'",
    },
]

# convenience: translate each task's source -> target language label
TRANSLATION_LABELS = {t["name"]: t["name"].split("-")[1].upper() for t in TRANSLATION_TASKS}


def translate_tasks():
    """Return the translation battery as a list (kept simple for reuse)."""
    return list(TRANSLATION_TASKS)