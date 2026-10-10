#!/usr/bin/env python3
"""
flint — a tiny terminal coding agent that runs on free OpenRouter models
(default: inclusionAI Ling 3.0 Flash Fin), and on any local server that
speaks OpenAI-compatible chat completions or Anthropic messages:

    python flint.py -m ollama:qwen2.5-coder:7b     # Ollama (no key needed)
    python flint.py --provider lm-studio -m ...     # LM Studio / MLX / llama.cpp

A `backend:` prefix or a known vendor slug (`anthropic/...`) picks the provider;
endpoint overrides live in the environment (OLLAMA_BASE_URL, ...) or in
~/.flint/providers.json. Local providers skip the OpenRouter request budget.

    # Set OPENROUTER_API_KEY=sk-or-... in .env beside this script, or:
    export OPENROUTER_API_KEY=sk-or-...
    python flint.py                    # interactive
    python flint.py -p "fix the tests" # one-shot, prints the answer, exits
"""
import argparse
import calendar
import datetime
import difflib
import fnmatch
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import urllib.error
import urllib.request
try:
    from dotenv import load_dotenv
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
except ModuleNotFoundError as e:
    sys.exit(f"flint: missing Python package '{e.name}' — run flint with the project's virtualenv:\n"
             f"  python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt\n"
             f"  .venv/bin/python flint.py")

# The provider registry lives beside flint.py (swarm/providers.py). It is imported at
# module level only when present; the repo root is on sys.path once main() runs, and the
# daemon/tests load it directly.
try:
    from swarm import providers as _providers
except (ImportError, ModuleNotFoundError):
    _providers = None

try:
    import readline  # noqa: F401  (arrow keys + history in input())
except ImportError:
    pass
try:
    import fcntl  # cross-process lock for the shared request budget (macOS/Linux)
except ImportError:
    fcntl = None

# ─── config ──────────────────────────────────────────────────────────────────
# Read the .env beside this script, even when launched from another directory.
# Explicitly exported environment variables take precedence. Swarm workers run
# in a sandbox that denies reading .env; they receive the key via environment.
try:
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
except OSError:
    pass

DEFAULT_MODEL = os.environ.get("FLINT_MODEL", "inclusionai/ling-3.0-flash-fin:free")
FALLBACKS = [m.strip() for m in os.environ.get("FLINT_FALLBACKS", "").split(",") if m.strip()]
MAX_STEPS = int(os.environ.get("FLINT_MAX_STEPS", "40"))  # tool rounds per turn
MAX_TOOL_OUTPUT = 20_000                                   # chars sent back to the model
CONTEXT_WARN_TOKENS = 200_000
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
                ".pytest_cache", "dist", "build", ".next", ".idea", ".DS_Store", "target"}
NEEDS_APPROVAL = {"write_file", "edit_file", "bash"}
STATE_DIR = Path(os.environ.get("FLINT_HOME", "~/.flint")).expanduser()
RPM_LIMIT = int(os.environ.get("FLINT_RPM", "18"))  # OpenRouter caps :free models at 20/min
OPENROUTER = "https://openrouter.ai/api/v1"  # the default provider's base URL (providers.py reads this)
CORPUS_DB = Path(os.environ.get("FLINT_CORPUS_DB", "~/.flint/corpus.db")).expanduser()
# Headless shell commands do not inherit credentials from the environment.
SECRET_ENV = re.compile(r"API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE_KEY|CREDENTIAL", re.I)

# Path of the machine-readable request sidecar (work order §1). Every HTTP request
# attempt — the initial call and every provider retry — appends one JSON line here
# *before* it is made, so a SIGKILLed turn still leaves an exact count behind and the
# daemon never has to parse human log lines. The daemon names the file per turn and
# banks the total after flint exits.
REQUEST_FILE = os.environ.get("FLINT_REQUEST_FILE", "")

ACCENT = "cyan"
console = Console(highlight=False)


# ─── tools ───────────────────────────────────────────────────────────────────
def _p(path):
    return Path(path).expanduser()


def _truncate(s, limit=MAX_TOOL_OUTPUT):
    if len(s) <= limit:
        return s
    half = limit // 2
    return f"{s[:half]}\n\n… [{len(s) - limit} chars truncated] …\n\n{s[-half:]}"


def read_file(path, offset=1, limit=400):
    p = _p(path)
    if not p.exists():
        return f"Error: {path} does not exist."
    if p.is_dir():
        return f"Error: {path} is a directory. Use list_files."
    if b"\x00" in p.read_bytes()[:8192]:
        return f"{path} looks like a binary file; not shown."
    lines = p.read_text(errors="replace").splitlines()
    if not lines:
        return "(empty file)"
    start = max(1, int(offset))
    end = min(len(lines), start + int(limit) - 1)
    body = "\n".join(f"{i:>5}\t{lines[i - 1]}" for i in range(start, end + 1))
    if start > 1 or end < len(lines):
        body += f"\n[showing lines {start}-{end} of {len(lines)}]"
    return body


def read_files(paths, limit=400):
    """Read several files in one call.

    A weak model makes one tool call per round, so reading four files costs four of the twelve
    rounds a swarm turn has. One call for the lot leaves the rounds for the edit."""
    if isinstance(paths, str):          # text markup, or a model that ignored the array type
        try:
            parsed = json.loads(paths)
        except ValueError:
            parsed = None
        paths = parsed if isinstance(parsed, list) else re.split(r"[,\n]", paths)
    paths = [str(x).strip().strip('"\'') for x in (paths or [])]
    paths = [x for x in paths if x][:20]
    if not paths:
        return "Error: read_files needs at least one path."
    return "\n\n".join(f"=== {x} ===\n{read_file(x, 1, limit)}" for x in paths)


def write_file(path, content):
    check_write_scope(path)
    p = _p(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    p.write_text(content)
    return f"{'Overwrote' if existed else 'Created'} {path} ({len(content.splitlines())} lines)."


def _plan_edit(path, old_str, new_str, replace_all=False):
    """Returns (old_text, new_text, error)."""
    p = _p(path)
    if not p.exists():
        return None, None, f"Error: {path} does not exist. Use write_file to create it."
    text = p.read_text(errors="replace")
    n = text.count(old_str)
    if n == 0:
        return None, None, ("Error: old_str not found. It must match the file exactly, "
                            "including whitespace and indentation. Re-read the file and retry.")
    if n > 1 and not replace_all:
        return None, None, (f"Error: old_str appears {n} times. Include more surrounding "
                            "lines to make it unique, or set replace_all=true.")
    new_text = text.replace(old_str, new_str) if replace_all else text.replace(old_str, new_str, 1)
    return text, new_text, None


def edit_file(path, old_str, new_str, replace_all=False):
    check_write_scope(path)
    old, new, err = _plan_edit(path, old_str, new_str, replace_all)
    if err:
        return err
    _p(path).write_text(new)
    n = old.count(old_str) if replace_all else 1
    return f"Edited {path} ({n} replacement{'s' if n != 1 else ''})."


def check_write_scope(path):
    raw = os.environ.get("FLINT_ALLOWED_PATHS")
    if not raw:
        return
    scope = json.loads(raw)
    root = Path.cwd().resolve()
    target = _p(path).resolve()
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError:
        raise PermissionError("task writes must remain inside its worktree")
    if not any(relative == p or (p.endswith("/") and relative.startswith(p)) for p in scope):
        raise PermissionError(f"{relative} is outside this task's allowed_paths: {scope}")


def list_files(pattern="**/*", path="."):
    root = _p(path)
    if not root.is_dir():
        return f"Error: {path} is not a directory."
    pat2 = pattern[3:] if pattern.startswith("**/") else pattern
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for f in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, f), root)
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pat2):
                out.append(rel)
                if len(out) >= 300:
                    return "\n".join(out) + "\n[truncated at 300 files — narrow the pattern]"
    return "\n".join(out) or "No files matched."


def search(pattern, path=".", glob=None):
    if shutil.which("rg"):
        cmd = ["rg", "-n", "--no-heading", "--color", "never", "-e", pattern]
        if glob:
            cmd += ["--glob", glob]
        cmd.append(path)
        r = subprocess.run(cmd, capture_output=True, text=True)
        lines = r.stdout.splitlines()
    else:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: bad regex ({e})"
        lines = []
        for rel in list_files(glob or "**/*", path).splitlines():
            fp = _p(path) / rel
            try:
                for i, line in enumerate(fp.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        lines.append(f"{os.path.join(path, rel)}:{i}:{line}")
            except (OSError, UnicodeDecodeError):
                continue
            if len(lines) > 200:
                break
    if not lines:
        return "No matches."
    extra = f"\n[{len(lines) - 200} more matches not shown]" if len(lines) > 200 else ""
    return "\n".join(lines[:200]) + extra


BASH_TIMEOUT = int(os.environ.get("FLINT_BASH_TIMEOUT", "120"))


def bash(command, timeout=None):
    """Run a shell command. The default timeout is FLINT_BASH_TIMEOUT, which a swarm sets to
    its own test timeout: the suite it asks the model to run can take longer than 120s."""
    timeout = BASH_TIMEOUT if timeout is None else timeout
    shell = os.environ.get("SHELL") or "/bin/sh"
    env = None
    if os.environ.get("FLINT_HEADLESS"):
        env = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    try:
        r = subprocess.run(command, shell=True, executable=shell, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=int(timeout), env=env)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s."
    out = r.stdout or ""
    if r.stderr:
        out += ("\n" if out else "") + "[stderr]\n" + r.stderr
    return f"exit code {r.returncode}\n{out.strip() or '(no output)'}"


def study(query, k=5):
    """Search the local study corpus: MIT OCW lecture cards first, then source pages, with
    absolute paths a read_file call can open. Local BM25 lookup; costs no API quota."""
    try:
        from swarm import mit_corpus
        text = mit_corpus.study(query, max(1, min(int(k), 10)), db=CORPUS_DB)
    except Exception as e:
        return f"Error: study corpus unavailable ({type(e).__name__}: {e})"
    return text or "No matches in the study corpus. Try different or more specific terms."


# Max bytes a web_fetch will keep; anything past this is cut with a marker.
WEB_FETCH_LIMIT = 40_000


def web_fetch(url):
    """Fetch one http(s) URL and return the text, sized for a tool call.

    Phase 3 web_fetch: the model may read a release note, a docs page or an issue
    thread instead of guessing. HTML is stripped to text; binary or oversized pages
    are cut off rather than ballooning the turn.
    """
    import urllib.error
    import urllib.request
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return "Error: web_fetch needs an http(s) URL."
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "flint/1.0 (research fetch)"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read(WEB_FETCH_LIMIT + 1024)
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} fetching {url}"
    except (urllib.error.URLError, OSError, ValueError) as e:
        return f"Error: could not fetch {url}: {e}"
    text = raw.decode("utf-8", "replace")
    if b"<" in raw[:512]:
        text = re.sub(r"(?is)<(script|style|svg)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
    if len(text) > WEB_FETCH_LIMIT:
        text = text[:WEB_FETCH_LIMIT] + f"\n… [{len(raw) - WEB_FETCH_LIMIT} more bytes not shown]"
    return text.strip() or f"(empty page at {url})"


TOOLS = {"read_file": read_file, "read_files": read_files, "write_file": write_file,
         "edit_file": edit_file, "list_files": list_files, "search": search, "bash": bash,
         "study": study, "web_fetch": web_fetch}


def _schema(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOL_SCHEMAS = [
    _schema("read_file", "Read a text file. Returns numbered lines. Use offset/limit for big files.",
            {"path": {"type": "string"},
             "offset": {"type": "integer", "description": "1-based first line (default 1)"},
             "limit": {"type": "integer", "description": "max lines (default 400)"}}, ["path"]),
    _schema("read_files", "Read several files in one call — use this instead of one read_file per "
                          "round. Each file appears under a `=== path ===` header with numbered lines.",
            {"paths": {"type": "array", "items": {"type": "string"},
                       "description": 'the files to read, e.g. ["src/game.js", "tests/test_game.py"]'},
             "limit": {"type": "integer", "description": "max lines per file (default 400)"}}, ["paths"]),
    _schema("write_file", "Create a file or fully overwrite one. Prefer edit_file for changes.",
            {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    _schema("edit_file", "Replace old_str with new_str in a file. old_str must match exactly "
                         "(whitespace included) and be unique unless replace_all is true. "
                         "Always read_file first.",
            {"path": {"type": "string"}, "old_str": {"type": "string"},
             "new_str": {"type": "string"}, "replace_all": {"type": "boolean"}},
            ["path", "old_str", "new_str"]),
    _schema("list_files", "List files matching a glob like '**/*.py' (skips .git, node_modules, venvs).",
            {"pattern": {"type": "string"}, "path": {"type": "string"}}, []),
    _schema("search", "Regex search file contents. Returns path:line:text.",
            {"pattern": {"type": "string"}, "path": {"type": "string"},
             "glob": {"type": "string", "description": "optional file filter, e.g. '*.go'"}},
            ["pattern"]),
    _schema("bash", "Run a shell command in the project directory. Each call is a fresh shell, "
                    "so chain with && if you need cd. Use for tests, git, builds, installs.",
            {"command": {"type": "string"},
             "timeout": {"type": "integer",
                         "description": f"seconds (default {BASH_TIMEOUT})"}}, ["command"]),
    _schema("study", "Search the local study corpus (MIT OCW lecture cards, slides, notes, problem sets "
                     "and transcripts on deep learning and machine learning, probability, matrix calculus, discrete math and combinatorics, algorithms and Python, mathematical finance, fintech, blockchain, risk and decision analysis, venture finance, microeconomics, game theory, public finance, perception and psychology, math for computer science, cryptography (interactive proofs, SNARGs), poker strategy, semiconductor microfabrication, game design; plus past code reviews) for techniques "
                     "and reference material. Hits under flint/cards/ name the course and lecture; follow "
                     "their page links for source text. Free: no API quota.",
            {"query": {"type": "string", "description": "specific terms, e.g. 'dynamic programming subproblem memo'"},
             "k": {"type": "integer", "description": "number of excerpts (default 5, max 10)"}}, ["query"]),
    _schema("web_fetch", "Fetch one http(s) URL and return its text (release notes, docs, an issue "
                         "thread). Size-capped; surfaces HTTP errors instead of guessing.",
            {"url": {"type": "string", "description": "the http(s) URL to fetch"}}, ["url"]),
]


def _corpus_ready():
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{CORPUS_DB}?mode=ro", uri=True)
        try:
            return con.execute("SELECT 1 FROM files WHERE nchunks > 0 LIMIT 1").fetchone() is not None
        finally:
            con.close()
    except Exception:
        return False


# ─── system prompt ───────────────────────────────────────────────────────────
def build_system_prompt():
    cwd = os.getcwd()
    git = ""
    if shutil.which("git"):
        r = subprocess.run("git rev-parse --abbrev-ref HEAD && git status --short | head -20",
                           shell=True, capture_output=True, text=True, cwd=cwd)
        if r.returncode == 0:
            git = f"\nGit branch and status:\n{r.stdout.strip()}"
    project = ""
    for name in ("FLINT.md", "AGENTS.md"):
        f = Path(cwd) / name
        if f.exists():
            project += f"\n\nProject instructions from {name}:\n{f.read_text(errors='replace')[:8000]}"
    if _corpus_ready():
        project += ("\n\nA `study` tool searches local MIT OpenCourseWare material (deep learning and machine learning, probability, matrix calculus, discrete math and combinatorics, algorithms and Python, mathematical finance, fintech, blockchain, risk and decision analysis, venture finance, microeconomics, game theory, public finance, perception and psychology, math for computer science, cryptography (interactive proofs, SNARGs), poker strategy, semiconductor microfabrication, game design) "
                    "and past code reviews. When a problem calls for a known algorithm, model or technique, "
                    "look it up there and cite the course and lecture it came from.")
    return f"""You are flint, a capable agent running in the user's terminal.
You help with software work and with everyday plain-language requests — questions, explanations, writing, translations, data — by reading, searching, editing files and running commands when a task needs them.

How to work:
- Interpret requests as intent, not literally: a loosely worded ask still has a concrete outcome; restate what is being asked before acting.
- Separate \"answer me\" from \"do it\": a question or a chat turn that needs no file change gets a direct answer — do not edit files just because tools exist.
- Match the person's language and tone in replies, and translate text when asked.
- Use your general knowledge for non-code work (explanations, summaries, copy, analysis); reach for web_fetch or study when an answer needs facts you don't have.
- Investigate before acting: list_files / search / read_file to understand the code first.
- Reading several files? Pass them all to read_files at once rather than one read_file per round.
- Always read_file before edit_file. Keep old_str small but unique, copied exactly.
- Make the smallest change that solves the task. Don't rewrite files that only need a tweak.
- After changing code, verify it: run the tests, the build, or the script with bash.
- If a tool returns an error, read it and fix your approach instead of repeating the same call.
- Never print or exfiltrate secrets, keys, or wallet material you come across.
- If the user denies a tool call, respect it and ask or adjust.
- Replies are shown in a terminal: be brief, use markdown sparingly, no filler.
- When the task is done, say what you changed in 1-3 sentences.

Environment:
- Working directory: {cwd}
- OS: {platform.system()} {platform.release()} ({platform.machine()})
- Date: {datetime.date.today().isoformat()}{git}{project}"""


# ─── rate limits ─────────────────────────────────────────────────────────────
class DailyCapReached(Exception):
    def __init__(self, reset_at):
        super().__init__("daily free-model request cap reached")
        self.reset_at = reset_at


class OutOfCredits(Exception):
    pass


class StepLimitReached(Exception):
    pass


class ExplorationExhausted(Exception):
    """A turn spent 70% of its round budget without editing; stopped before the full timeout."""


# Some providers pass a model's native tool-call markup through as reply text instead of
# translating it into tool_calls. qwen3-coder's `<function=name><parameter=k>v</parameter>`
# is the common case. Unparsed, the edit never runs, the turn ends looking like a final
# answer, and the swarm scores the attempt "no change" — which then gets retried and split.
_TEXT_CALL = re.compile(r"<function=([\w.\-]+)>(.*?)(?:</function>|(?=<function=)|\Z)", re.S)
_TEXT_PARAM = re.compile(r"<parameter=([\w.\-]+)>(.*?)</parameter>", re.S)
_TEXT_WRAP = re.compile(r"</?tool_call>")
# Models writing their own markup also invent their own parameter names. The log for
# w0-implementer-1790428940089493000 has `<parameter=file>` where the schema says `path`, which
# on its own turned a finished edit into a "no change" attempt. An alias is only taken when the
# tool has no parameter by that name and does have the one it stands for.
_PARAM_ALIASES = {"file": "path", "filename": "path", "filepath": "path", "file_path": "path",
                  "old": "old_str", "new": "new_str", "old_string": "old_str",
                  "new_string": "new_str", "old_text": "old_str", "new_text": "new_str",
                  "text": "content", "contents": "content"}


def recover_text_calls(content, schemas):
    """Tool calls a model wrote into its reply text, as (calls, text_without_them).

    Only a call to a known tool with every required parameter closed is recovered, so a
    reply cut off mid-edit is left alone rather than half-applied. Each value loses the one
    newline the format wraps it in, and no more: an edit's old_str must match byte for byte.
    """
    if "<function=" not in (content or ""):
        return [], content
    spec = {}
    for t in schemas:
        f = t["function"]
        params = f.get("parameters", {})
        spec[f["name"]] = ({k: v.get("type") for k, v in params.get("properties", {}).items()},
                           set(params.get("required", [])))
    calls = []
    for m in _TEXT_CALL.finditer(content):
        name, body = m.group(1), m.group(2)
        if name not in spec:
            continue
        types, required = spec[name]
        args = {}
        for p in _TEXT_PARAM.finditer(body):
            key, val = p.group(1), p.group(2)
            if val.startswith("\n"):
                val = val[1:]
            if val.endswith("\n"):
                val = val[:-1]
            args[key] = val
        for alias, real in _PARAM_ALIASES.items():
            if alias in args and alias not in types and real in types and real not in args:
                args[real] = args.pop(alias)
        for key, val in list(args.items()):
            if types.get(key) in ("integer", "number", "boolean", "array"):
                try:
                    args[key] = json.loads(val.strip().lower() if types[key] == "boolean" else val.strip())
                except ValueError:
                    pass
        if required <= set(args):
            calls.append({"id": "", "name": name, "args": json.dumps(args)})
    if not calls:
        return [], content
    return calls, _TEXT_WRAP.sub("", _TEXT_CALL.sub("", content)).strip()


class IncompleteResponse(Exception):
    def __init__(self, msg, finish_reason=None):
        super().__init__(msg)
        self.finish_reason = finish_reason


class SpendExhausted(Exception):
    """This swarm's daily budget for paid models is used up.

    Free models never touch it. The budget exists so that a swarm which falls back to
    paid models when the free allowance runs out cannot quietly run up a bill."""

    def __init__(self, spent, cap):
        self.spent, self.cap = spent, cap
        super().__init__(f"daily paid-model budget spent (${spent:.4f} of ${cap:.2f})")


class BudgetPaused(Exception):
    pass


class WalletEmpty(Exception):
    """The dollar allowance handed down for this run (FLINT_WALLET) is used up."""


def wallet_from_env():
    """The paid allowance a parent handed down, or None. The swarm daemon never sets one: only
    the editor bridge does, so only editor-initiated turns can reach the account's credits."""
    if not os.environ.get("FLINT_WALLET"):
        return None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from swarm.wallet import Wallet
        return Wallet.from_env()
    except Exception:
        return None


class ProviderUnavailable(Exception):
    """The model or its providers cannot serve requests right now (overload, 404, 5xx)."""


class ProviderBusy(ProviderUnavailable):
    """The model works but is rate-limited upstream (429): try another model, or this one shortly."""


class ModelUnavailable(ProviderUnavailable):
    """This key cannot use this model at all (403 forbidden, 404 no such model/endpoint).
    Waiting will not help: the model has to be replaced in the pool."""


def _next_utc_midnight():
    now = datetime.datetime.now(datetime.timezone.utc)
    return (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                      microsecond=0).timestamp()


def fmt_reset(ts):
    local = datetime.datetime.fromtimestamp(ts)
    mins = max(0, int((ts - time.time()) // 60))
    return f"{local:%H:%M} local (in {mins // 60}h {mins % 60}m)"


def spend_period(reset_day=None, now=None):
    """The name of the budget window containing `now`: the date the window began.

    Without a reset day the window is the UTC calendar day, which is what a daily
    budget has always meant. With one (say 5) the window is the month that starts on
    the 5th, so `2026-09-26` and `2026-10-04` are both the window named `2026-09-05`.
    A month too short for the day — the 31st of February — starts on its last day
    instead, so no window is ever skipped."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    today = now.date() if isinstance(now, datetime.datetime) else now
    if reset_day in (None, ""):
        return today.isoformat()
    day = max(1, min(31, int(reset_day)))

    def began(year, month):
        return datetime.date(year, month, min(day, calendar.monthrange(year, month)[1]))

    start = began(today.year, today.month)
    if today < start:      # this month's reset has not happened yet; we are in the last one
        year, month = (today.year - 1, 12) if today.month == 1 else (today.year, today.month - 1)
        start = began(year, month)
    return start.isoformat()


class Spend:
    """Dollars charged to paid models over one budget window.

    `FLINT_SPEND_FILE` names the ledger, `FLINT_SPEND_CAP` the window's limit in
    dollars and `FLINT_SPEND_RESET_DAY` the day of the month the window restarts;
    without a reset day the window is the UTC day, which is what this meant before.
    Without a file there is no dollar accounting and nothing is gated, so single-agent
    flint behaves exactly as before. Point several swarms at one file to hold them to a
    shared budget, or give each its own to hold separate ones.

    Costs are what OpenRouter actually charged, read from the usage of each reply,
    not an estimate from token counts."""

    def __init__(self, path=None, cap=None, reset_day=None):
        path = path if path is not None else os.environ.get("FLINT_SPEND_FILE")
        self.path = Path(path).expanduser() if path else None
        cap = cap if cap is not None else os.environ.get("FLINT_SPEND_CAP")
        self.cap = float(cap) if cap not in (None, "") else None
        reset = reset_day if reset_day is not None else os.environ.get("FLINT_SPEND_RESET_DAY")
        self.reset_day = int(reset) if str(reset if reset is not None else "").strip().isdigit() else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.lock = self.path.with_suffix(".lock")

    def _txn(self, fn):
        if not self.path:
            return None
        with open(self.lock, "w") as lf:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                d = json.loads(self.path.read_text()) if self.path.exists() else {}
            except (json.JSONDecodeError, OSError):
                d = {}
            period = spend_period(self.reset_day)
            if d.get("period") != period:
                d.update(period=period, usd=0.0, requests=0, recent=[])
            out = fn(d)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(self.path)
            return out

    def today(self):
        return self._txn(lambda d: float(d.get("usd", 0.0))) or 0.0

    def add(self, usd, model=""):
        """Record what a reply cost. Called once per paid reply, after it arrives."""
        try:
            usd = float(usd)
        except (TypeError, ValueError):
            return
        if usd <= 0:
            return

        def step(d):
            d["usd"] = round(float(d.get("usd", 0.0)) + usd, 6)
            d["requests"] = int(d.get("requests", 0)) + 1
            # A cap you cannot audit is a cap you cannot trust: without this, a single
            # wrong figure disappears into the running total and cannot be told from a
            # month of honest usage. Bounded so the file cannot grow without limit.
            recent = [r for r in d.get("recent", []) if isinstance(r, dict)]
            recent.append({"iso": datetime.datetime.now().isoformat(timespec="seconds"),
                           "model": str(model or ""), "usd": usd})
            d["recent"] = recent[-50:]
        self._txn(step)

    def remaining(self):
        return None if self.cap is None or not self.path else max(0.0, self.cap - self.today())

    def check(self):
        """Raise before a paid request the window's budget can no longer cover."""
        if self.cap is None or not self.path:
            return
        spent = self.today()
        if spent >= self.cap:
            raise SpendExhausted(spent, self.cap)


class Throttle:
    """One request budget shared by every flint process on this machine.

    Keeps a rolling 60s window under RPM_LIMIT, counts requests per UTC day,
    and remembers a daily-cap block so a swarm stops instantly instead of each
    agent separately burning a failed request to discover it."""

    def __init__(self, rpm=RPM_LIMIT):
        if rpm < 1:
            raise ValueError("FLINT_RPM must be positive")
        self.rpm = rpm
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = STATE_DIR / "requests.json"
        self.lock = STATE_DIR / "requests.lock"

    def _txn(self, fn):
        with open(self.lock, "w") as lf:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                d = json.loads(self.path.read_text()) if self.path.exists() else {}
            except (json.JSONDecodeError, OSError):
                d = {}
            day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
            if d.get("day") != day:
                d.update(day=day, count=0)
            out = fn(d)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(self.path)
            return out

    def acquire(self, on_wait=None, paid=False, local=False):
        """Wait for a slot in the rolling per-minute window.

        A paid request is the point of the fallback, so it is not held back by the free
        allowance and does not count against it: the daily-cap block and the swarm's
        request budget both describe free capacity only. It still takes a slot in the
        per-minute window, which keeps a swarm that has switched to paid models spending
        at the same deliberate pace rather than as fast as the network allows.

        A local request is none of OpenRouter's business: no RPM window, no daily cap, no
        swarm budget. Local servers are the user's own machine, so they return at once.
        """
        if local:
            return
        while True:
            def step(d):
                now = time.time()
                if not paid:
                    if d.get("blocked_until", 0) > now:
                        raise DailyCapReached(d["blocked_until"])
                    policy = os.environ.get("FLINT_SWARM_BUDGET")
                    if policy:
                        from swarm.budget import Budget
                        allowed, delay, reason = Budget(**json.loads(policy)).check(state=d)
                        if not allowed:
                            raise BudgetPaused(f"{reason}; check again in {delay:.0f}s")
                recent = [t for t in d.get("recent", []) if now - t < 60]
                d["recent"] = recent
                if len(recent) < self.rpm:
                    recent.append(now)
                    if not paid:
                        d["count"] = d.get("count", 0) + 1
                    return 0
                return 60 - (now - recent[0]) + 0.2
            wait = self._txn(step)
            if wait <= 0:
                return
            if on_wait:
                on_wait(wait)
            time.sleep(wait)

    def block(self, until):
        self._txn(lambda d: d.update(blocked_until=until))

    def clear_block(self):
        self._txn(lambda d: d.pop("blocked_until", None))

    def today(self):
        return self._txn(lambda d: d.get("count", 0))


def _error_parts(e):
    """(error dict, metadata dict, lower-cased headers) of an OpenRouter API error."""
    body = getattr(e, "body", None)
    err = body.get("error", body) if isinstance(body, dict) else {}
    err = err if isinstance(err, dict) else {}
    meta = err.get("metadata") if isinstance(err.get("metadata"), dict) else {}
    headers = {k.lower(): v for k, v in (meta.get("headers") or {}).items()}
    resp = getattr(e, "response", None)
    if resp is not None:
        headers.update({k.lower(): v for k, v in resp.headers.items()})
    return err, meta, headers


def classify_429(e):
    """Return (kind, reset_at, message). kind: daily | minute | provider | unknown.

    OpenRouter's own limits are named in `message`. An upstream provider's throttle arrives as
    the generic "Provider returned error" with the provider's text in metadata.raw ("… is
    temporarily rate-limited upstream"); that text is shown but never read as the account's
    daily cap, which would stop every model."""
    err, meta, headers = _error_parts(e)
    msg = str(err.get("message") or getattr(e, "message", "") or e)
    reset_at = None
    try:
        r = float(headers.get("x-ratelimit-reset", ""))
        reset_at = r / 1000 if r > 1e11 else r  # header is usually epoch milliseconds
    except (TypeError, ValueError):
        pass
    low = msg.lower()
    raw = meta.get("raw")
    raw = (raw if isinstance(raw, str) else json.dumps(raw) if raw else "").strip()
    if any(s in low for s in ("per-day", "per day", "daily", "free-models-per-day")):
        kind = "daily"
    elif any(s in low for s in ("per-min", "per minute", "free-models-per-min")):
        kind = "minute"
    elif meta.get("provider_name") or raw or "upstream" in low or "provider" in low:
        kind = "provider"
    elif reset_at and reset_at - time.time() > 600:
        kind = "daily"
    else:
        kind = "unknown"
    if raw and raw not in msg:
        msg = f"{msg} ({meta.get('provider_name') or 'upstream'}: {raw[:300]})"
    return kind, reset_at, msg


def retry_after(e):
    """Seconds the server asked us to wait (Retry-After), or None."""
    try:
        return max(0.0, float(_error_parts(e)[2].get("retry-after", "")))
    except (TypeError, ValueError):
        return None


def _ssl_context():
    """urllib on python.org macOS builds has no CA bundle; use the system trust store
    the same way the openai client does."""
    try:
        import ssl
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return None


def openrouter_get(path, api_key, timeout=10):
    req = urllib.request.Request(OPENROUTER + path, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
        return json.loads(r.read())


def fetch_key_info(api_key):
    """Account info from OpenRouter: is_free_tier, usage, limit_remaining, ..."""
    for path in ("/key", "/auth/key"):
        try:
            return openrouter_get(path, api_key).get("data", {})
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return None


# ─── agent ───────────────────────────────────────────────────────────────────
def _bracket_braces(args):
    """Close unbalanced braces on a streamed tool-arguments fragment.

    Anthropic servers stream tool arguments as partial JSON which is usually complete
    by content_block_stop, but some terminate on an open brace. Try to close it, then
    fall back to an escaped null rather than raising on the call dict.
    """
    s = args or ""
    for _ in range(4):
        try:
            json.loads(s)
            return s
        except ValueError:
            s += "}"
    return "null"


class _AnthropicUsage:
    """Adapter from Anthropic's message_delta usage to the shape `_charge` reads.

    The OpenAI stream hands back a pydantic usage object with `.cost` (an OpenRouter
    extension field) and `.prompt_tokens`. Anthropic's usage has input_tokens /
    output_tokens and no cost. `.cost` reads 0 so a local or Anthropic-direct request
    never charges a ledger that only OpenRouter's number is trusted into; the token
    counts satisfy the context-size warning in turn(). model_extra is the raw dict so
    any `.cost` the endpoint does report still surfaces.
    """
    def __init__(self, usage):
        self.usage = usage or {}
        self.prompt_tokens = self.usage.get("input_tokens", 0)
        self.output_tokens = self.usage.get("output_tokens", 0)
        self.model_extra = self.usage

    @property
    def cost(self):
        c = self.usage.get("cost")
        return c if isinstance(c, (int, float)) else 0.0


def _resolve_provider(model, provider="openrouter"):
    """Transport facts for a model string, via the provider registry.

    A model may name a backend explicitly (`ollama:qwen2.5-coder:7b` or
    `anthropic:claude-sonnet-4.5`), or the caller may have picked a provider for the
    whole ask (the editor's picker). The registry maps a well-known model prefix
    (`anthropic/…`, `openai/…`) and leaves everything else on the default provider, so
    `-m anthropic/claude-sonnet-4.5` alone talks to Anthropic without configuration.

    Local providers need no key, and their requests must not consume the OpenRouter
    free-request counters or RPM window (they are OpenRouter-only accounting), so the
    result carries `local` for the throttle and wallet to check.
    """
    p = _providers
    if p is None:
        return {"model": model, "backend": "openrouter", "provider": provider or "openrouter",
                "base_url": OPENROUTER, "api_type": "openai", "api_key": "",
                "local": False, "default": True, "custom": False}
    try:
        r = p.resolve(model, default=provider or "openrouter")
        if not r.get("api_key") and not r.get("local"):
            r["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
        return r
    except Exception:
        return {"model": model, "backend": provider or "openrouter", "provider": provider or "openrouter",
                "base_url": OPENROUTER, "api_type": "openai", "api_key": "",
                "local": False, "default": True, "custom": False}


class Agent:
    # An agent spends nothing unless a parent handed it a wallet or a spend ledger, so these are
    # the defaults for every agent, however it was built. `Spend(path="")` is a null ledger: it
    # gates nothing and records nothing, which is what an agent built without __init__ should do.
    wallet = None
    spend_usd = 0.0
    spend = Spend(path="", cap="")
    last_cost = 0.0
    total_cost = 0.0
    # An agent assembled without __init__ (tests, checkpoints) edits nothing, offers every tool
    # and is not a role whose product is a diff.
    edits = 0
    only_tools = None
    final_edit = False

    def __init__(self, model, yolo=False, headless=False, read_only=False, provider=None):
        self.provider = provider or os.environ.get("FLINT_PROVIDER", "openrouter")
        resolved = _resolve_provider(model, self.provider)
        self.model = resolved["model"]
        self.backend = resolved["backend"]
        self.provider = resolved["provider"]
        self.base_url = resolved["base_url"]
        self.api_type = resolved["api_type"]
        self.local = bool(resolved["local"])
        key = resolved.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not key and not self.local:
            print("No API key set. Add OPENROUTER_API_KEY to .env beside flint.py "
                  "or export it in your shell (local providers need no key).", file=sys.stderr)
            sys.exit(1)
        # Undo log identity: one per process, so a whole turn is one restorable unit.
        import uuid as _uuid
        self.undo_session = os.environ.get("FLINT_UNDO_SESSION") or _uuid.uuid4().hex[:12]
        self.api_key = "" if self.local else (key or "")
        # The OpenAI client refuses an empty api_key; local servers (Ollama, LM Studio,
        # llama.cpp, custom endpoints) accept and ignore any value, so a placeholder
        # satisfies the constructor without ever being sent as a credential the server
        # would trust.
        client_key = self.api_key or "local-unused"
        self.client = OpenAI(base_url=self.base_url, api_key=client_key, max_retries=0,
                             timeout=float(os.environ.get("FLINT_TIMEOUT", "90")),
                             default_headers={"X-Title": "flint"})
        self.throttle = Throttle()
        self.yolo = yolo
        self.headless = headless
        self.read_only = read_only or (headless and not yolo)
        # Roles whose whole product is a diff (a swarm's implementer and repair turns) say so,
        # and get the edit-only round below rather than losing the turn to the step limit.
        self.final_edit = bool(os.environ.get("FLINT_FINAL_EDIT")) and not self.read_only
        self.corpus = _corpus_ready()
        if headless:
            os.environ["FLINT_HEADLESS"] = "1"
        self.always = set()
        self.last_reasoning = ""
        self.last_prompt_tokens = 0
        self.wallet = wallet_from_env()
        self.spend_usd = 0.0
        self.spend = Spend()
        self.last_cost = 0.0
        self.total_cost = 0.0
        self.reset()

    def reset(self):
        self.edits = 0                  # files created, overwritten or edited this turn
        self.only_tools = None          # when set, the only tools offered for one request
        self.messages = [{"role": "system", "content": build_system_prompt()}]
        if self.read_only:
            self.messages[0]["content"] += "\nRead-only mode: inspect and advise; you cannot edit files or run shell commands."

    # ── model call ──
    @property
    def paid(self):
        """Free model ids end in `:free`; everything else is charged to the account.

        Local providers never charge and therefore are never paid, whatever the model
        name looks like (an Ollama model "qwen2.5-coder:7b" would otherwise read as
        paid and get gated on a spend ledger that does not exist for local turns)."""
        if getattr(self, "local", False):
            return False
        return not str(self.model).endswith(":free")

    def _request_kwargs(self):
        schemas = [t for t in TOOL_SCHEMAS
                   if (not self.read_only or t["function"]["name"] not in NEEDS_APPROVAL)
                   and (t["function"]["name"] != "study" or getattr(self, "corpus", False))]
        if getattr(self, "only_tools", None):
            schemas = [t for t in schemas if t["function"]["name"] in self.only_tools]
        kw = dict(model=self.model, messages=self.messages, tools=schemas,
                  stream=True, stream_options={"include_usage": True})
        if getattr(self, "final_answer", False):
            kw["tool_choice"] = "none"
        if not getattr(self, "local", False):
            if FALLBACKS:
                kw["extra_body"] = {"models": [self.model] + [m for m in FALLBACKS if m != self.model]}
            if self.wallet is not None or self.spend.path:
                # Bill from the provider's own number rather than a local price table. Both
                # ledgers need it: the editor's wallet and a swarm's daily spend budget.
                kw.setdefault("extra_body", {})["usage"] = {"include": True}
        return kw

    def _charge(self, usage):
        """Bank what OpenRouter says the reply cost, once per request.

        `cost` is the amount charged to the account and rides along in the final usage
        message of the stream. The OpenAI client does not model it, so it arrives as an
        extra field. A free model reports 0 and touches neither ledger.

        There are two ledgers and a reply is banked to whichever are active: the wallet
        is the editor's fixed pot of credits, handed down through FLINT_WALLET, and the
        spend ledger is one swarm's budget for a single day, named by FLINT_SPEND_FILE.
        A turn normally has one or the other, never both."""
        if usage is None:
            return
        cost = getattr(usage, "cost", None)
        if cost is None:
            extra = getattr(usage, "model_extra", None)
            if isinstance(extra, dict):
                cost = extra.get("cost")
        try:
            usd = round(max(0.0, float(cost)), 6)
        except (TypeError, ValueError):
            return
        self.last_cost = usd
        self.total_cost = round(self.total_cost + usd, 6)
        if usd:
            self.spend.add(usd, self.model)
            self._record_charge(usd)
        # The lasting receipt ledger (~/.albatross/routes.jsonl by default) gets
        # one row per request — local rows at $0.00 so the OpenRouter spend number
        # is never diluted by free or local turns (change request §6).
        if not getattr(self, "local", False):
            try:
                from swarm.receipts import record
                record({"model": self.model, "backend": getattr(self, "backend", "openrouter"),
                        "usd": usd, "local": False})
            except Exception:
                pass  # a receipt is banked, never fatal
        if self.wallet is not None:
            self.spend_usd = round(self.spend_usd + usd, 6)
            snap = self.wallet.record(usd, self.model)
            if usd:
                self._note(f"[dim]spend: ${usd:.4f} this request, "
                           f"${snap['spent']:.4f} of ${snap['cap']:.2f} allowance[/]")

    def _record_charge(self, usd):
        """Append this request's charge to FLINT_CHARGE_FILE, for the caller to bank.

        391 paid_fallback rows were recorded and not one carried a confirmed charge, so cost per
        landed commit and cost per model could not be worked out at all. The number here is
        OpenRouter's own, not a local price table."""
        path = os.environ.get("FLINT_CHARGE_FILE")
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps({"at": time.time(), "model": self.model, "usd": usd}) + "\n")
        except OSError as e:                # a charge is banked, never fatal to the turn
            print(f"could not record the charge: {e}", file=sys.stderr, flush=True)

    def _note(self, msg):
        (console.print if not self.headless else
         (lambda m: print(re.sub(r"\[/?[^\]]*\]", "", m), file=sys.stderr)))(msg)

    def _note_request(self, attempt=1, retry_of=None, error_class=None):
        """Record one HTTP request attempt to the sidecar, before it is made.

        The daily quota counts attempts, not successes, so a retry that will be eaten by
        a provider error still needs its row. Writing before the call means a SIGKILL
        between the write and the response still leaves the count."""
        if not REQUEST_FILE:
            return
        try:
            with open(REQUEST_FILE, "a") as f:
                f.write(json.dumps({"t": time.time(), "model": self.model,
                                    "provider": getattr(self, "backend", "openrouter"),
                                    "attempt": int(attempt),
                                    "retry_of": retry_of, "error_class": error_class,
                                    "paid": bool(self.paid), "local": bool(getattr(self, "local", False))}) + "\n")
        except OSError:                # a count line must never kill a turn
            pass

    def complete(self):
        # Failed attempts can still count toward the daily quota, so retries are few and deliberate.
        provider_retries = 0
        transport_attempt = 0
        for attempt in range(8):
            if self.wallet:
                allowed, why = self.wallet.check()
                if not allowed:
                    raise WalletEmpty(why)
            if self.paid:
                self.spend.check()
            self.throttle.acquire(paid=self.paid, local=getattr(self, "local", False),
                                  on_wait=lambda s: self._note(
                f"[dim]pacing: {RPM_LIMIT}/min budget used, waiting {s:.0f}s[/]"))
            transport_attempt += 1
            self._note_request(attempt=transport_attempt)
            try:
                return self._stream_once()
            except IncompleteResponse as e:
                # finish_reason "error" means the upstream provider failed mid-reply; the model
                # did not answer badly. Retry briefly, then report the provider as unavailable
                # (exit 7), which rests the model instead of scoring it.
                if e.finish_reason != "error":
                    raise
                provider_retries += 1
                if provider_retries > 2:
                    raise ProviderUnavailable(f"{self.model}: provider kept failing mid-reply: {e}") from e
                wait = 5 * 2 ** (provider_retries - 1)
                self._note(f"[yellow]provider failed mid-reply — retry {provider_retries}/2 in {wait}s[/]")
                time.sleep(wait)
                continue
            except APIError as e:
                body = e.body if isinstance(e.body, dict) else {}
                body = body.get("error", body)
                code = getattr(e, "status_code", None) or (body.get("code") if isinstance(body, dict) else None)
                if isinstance(e, APIConnectionError) or str(code) in ("500", "502", "503", "504", "529"):
                    if attempt >= 3:
                        raise ProviderUnavailable(f"{self.model}: repeated connection/provider errors: {e}") from e
                    wait = 3 * 2 ** attempt
                    self._note(f"[yellow]connection/provider error — retry {attempt + 1}/3 in {wait}s[/]")
                    time.sleep(wait)
                    continue
                if str(code) == "402":
                    raise OutOfCredits("OpenRouter returned 402: insufficient credits or account limit reached.") from e
                # 403/404 are permanent for this key: the model is gated (e.g. "only available on
                # agentic harnesses"), renamed or gone. Retrying only spends the daily allowance.
                if str(code) in ("403", "404"):
                    detail = str(_error_parts(e)[0].get("message") or e)
                    raise ModelUnavailable(f"{self.model} is not available to this API key "
                                           f"({code}): {detail[:300]}") from e
                if not isinstance(e, RateLimitError) and str(code) != "429":
                    raise
                kind, reset_at, msg = classify_429(e)
                if kind == "daily":
                    reset_at = reset_at or _next_utc_midnight()
                    self.throttle.block(reset_at)
                    raise DailyCapReached(reset_at)
                if kind == "minute":
                    wait = max(5, min(65, (reset_at or time.time() + 30) - time.time()))
                    self._note(f"[yellow]per-minute cap hit — waiting {wait:.0f}s[/]")
                    time.sleep(wait)
                    continue
                # Rate-limited upstream: the model works, its free capacity is taken. Each retry
                # costs a request, so a swarm is better off switching models (exit 8) than waiting.
                provider_retries += 1
                if provider_retries > 3:
                    raise ProviderBusy(f"{self.model} is rate-limited upstream: {msg}. Try again "
                                       "shortly or use another model (FLINT_FALLBACKS).")
                asked = retry_after(e)
                wait = round(min(60, max(5, asked))) if asked is not None else 10 * 2 ** (provider_retries - 1)
                self._note(f"[yellow]provider busy ({kind}) — retry {provider_retries}/3 in {wait}s: {msg[:400]}[/]")
                time.sleep(wait)
        raise ProviderUnavailable("Gave up after repeated errors.")

    def _stream_once(self):
        """One request to whatever backend this agent is pointed at.

        OpenAI-compatible endpoints (OpenRouter, Ollama, LM Studio, MLX, llama.cpp,
        custom URLs) go through the OpenAI client's streaming chat completions, whose
        chunk layout the loop below knows well. The Anthropic Messages API is a
        different wire format, so it gets its own streamer that produces the same
        (content, reasoning, calls, finish_reason, billed) shape before the shared
        tail — which then does not care which transport produced the reply.
        """
        if getattr(self, "api_type", "openai") == "anthropic":
            return self._stream_anthropic()
        stream = self.client.chat.completions.create(**self._request_kwargs())
        content, reasoning, calls = "", "", {}
        finish_reason, billed = None, None
        live = None
        status = None if self.headless else console.status("[dim]thinking…[/]", spinner="dots")
        if status:
            status.start()
        try:
            for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    billed = usage
                if usage and getattr(usage, "prompt_tokens", None):
                    self.last_prompt_tokens = usage.prompt_tokens
                if not chunk.choices:
                    continue
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
                delta = chunk.choices[0].delta
                r = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
                if r:
                    reasoning += r
                    if status:
                        status.update(f"[dim]thinking… ({len(reasoning) // 4} tokens)[/]")
                if delta.content:
                    content += delta.content
                    if not self.headless:
                        if live is None:
                            if status:
                                status.stop()
                            live = Live(Markdown(content), console=console,
                                        refresh_per_second=12, vertical_overflow="visible")
                            live.start()
                        live.update(Markdown(content))
                for tc in delta.tool_calls or []:
                    idx = tc.index if tc.index is not None else (
                        max(calls) if calls and not tc.id else len(calls))
                    slot = calls.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = tc.function
                    if fn and fn.name and not slot["name"]:
                        slot["name"] = fn.name
                    if fn and fn.arguments:
                        slot["args"] += fn.arguments
        finally:
            stream.close()
            self._charge(billed)      # a reply that failed halfway through still cost money
            if status:
                status.stop()
            if live:
                live.stop()
        return self._finish_stream(content, reasoning, calls, finish_reason)

    # ── Anthropic Messages API ────────────────────────────────────────────────
    def _stream_anthropic(self):
        """Stream one request over /v1/messages (Anthropic native, or a local server
        that only speaks it). Yields the same tuple as the OpenAI path.

        SSE wire events: `message_start`, `content_block_start`, `content_block_delta`
        with `text_delta`/`input_json_delta`/`signature_delta`, `content_block_stop`,
        `message_delta` (carries stop_reason and per-message usage), `message_stop`.
        Tool use arrives as a block with `{}`-formed JSON arguments streamed in
        input_json_delta pieces; arguments are closed with literal `{`/`}` when a
        server forgets them.
        """
        base = (self.base_url or "").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        url = base + "/v1/messages"
        headers = {"content-type": "application/json", "accept": "text/event-stream"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        body = self._anthropic_body()
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST")
        content, reasoning, calls = "", "", {}
        finish_reason, billed = None, None
        pending = []          # tool-use blocks being streamed in, by index
        tool_order = []       # indices in arrival order, for stable ids at the end
        live = None
        status = None if self.headless else console.status("[dim]thinking…[/]", spinner="dots")
        if status:
            status.start()
        try:
            with urllib.request.urlopen(req, timeout=float(os.environ.get("FLINT_TIMEOUT", "90")),
                                        context=_ssl_context()) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        ev = json.loads(data)
                    except ValueError:
                        continue
                    etype = ev.get("type")
                    if etype == "message_start":
                        msg = ev.get("message") or {}
                        usage = msg.get("usage") or {}
                        tokens = usage.get("input_tokens")
                        if tokens:
                            self.last_prompt_tokens = tokens
                    elif etype == "content_block_start":
                        idx = ev.get("index", len(pending))
                        cb = ev.get("content_block") or {}
                        if cb.get("type") == "tool_use":
                            pending.append({"id": cb.get("id", ""), "name": cb.get("name", ""),
                                            "args": "", "json_started": False})
                            tool_order.append(idx)
                        elif cb.get("type") == "thinking" and status:
                            status.update("[dim]thinking…[/]")
                    elif etype == "content_block_delta":
                        d = ev.get("delta") or {}
                        if d.get("type") == "text_delta":
                            content += d.get("text", "")
                            if not self.headless:
                                if live is None:
                                    if status:
                                        status.stop()
                                    live = Live(Markdown(content), console=console,
                                                refresh_per_second=12, vertical_overflow="visible")
                                    live.start()
                                live.update(Markdown(content))
                        elif d.get("type") == "thinking_delta":
                            reasoning += d.get("thinking", "")
                            if status:
                                status.update(f"[dim]thinking… ({len(reasoning) // 4} tokens)[/]")
                        elif d.get("type") == "input_json_delta":
                            idx = ev.get("index")
                            slot = pending[idx] if idx is not None and idx < len(pending) else None
                            if slot is not None:
                                piece = d.get("partial_json", "")
                                if piece and not slot["json_started"] and not piece.lstrip().startswith("{"):
                                    slot["args"] += "{"
                                    slot["json_started"] = True
                                slot["args"] += piece
                    elif etype == "content_block_stop":
                        idx = ev.get("index")
                        if idx is not None and idx < len(pending):
                            slot = pending[idx]
                            if not slot["json_started"] and slot["args"].strip() == "":
                                slot["args"] = "{}"
                            elif slot["args"] and not slot["args"].rstrip().endswith("}"):
                                slot["args"] += "}"
                    elif etype == "message_delta":
                        d = ev.get("delta") or {}
                        sr = d.get("stop_reason")
                        if sr:
                            finish_reason = {"end_turn": "stop", "tool_use": "tool_calls",
                                             "max_tokens": "length"}.get(sr, sr)
                        usage = ev.get("usage") or {}
                        if usage.get("output_tokens"):
                            billed = _AnthropicUsage(usage)
                    elif etype == "error" or etype == "stream_error":
                        err = ev.get("error") or {}
                        raise APIError(
                            f"{err.get('type', 'error')}: {err.get('message', '')}",
                            body={"error": {"message": err.get("message", ""),
                                            "code": 500 if etype == "stream_error" else None}},
                            response=resp, request=req)
        finally:
            if status:
                status.stop()
            if live:
                live.stop()
        # Tool blocks arrive in event order but carry server indices; rebuild the calls
        # dict in arrival order so ids stay stable for the tool loop in turn().
        for i, idx in enumerate(tool_order):
            slot = pending[idx] if idx < len(pending) else None
            if slot is None:
                continue
            args = slot["args"] or "{}"
            try:
                json.loads(args)
            except ValueError:
                args = _bracket_braces(args)
            calls[i] = {"id": slot["id"] or f"call_{len(self.messages)}_{i}",
                        "name": slot["name"], "args": args}
        self._charge(billed)
        return self._finish_stream(content, reasoning, calls, finish_reason)

    def _anthropic_body(self):
        """The /v1/messages request body built from the same messages list and tool
        schemas the OpenAI path uses, so an agent's history is transport-agnostic."""
        system = "\n\n".join(m["content"] for m in self.messages if m["role"] == "system")
        msgs = [m for m in self.messages if m["role"] != "system"]
        out = []
        for m in msgs:
            if m["role"] == "user":
                out.append({"role": "user", "content": str(m.get("content") or "")})
            elif m["role"] == "assistant":
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": str(m["content"])})
                for c in m.get("tool_calls") or []:
                    fn = c.get("function") or {}
                    blocks.append({"type": "tool_use", "id": c.get("id") or "",
                                   "name": fn.get("name", ""),
                                   "input": json.loads(fn.get("arguments") or "{}")})
                if not blocks:                     # a bare tool-only assistant turn
                    continue
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                # Anthropic pairs tool results with the assistant's tool_use block,
                # matched by tool_call_id.
                out.append({"role": "user", "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "",
                    "content": str(m.get("content") or "")}]})
        body = {"model": self.model, "max_tokens": 64_000, "stream": True,
                "messages": out, "tools": self._anthropic_tools()}
        if system:
            body["system"] = system
        if getattr(self, "final_answer", False):
            body["tool_choice"] = {"type": "none"}
        return body

    def _anthropic_tools(self):
        """Tool schemas in Anthropic's shape: a `description` is required and
        `additionalProperties` must be absent (Anthropic rejects it)."""
        tools = []
        for t in self._request_kwargs()["tools"]:
            f = t["function"]
            params = dict(f.get("parameters") or {})
            params.pop("additionalProperties", None)
            tools.append({"name": f["name"],
                          "description": f.get("description", "")[:3000],
                          "input_schema": params})
        return tools

    def _finish_stream(self, content, reasoning, calls, finish_reason):
        """The tail shared by every transport: recover text-form tool calls, validate
        the finish, and return the (content, calls) the turn loop expects."""
        self.last_reasoning = reasoning
        if not calls and content:
            recovered, rest = recover_text_calls(content, self._request_kwargs()["tools"])
            if recovered:
                if self.headless:
                    print(f"recovered {len(recovered)} tool call(s) written as reply text",
                          file=sys.stderr, flush=True)
                content, calls = rest, dict(enumerate(recovered))
        if finish_reason not in ("stop", "tool_calls"):
            raise IncompleteResponse(f"Model response incomplete (finish_reason={finish_reason!r}); no tools executed.",
                                     finish_reason)
        if not content.strip() and not calls:
            raise IncompleteResponse("Model returned an empty response.")
        return content, [calls[k] for k in sorted(calls)]

    # ── tool execution ──
    def _preview(self, name, args):
        if name == "bash":
            return Syntax(args.get("command", ""), "bash", word_wrap=True)
        if name == "write_file":
            lexer = Syntax.guess_lexer(args.get("path", ""), args.get("content", ""))
            body = "\n".join(args.get("content", "").splitlines()[:40])
            return Syntax(body, lexer, line_numbers=True, word_wrap=True)
        if name == "edit_file":
            old, new, err = _plan_edit(args.get("path", ""), args.get("old_str", ""),
                                       args.get("new_str", ""), args.get("replace_all", False))
            if err:
                return None
            diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                "before", "after", n=2))
            return Syntax(diff or "(no change)", "diff", word_wrap=True)
        return None

    def _approve(self, name, args):
        if self.yolo or name in self.always:
            return True, None
        if self.headless:
            return False, "Denied: running non-interactively without --yolo."
        preview = self._preview(name, args)
        if preview is None and name == "edit_file":
            return True, None  # edit will fail with a helpful error; no need to ask
        console.print(Panel(preview, border_style=ACCENT, title=name, title_align="left"))
        ans = console.input(f"[bold]allow?[/] [green]y[/] yes · [red]n[/] no · "
                            f"[{ACCENT}]a[/] always ({name}) · or type instructions › ").strip()
        low = ans.lower()
        if low in ("y", "yes"):
            return True, None
        if low in ("a", "always"):
            self.always.add(name)
            return True, None
        if low in ("", "n", "no"):
            return False, "The user denied this tool call."
        return False, f"The user denied this tool call and said: {ans}"

    @staticmethod
    def _summary(name, args):
        key = {"bash": "command", "search": "pattern", "list_files": "pattern"}.get(name, "path")
        s = str(args.get(key, "")).replace("\n", " ")
        return s if len(s) < 80 else s[:77] + "…"

    def _snapshot_for_undo(self, name, args):
        """Record the before-state of every file a mutating tool is about to touch.

        A write_file or edit_file changes one known path; a bash command can change
        anything, so the known paths are snapshotted defensively (the model names the
        files it means in the command). Snapshotting only when the file exists keeps
        newly-created files out of the ledger — nothing to restore for them.
        """
        if getattr(self, "read_only", False) or not getattr(self, "undo_session", None):
            return
        try:
            from swarm.undo import snapshot
            paths = []
            if name in ("write_file", "edit_file") and args.get("path"):
                paths = [args["path"]]
            elif name == "bash" and args.get("command"):
                # Likely file names in a shell command: `pytest tests/test_x.py`,
                # `git add src/foo.py`, `head -5 pyproject.toml`. Best effort only.
                import re as _re
                for tok in _re.findall(r"[\w./-]+\.(?:py|js|ts|go|rs|c|h|json|toml|md|txt|sh|css|html)[\w./-]*",
                                       args["command"]):
                    if tok.startswith(("./", "../")) or "/" in tok:
                        paths.append(tok)
            for p in dict.fromkeys(paths):
                snapshot(self.undo_session, p, command=name)
        except Exception:
            pass  # undo is opportunistic; a turn must never fail because of it

    def _run_hooks(self, name, args, out):
        """Run PostToolUse hooks after a tool call (agent.config.json `hooks:` block).

        The hook gets the tool name, its arguments and a truncated result on stdin.
        A hook that fails or times out is swallowed — the tool result is already
        produced and must not be lost to a slow external command.
        """
        try:
            from swarm.hooks import run_hooks
            payload = {"tool": name, "arguments": args,
                       "result": str(out)[:2000], "model": getattr(self, "model", "")}
            run_hooks(["PostToolUse"], os.getcwd(), payload)
        except Exception:
            pass  # a hook is a side effect, never a failure of the turn

    def run_tool(self, call):
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return f"Error: tool arguments were not valid JSON ({e}). Retry with a valid JSON object."
        fn = TOOLS.get(name)
        if not fn:
            return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS)}."
        if self.read_only and name in NEEDS_APPROVAL:
            return "Denied: read-only mode does not allow edits or shell commands."

        if not self.headless:
            console.print(Text.assemble(("◆ ", ACCENT), (name, "bold"),
                                        (f"({self._summary(name, args)})", "dim")))
        else:
            print(f"tool: {name}", file=sys.stderr, flush=True)
        if name in NEEDS_APPROVAL:
            ok, reason = self._approve(name, args)
            if not ok:
                if not self.headless:
                    console.print("  [red]└─ denied[/]")
                return reason
        self._snapshot_for_undo(name, args)
        try:
            out = fn(**args)
        except TypeError as e:
            out = f"Error: bad arguments for {name}: {e}"
        except Exception as e:  # tools should never crash the agent
            out = f"Error: {type(e).__name__}: {e}"

        if name in ("write_file", "edit_file") and out.startswith(("Created", "Overwrote", "Edited")):
            self.edits += 1
        self._run_hooks(name, args, out)
        if not self.headless:
            lines = out.splitlines() or [""]
            shown = "\n     ".join(l[:160] for l in lines[:4])
            more = f"\n     … +{len(lines) - 4} lines" if len(lines) > 4 else ""
            style = "red" if out.startswith("Error") else "dim"
            console.print(Text(f"  └─ {shown}{more}", style=style))
        return _truncate(out)

    # ── turn loop ──
    def save_checkpoint(self):
        path = getattr(self, "checkpoint_path", None)
        if path:
            from nonstop import save_checkpoint
            save_checkpoint(path, self.messages)

    NO_EDIT_YET = ("Three rounds left and no file has changed. Stop reading. Make the edit now "
                   "with edit_file or write_file, then run the single test module that covers it.")

    EXPLORE_LIMIT_MSG = ("You have spent most of this turn's rounds reading and still have not "
                         "edited a file. Budget is a real constraint: stop exploring now and "
                         "implement the smallest verified slice with edit_file or write_file, "
                         "run the narrowest test that covers it, and give your final answer.")

    EDIT_ONLY_MSG = ("Most of this turn's budget is gone. Only edit_file and write_file are "
                     "available from here on; run the narrowest test that covers your change "
                     "and finish. No more reading or searching.")

    def _assistant_msg(self, content, calls):
        """The assistant message for a reply, with stable ids for the calls it made."""
        msg = {"role": "assistant", "content": content or ""}
        if calls:
            msg["tool_calls"] = [{
                "id": c["id"] or f"call_{len(self.messages)}_{i}", "type": "function",
                "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for i, c in enumerate(calls)]
        return msg

    def turn(self, user_text):
        self.messages.append({"role": "user", "content": user_text})
        self.edits, content = 0, ""
        # Work order §2 70% rule: a turn that has not edited by 70% of its round budget is
        # stopped early as exploration_exhausted. It only means something for turns long
        # enough to have a real exploration phase (implementer/repair budgets, 10+ rounds);
        # a 1-8 round turn that stops at 70% would leave almost nothing to show.
        explore_floor = max(1, int(MAX_STEPS * 0.7)) if MAX_STEPS >= 10 else None
        explore_nudged = False
        edit_only = False
        for step in range(MAX_STEPS):
            if self.headless:
                print(f"round {step + 1}/{MAX_STEPS}: requesting {self.model}", file=sys.stderr, flush=True)
            content, calls = self.complete()
            msg = self._assistant_msg(content, calls)
            self.messages.append(msg)
            self.save_checkpoint()
            if not calls:
                break
            left = MAX_STEPS - step - 1
            for i, c in enumerate(msg["tool_calls"]):
                result = self.run_tool(c)
                # Nothing else tells the model how much of the turn is left, so it reads for
                # eleven rounds of twelve and then loses the edit to the step limit.
                if self.headless and i == len(msg["tool_calls"]) - 1:
                    result += f"\n\n[round {step + 1} of {MAX_STEPS}; {left} left]"
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                self.save_checkpoint()
            if self.headless and not self.edits:
                if left == 3:
                    self.messages.append({"role": "user", "content": self.NO_EDIT_YET})
                elif explore_floor is not None:
                    # Work order §2 (long budgets only): at 60% of the role budget stop
                    # exploring and implement the smallest slice; at 80% drop read/search
                    # tools so only edit tools remain; a turn with no edit by 70% is stopped
                    # early as exploration_exhausted.
                    if not explore_nudged and left <= max(1, int(MAX_STEPS * 0.4)):
                        explore_nudged = True
                        self.messages.append({"role": "user", "content": self.EXPLORE_LIMIT_MSG})
                    if not edit_only and left <= max(1, int(MAX_STEPS * 0.2)):
                        edit_only = True
                        self.only_tools = set(self.EDIT_TOOLS)
                        self.messages.append({"role": "user", "content": self.EDIT_ONLY_MSG})
                    if step + 1 >= explore_floor:
                        # Keep whatever the model already wrote; the turn is over because
                        # reading exhausted it, not because it failed.
                        raise ExplorationExhausted(
                            f"Stopped after {step + 1} rounds with no edit; the turn spent 70% of its "
                            "budget exploring. Leave partial work to the normal gates.")
        else:
            content = self._final_answer()
        if self.last_prompt_tokens > CONTEXT_WARN_TOKENS:
            self._note("[yellow]context is getting large — consider /clear[/]")
        return content

    FINAL_NUDGE = ("You have used every tool round this turn allows. Do not call any tools. "
                   "Give your final answer now, in exactly the format the task asked for.")
    FINAL_EDIT_NUDGE = ("You have used every tool round this turn allows and no file has changed, "
                        "so nothing you did would be kept. This round offers two tools, edit_file "
                        "and write_file, and nothing else. Make the edit now. Do not read, search "
                        "or run anything.")
    EDIT_TOOLS = ("edit_file", "write_file")

    def _run_calls(self, content, calls):
        """Record a reply and run its calls, the way a normal round does."""
        msg = self._assistant_msg(content, calls)
        self.messages.append(msg)
        for c in msg.get("tool_calls", []):
            self.messages.append({"role": "tool", "tool_call_id": c["id"],
                                  "content": self.run_tool(c)})
        self.save_checkpoint()

    def _final_edit_round(self):
        """One round offering only the edit tools, for a role whose product is a diff.

        33 of 189 studio attempts ran out of rounds and 27 more read files and never edited: the
        model spends the turn understanding the code and the harness keeps none of it. This is
        the last chance to write the change down, and reading is what ran the clock out, so
        reading is not on offer."""
        print(f"step limit: offering {self.model} one edit-only round", file=sys.stderr, flush=True)
        self.messages.append({"role": "user", "content": self.FINAL_EDIT_NUDGE})
        self.only_tools = set(self.EDIT_TOOLS)
        try:
            content, calls = self.complete()
        except (DailyCapReached, OutOfCredits, BudgetPaused, SpendExhausted,
                ProviderUnavailable, WalletEmpty, KeyboardInterrupt):
            raise
        except Exception as e:
            print(f"edit-only round failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            self.messages.pop()          # leave history as the step limit found it
            return
        finally:
            self.only_tools = None
        self._run_calls(content, [c for c in calls if c["name"] in self.EDIT_TOOLS])

    def _final_answer(self):
        """After the step limit, a headless turn gets one tool-free round to answer, so the
        rounds already spent are not wasted (a reviewer that read everything but never wrote
        its verdict). Interactive turns stop as before."""
        limit = StepLimitReached(f"Stopped after {MAX_STEPS} rounds; task is incomplete. Changes may already exist.")
        if not self.headless:
            raise limit
        if self.final_edit and not self.edits:
            self._final_edit_round()
        print(f"step limit: asking {self.model} for a final answer without tools", file=sys.stderr, flush=True)
        self.messages.append({"role": "user", "content": self.FINAL_NUDGE})
        self.final_answer = True
        try:
            content, calls = self.complete()
        except (DailyCapReached, OutOfCredits, BudgetPaused, SpendExhausted,
                ProviderUnavailable, KeyboardInterrupt):
            raise
        except Exception as e:
            raise limit from e
        finally:
            self.final_answer = False
        if calls:
            # Tools were off for that request, so these are calls the model wrote into its reply
            # text and recover_text_calls picked up. Throwing an edit away here is how a finished
            # change became a "no change" attempt, a punished model and a split task.
            edits = [c for c in calls if c["name"] in self.EDIT_TOOLS]
            if not (edits and self.final_edit):
                raise limit          # still reaching for tools: no answer, and none run
            print(f"applying {len(edits)} edit(s) the final answer wrote as text",
                  file=sys.stderr, flush=True)
            self._run_calls(content, edits)
            return content.strip() or f"Applied {len(edits)} edit(s) written as reply text."
        if not content.strip():
            raise limit
        self.messages.append({"role": "assistant", "content": content})
        self.save_checkpoint()
        return content

    def repair_after_interrupt(self):
        """Make history valid again if Ctrl+C landed mid tool-round."""
        answered = {m.get("tool_call_id") for m in self.messages if m["role"] == "tool"}
        for m in reversed(self.messages):
            if m["role"] == "assistant" and m.get("tool_calls"):
                for c in m["tool_calls"]:
                    if c["id"] not in answered:
                        self.messages.append({"role": "tool", "tool_call_id": c["id"],
                                              "content": "Interrupted by the user."})
                break
        if self.messages[-1]["role"] == "user":
            self.messages.pop()


# ─── ui ──────────────────────────────────────────────────────────────────────
MASCOT = [
    " ▗▄▄▄▄▖ ",
    " ▐▘◆◆▝▌ ",
    " ▐ ▔▔ ▌ ",
    "  ▀▘▝▀  ",
]

HELP = """[bold]commands[/]
  /help            this list
  /clear           start a fresh conversation
  /model [slug]    show or switch the OpenRouter model
  /yolo            toggle auto-approve for edits and shell commands
  /nonstop [for Nh]: goal   unattended edits/tests (default 8 hours)
  /think           show the model's reasoning from the last reply
  /tokens          context size of the last request
  /usage           account tier, credits, and requests used today
  /study <query>   search the local study corpus (no API quota)
  /exit            quit (or Ctrl+D)
[dim]Ctrl+C interrupts the current turn. Put project notes in FLINT.md or AGENTS.md.[/]"""


def banner(agent):
    cwd = os.getcwd().replace(str(Path.home()), "~")
    side = ["[bold]flint[/]", f"[dim]{agent.model}[/]", f"[dim]{cwd}[/]",
            "[dim]/help for commands[/]"]
    rows = [f"[{ACCENT}]{m}[/]  {s}" for m, s in zip(MASCOT, side)]
    console.print(Panel("\n".join(rows), border_style=ACCENT, expand=False))


def show_usage(agent):
    info = fetch_key_info(agent.api_key)
    today = agent.throttle.today()
    if info is None:
        console.print(f"[dim]couldn't reach OpenRouter for account info.[/] requests today (local count): {today}")
        return
    quota = info.get("free_model_daily_requests") or {}
    tier = ("[yellow]free tier — no credits purchased yet[/]" if info.get("is_free_tier")
            else "[green]credits purchased[/]")
    remaining = info.get("limit_remaining")
    console.print(f"[dim]account:[/] {tier}\n"
                  f"[dim]requests today (local attempts):[/] {today}\n"
                  f"[dim]free-model requests (provider):[/] {quota.get('used', 'unknown')} / {quota.get('limit', 'unknown')}\n"
                  f"[dim]paid usage today:[/] ${info.get('usage_daily', 0):.4f}"
                  + (f"   [dim]key limit remaining:[/] ${remaining:.2f}" if remaining is not None else ""))


def handle_command(agent, line):
    cmd, _, arg = line.partition(" ")
    arg = arg.strip()
    if cmd in ("/exit", "/quit"):
        raise EOFError
    if cmd == "/help":
        console.print(HELP)
    elif cmd == "/clear":
        agent.reset()
        console.print("[dim]conversation cleared[/]")
    elif cmd == "/model":
        if arg:
            agent.model = arg
        console.print(f"[dim]model:[/] {agent.model}")
    elif cmd == "/yolo":
        agent.yolo = not agent.yolo
        console.print(f"[dim]auto-approve:[/] {'[red]on[/]' if agent.yolo else 'off'}")
    elif cmd == "/think":
        console.print(Text(agent.last_reasoning or "(no reasoning captured)", style="dim italic"))
    elif cmd == "/usage":
        show_usage(agent)
    elif cmd == "/study":
        console.print(Text(study(arg) if arg else "usage: /study <query>", style="dim"))
    elif cmd == "/tokens":
        console.print(f"[dim]last prompt:[/] {agent.last_prompt_tokens:,} tokens")
    else:
        console.print(f"[red]unknown command {cmd}[/] — /help")


def repl(agent):
    banner(agent)
    while True:
        try:
            console.print()
            line = console.input(f"[bold {ACCENT}]›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/]")
            return
        if not line:
            continue
        from nonstop import parse_trigger, resolve_goal
        trigger = parse_trigger(line)
        if trigger is not None:
            if agent.read_only:
                console.print("[red]Nonstop mode needs edits and shell access; restart without --read-only.[/]")
                continue
            hours, goal = trigger
            previous = next((m["content"] for m in reversed(agent.messages) if m["role"] == "user"), "")
            try:
                goal = resolve_goal(goal, os.getcwd(), previous)
                run_unattended(goal, agent.model, hours if hours is not None else 8)
                agent.reset()  # Files may have changed substantially during the run.
            except ValueError as e:
                console.print(f"[red]{e}[/]")
            continue
        if line.startswith("/"):
            try:
                handle_command(agent, line)
            except EOFError:
                console.print("[dim]bye[/]")
                return
            continue
        try:
            agent.turn(line)
            console.print(f"[dim]· {agent.throttle.today()} requests today[/]")
        except KeyboardInterrupt:
            agent.repair_after_interrupt()
            console.print("\n[yellow]interrupted[/]")
        except DailyCapReached as e:
            agent.repair_after_interrupt()
            console.print(f"[red]daily free-model cap reached.[/] Resets {fmt_reset(e.reset_at)}. "
                          "Switch to a paid model with /model to keep going.")
        except OutOfCredits as e:
            agent.repair_after_interrupt()
            console.print(f"[red]{e}[/]")
        except Exception as e:
            agent.repair_after_interrupt()
            console.print(f"[red]error:[/] {e}")


def run_unattended(goal, model, hours=8, test_command=None):
    from nonstop import run_nonstop
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is not set; add it to .env beside flint.py.")
    return run_nonstop(goal, os.getcwd(), model, STATE_DIR, hours, test_command)


def main():
    ap = argparse.ArgumentParser(description="flint — a tiny terminal coding agent on OpenRouter")
    ap.add_argument("-p", "--prompt", help="run one task non-interactively and print the result")
    ap.add_argument("--prompt-file", metavar="PATH",
                    help="read that task from a file instead. A swarm prompt is 10-25 KB, and in "
                         "argv it lands in `ps` and in every exception's message.")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"model slug (default {DEFAULT_MODEL}); a `backend:` prefix or a known vendor slug (`anthropic/claude-…`) picks the provider")
    ap.add_argument("--provider", default=None,
                    help="provider backend (openrouter, ollama, lm-studio, mlx, llamacpp, openai, anthropic); env FLINT_PROVIDER, default openrouter")
    ap.add_argument("--yolo", action="store_true", help="auto-approve edits and shell commands")
    ap.add_argument("--read-only", action="store_true", help="allow only file reading, listing and searching")
    ap.add_argument("--nonstop", action="store_true", help="unattended work cycles; enables edits and shell commands")
    ap.add_argument("--hours", type=float, help="nonstop duration (default 8 hours)")
    ap.add_argument("--test-command", help="shell verification command to run after each nonstop work cycle")
    ap.add_argument("--checkpoint", help=argparse.SUPPRESS)
    ap.add_argument("-C", "--cwd", help="project directory to work in")
    ap.add_argument("--continue", dest="resume", action="store_true",
                    help="resume the most recent nonstop run in ~/.flint/nonstop: same goal, same model, another N hours")
    ap.add_argument("--reset-cap", action="store_true",
                    help="clear a remembered daily-cap block (e.g. after buying credits)")
    a = ap.parse_args()
    if a.cwd:
        os.chdir(Path(a.cwd).expanduser())
    if a.reset_cap:
        Throttle().clear_block()
    if a.resume:
        # `flint --continue`: pick up the most recent nonstop session's goal and model
        # and run it for `--hours` more (default 4). The handoff artifacts are designed
        # for exactly this — the next cycle reads the previous one's evidence.
        ns_root = STATE_DIR / "nonstop"
        sessions = sorted(ns_root.glob("*/status.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True) if ns_root.is_dir() else []
        if not sessions:
            ap.error("--continue: no previous nonstop run in " + str(ns_root))
        state = json.loads(sessions[0].read_text())
        goal = state.get("goal") or ""
        model = state.get("model") or a.model
        hours = a.hours or state.get("hours") or 8
        if not goal:
            ap.error("--continue: the previous run has no goal recorded")
        print(f"flint: resuming the run from {sessions[0].parent.name}", file=sys.stderr)
        sys.exit(run_unattended(goal, model, hours, None))

    if a.prompt_file:
        if a.prompt is not None:
            ap.error("give a prompt with --prompt or --prompt-file, not both")
        try:
            a.prompt = Path(a.prompt_file).expanduser().read_text()
        except OSError as e:
            ap.error(f"could not read --prompt-file: {e}")
    prompt = (a.prompt if a.prompt != "-" else sys.stdin.read()) if a.prompt is not None else ""
    from nonstop import parse_trigger
    trigger = parse_trigger(prompt)
    if a.nonstop or trigger is not None:
        if a.read_only:
            ap.error("nonstop mode enables edits and shell commands; it cannot use --read-only")
        phrase_hours, goal = trigger if trigger is not None else (None, prompt)
        hours = a.hours if a.hours is not None else (phrase_hours if phrase_hours is not None else 8)
        try:
            sys.exit(run_unattended(goal, a.model, hours, a.test_command))
        except ValueError as e:
            ap.error(str(e))
    if a.hours is not None or a.test_command:
        ap.error("--hours and --test-command require --nonstop or a 'run this nonstop' prompt")

    if a.prompt is not None:
        agent = Agent(a.model, yolo=a.yolo, headless=True, read_only=a.read_only, provider=a.provider)
        agent.checkpoint_path = a.checkpoint
        if agent.read_only:
            print("flint: read-only mode; use --yolo to allow edits and shell commands.", file=sys.stderr)
        try:
            print(agent.turn(prompt))
        except KeyboardInterrupt:
            print("\nflint: interrupted; changes already made remain on disk.", file=sys.stderr)
            sys.exit(130)
        except StepLimitReached as e:
            print(f"flint: {e}", file=sys.stderr)
            sys.exit(5)
        except ExplorationExhausted as e:
            print(f"flint: {e}", file=sys.stderr)
            sys.exit(11)
        except BudgetPaused as e:
            print(f"flint: budget paused: {e}", file=sys.stderr)
            sys.exit(6)
        except WalletEmpty as e:
            print(f"flint: paid allowance exhausted: {e}", file=sys.stderr)
            sys.exit(10)
        except SpendExhausted as e:
            # Same exit code as a budget pause: the supervisor defers the task and tries
            # again after the ledger rolls over at midnight UTC.
            print(f"flint: {e}", file=sys.stderr)
            sys.exit(6)
        except DailyCapReached as e:
            print(f"flint: daily free-model cap reached, resets {fmt_reset(e.reset_at)}", file=sys.stderr)
            sys.exit(3)  # distinct code so a swarm orchestrator can back off
        except OutOfCredits as e:
            print(f"flint: {e}", file=sys.stderr)
            sys.exit(4)
        except ModelUnavailable as e:
            print(f"flint: model unavailable: {e}", file=sys.stderr)
            sys.exit(9)  # permanent for this key: the swarm drops the model rather than waiting
        except ProviderBusy as e:
            print(f"flint: model busy: {e}", file=sys.stderr)
            sys.exit(8)  # rate-limited upstream: the swarm hands the role to another model
        except ProviderUnavailable as e:
            print(f"flint: provider unavailable: {e}", file=sys.stderr)
            sys.exit(7)  # the model is down, not wrong: the swarm cools it down
        except Exception as e:
            print(f"flint: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
        finally:
            agent.save_checkpoint()
            if agent.wallet:
                print(f"cost: ${agent.spend_usd:.4f} this turn, "
                      f"${agent.wallet.spent():.4f} of ${agent.wallet.cap:.2f} allowance",
                      file=sys.stderr)
    else:
        repl(Agent(a.model, yolo=a.yolo, read_only=a.read_only, provider=a.provider))


if __name__ == "__main__":
    main()
