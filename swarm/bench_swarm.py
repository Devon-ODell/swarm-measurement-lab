#!/usr/bin/env python3
"""Benchmark the OpenRouterSwarm's harness across model tiers on identical coding tasks.

Each task runs through the SAME flint harness the swarm uses (flint.py --yolo in a
fresh worktree), so the only variable is the model. A task is scored by an
objective verifier (exact file contents / test exit code), matched to how the
swarm's own gate scores work.

Usage:
    python3 swarm/bench_swarm.py --models a,b,c [--tasks 1,2,3] [--dry]

    --models   comma-separated model slugs (openrouter: prefix forces OpenRouter)
    --tasks    comma-separated 1..N tasks
    --dry      print the task battery and exit without spending
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FLINT = [sys.executable, str(Path(__file__).resolve().parent.parent / "flint.py")]

# ---------------------------------------------------------------- task battery
# Each task: (name, prompt, [files to seed], verifier)
# verifier(cwd) -> (bool, detail). Tasks are independent and gate-checkable.

TASKS = [
    {
        "name": "smoke-status",
        "seeded": {"game.js": "const W=480,H=720;\nglobalThis.Game={step(){},run(){}};\n",
                   "index.html": "<!doctype html><html><body>game</body></html>\n"},
        "prompt": (
            "Work in this directory only. Create a file called docs/STATUS.md with "
            "exactly this content:\\n"
            "## Games\\n"
            "- game.js: PASS\\n"
            "Checked: yes\\n"
            "Then reply with the single line: DONE. Do not touch any other file."
        ),
        "verify": lambda cwd: (
            (cwd / "docs" / "STATUS.md").exists()
            and (cwd / "docs" / "STATUS.md").read_text().strip()
            in (
                "## Games\ngame.js: PASS\nChecked: yes",
                "## Games\n- game.js: PASS\nChecked: yes",
            )
        ),
        "detail": "docs/STATUS.md exists with Games heading, game.js PASS line, Checked line",
    },
    {
        "name": "fn-sum-array",
        "seeded": {"math.js": "// sum all numbers in an array\n"},
        "prompt": (
            "Edit math.js so it defines `function sumAll(xs) { return ...; }` that "
            "returns the sum of every number in the array xs. Handle empty arrays "
            "(return 0). Reply with the line: DONE."
        ),
        "verify": lambda cwd: _js_check(cwd / "math.js",
            "sumAll([1,2,3]) === 6 && sumAll([]) === 0 && sumAll([-1,5,2.5]) === 6.5"),
        "detail": "math.js defines sumAll that sums and handles empty arrays",
    },
    {
        "name": "fix-typo",
        "seeded": {"readme.txt": "The quikc brown fox\njumps over teh lazy dog.\n"},
        "prompt": (
            "Edit readme.txt: fix the two misspellings ('quikc' -> 'quick' and "
            "'teh' -> 'the'). Reply with the single line: DONE."
        ),
        "verify": lambda cwd: (cwd / "readme.txt").read_text()
            == "The quick brown fox\njumps over the lazy dog.\n",
        "detail": "readme.txt has both typos fixed",
    },
    {
        "name": "wrap-fn",
        "seeded": {"util.py": "def greet(name):\n    return f'Hello, {name}'\n"},
        "prompt": (
            "Edit util.py: add a function `parse_names(s)` that splits a "
            "comma-separated string into trimmed non-empty names. "
            "Example: parse_names('alice, bob,,carol') == ['alice','bob','carol']. "
            "Reply with the line: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "util.py",
            "parse_names('alice, bob,,carol') == ['alice','bob','carol'] "
            "and parse_names('') == []"),
        "detail": "util.py defines parse_names handling comma/space/empty",
    },
    {
        "name": "calc-js",
        "seeded": {"calc.js": "// a tiny calculator\n"},
        "prompt": (
            "Edit calc.js so it exposes `function calc(expr) { ... }` that evaluates "
            "a simple expression of the form '<number> <op> <number>' where op is "
            "+ - * or / (plain JavaScript, no eval). Example: calc('3 + 4') == 7, "
            "calc('10 / 4') == 2.5. Reply with the line: DONE."
        ),
        "verify": lambda cwd: _js_check(cwd / "calc.js",
            "calc('3 + 4') === 7 && calc('10 / 4') === 2.5 && calc('2 * 3') === 6 && calc('8 - 3') === 5"),
        "detail": "calc.js defines calc that safely parses '<num> <op> <num>'",
    },
    {
        "name": "bubble-sort",
        "seeded": {"sort.c": "// implement bubble_sort\n"},
        "prompt": (
            "Edit sort.c so it defines `void bubble_sort(int *a, int n)` that sorts "
            "the array in ascending order in place, then `int main(void)` that reads "
            "a line of integers from stdin, sorts them, and prints them space-"
            "separated. Reply with the line: DONE."
        ),
        "verify": lambda cwd: _c_check(cwd / "sort.c", "5 1 4 2 8", "1 2 4 5 8"),
        "detail": "sort.c defines bubble_sort and a working main",
    },
    {
        "name": "fizzbuzz-n",
        "seeded": {"fb.py": "# print fizz/buzz lines\n"},
        "prompt": (
            "Edit fb.py so it defines `def fizzbuzz(limit)` that returns a list of "
            "strings, one per integer 1..limit, where multiples of 3 are 'fizz', "
            "multiples of 5 are 'buzz', multiples of both are 'fizzbuzz', and "
            "everything else is the number as a string. Then define main that calls "
            "fizzbuzz(15) and prints its lines separated by newlines. Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "fb.py",
            "fizzbuzz(3) == ['1','2','fizz'] and fizzbuzz(5)[-1] == 'buzz' "
            "and fizzbuzz(15)[14] == 'fizzbuzz' and fizzbuzz(1) == ['1']"),
        "detail": "fb.py fizzbuzz(limit) is correct incl. fizzbuzz at 15 and edge 1",
    },
    {
        "name": "regex-phone",
        "seeded": {"ph.py": "# normalize phone numbers\n"},
        "prompt": (
            "Edit ph.py so it defines `def norm(phone)` returning a normalized "
            "10-digit US phone string for inputs like '(555) 123-4567', '555-123-4567', "
            "and '5551234567', and None for anything that is not exactly 10 digits "
            "(ignoring spaces, dashes, parens, dots). Then main() prints norm('(555) 123-4567') "
            "and norm('555-123-4567'). Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "ph.py",
            "norm('(555) 123-4567')=='5551234567' and norm('555-123-4567')=='5551234567' "
            "and norm('5551234567')=='5551234567' and norm('12') is None "
            "and norm('555-123-456') is None"),
        "detail": "ph.py norm normalizes US phones and rejects bad lengths",
    },
    {
        "name": "json-filter",
        "seeded": {"data.json": '{"users": [{"name": "alice", "age": 30, "active": true}, {"name": "bob", "age": 17, "active": true}, {"name": "carol", "age": 42, "active": false}]}\n',
                     "filter.py": "# filter users\n"},
        "prompt": (
            "Edit filter.py so it reads data.json (in this directory) and defines "
            "`def active_adults(users)` returning a sorted list of names of users "
            "who are active and 18 or older. Then main() prints them, one per line. "
            "Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "filter.py",
            "import json; users=json.load(open('data.json'))['users']; "
            "assert active_adults(users)==['alice']",
            cwd=cwd)
            and "json" in (cwd / "filter.py").read_text(),
        "detail": "filter.py reads data.json and returns active adults sorted",
    },
    {
        "name": "tic-tac-toe-winner",
        "seeded": {"ttt.py": "# winner\n"},
        "prompt": (
            "Edit ttt.py so it defines `def winner(board)` that takes a 3x3 list of "
            "lists with 'x', 'o' or '' and returns 'x', 'o', or None. A win is three "
            "in a row, column or diagonal. Then main() checks the board "
            "[['x','','o'],['x','o',''],['x','','']] and prints the winner. "
            "Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "ttt.py",
            "winner([['x','','o'],['x','o',''],['x','','']])=='x' "
            "and winner([['','',''],['','',''],['','','']]) is None "
            "and winner([['o','o','o'],['','',''],['','','']])=='o' "
            "and winner([['x','',''],['','x',''],['','','x']])=='x'"),
        "detail": "ttt.py winner detects rows, columns and diagonals",
    },
    {
        "name": "longest-substring",
        "seeded": {"lss.py": "# longest unique substring\n"},
        "prompt": (
            "Edit lss.py so it defines `def longest(s)` returning the length of the "
            "longest substring without repeating characters (sliding window). "
            "Examples: longest('abcabcbb')==3, longest('bbbbb')==1, "
            "longest('pwwkew')==3, longest('')==0. Then main() prints "
            "longest('abcabcbb'). Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "lss.py",
            "longest('abcabcbb')==3 and longest('bbbbb')==1 and longest('pwwkew')==3 "
            "and longest('')==0 and longest('abba')==2 and longest('a')==1"),
        "detail": "lss.py sliding-window longest unique substring incl. abba edge",
    },
    {
        "name": "word-wrap",
        "seeded": {"wrap.py": "# wrap text\n"},
        "prompt": (
            "Edit wrap.py so it defines `def wrap(text, width)` returning the text "
            "wrapped so each line is at most `width` characters, breaking only at "
            "spaces (a word longer than width stays whole on its own line), and "
            "joined with newlines. No trailing space on any line. "
            "Example: wrap('a b c d', 3) == 'a b\\nc d' and wrap('hello world', 5) == "
            "'hello\\nworld'. Then main() prints wrap('the quick brown fox', 8). "
            "Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "wrap.py",
            "wrap('a b c d', 3)=='a b\\nc d' and wrap('hello world', 5)=='hello\\nworld' "
            "and wrap('x', 1)=='x' and wrap('ab cd', 3)=='ab\\ncd' "
            "and wrap('one two three four', 7)=='one two\\nthree\\nfour'"),
        "detail": "wrap.py greedy word wrap with exact space handling",
    },
    {
        "name": "matrix-rotate",
        "seeded": {"mx.py": "# rotate matrix\n"},
        "prompt": (
            "Edit mx.py so it defines `def rotate90(m)` returning a new matrix (list "
            "of lists) that is m rotated 90 degrees clockwise, without mutating m. "
            "Then main() prints rotate90([[1,2],[3,4]]). Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "mx.py",
            "rotate90([[1,2],[3,4]])==[[3,1],[4,2]] and rotate90([[1]])==[[1]] "
            "and rotate90([[1,2,3],[4,5,6],[7,8,9]])==[[7,4,1],[8,5,2],[9,6,3]] "
            "and (lambda m: (rotate90(m), m)[1])([[1,2],[3,4]])==[[1,2],[3,4]]"),
        "detail": "mx.py 90-degree matrix rotation, non-mutating",
    },
    {
        "name": "csv-parse",
        "seeded": {"csv.py": "# parse csv\n"},
        "prompt": (
            "Edit csv.py so it defines `def parse(line)` that parses one CSV line into "
            "a list of fields, honoring double quotes and commas inside quotes. "
            "Examples: parse('a,b,c')==['a','b','c'], parse('\"a,b\",c')==['a,b','c'], "
            "parse('\"\"')==[''], parse('\"a\"\"b\"')==['a\"b']. "
            "Then main() prints parse('\"x,y\",z'). Reply with: DONE."
        ),
        "verify": lambda cwd: _py_check(cwd / "csv.py",
            "parse('a,b,c')==['a','b','c'] and parse('\"a,b\",c')==['a,b','c'] "
            "and parse('\"\"')==[''] and parse('\"a\"\"b\"')==['a\"b'] "
            "and parse('1,\"two, words\",3')==['1','two, words','3']"),
        "detail": "csv.py correct CSV parser with quotes/escapes",
    },
]


def _js_check(path, expr):
    if not path.exists():
        return False
    js = path.read_text()
    if "function " not in js:
        return False
    # crude: node loads the file then evaluates the assertion
    tmp = Path(tempfile.mkdtemp()) / "check.js"
    tmp.write_text(js + f"\nconsole.log(({expr}) ? 'PASS' : 'FAIL');\n")
    r = subprocess.run(["node", str(tmp)], capture_output=True, text=True, timeout=20)
    return r.stdout.strip() == "PASS"


def _py_check(path, expr, cwd=None):
    """Load path's module code, run any leading statements, then assert expr."""
    if not path.exists():
        return False
    code = path.read_text()
    pre, _, assertion = expr.rpartition("assert ")
    lines = [f"import ast; ast.parse({code!r})", code, pre, f"assert {assertion}"]
    r = subprocess.run([sys.executable, "-c", "\n".join(lines)],
                       capture_output=True, text=True, timeout=20, cwd=cwd)
    return r.returncode == 0


def _c_check(path, stdin, want):
    if not path.exists():
        return False
    r = subprocess.run(["cc", "-x", "c", "-o", "/tmp/swarm-bench/_cprobe", str(path)],
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False
    r = subprocess.run(["/tmp/swarm-bench/_cprobe"], input=stdin,
                       capture_output=True, text=True, timeout=20)
    return r.stdout.strip() == want


# ---------------------------------------------------------------- runner


def run_one(model, task, rounds=12, timeout=600, retain=False):
    """Run one task through flint in a temp worktree. Returns a result dict.

    With retain=True the worktree is kept under /tmp/swarm-bench/kept/<model>/<task>
    when the task fails, so a FAIL can be inspected instead of guessed at.
    """
    work = Path(tempfile.mkdtemp(prefix="bench-"))
    for name, body in task["seeded"].items():
        p = work / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    charge = work.parent / f"charges-{int(time.time()*1000)}.jsonl"
    prompt = task["prompt"]
    env = dict(os.environ, FLINT_CHARGE_FILE=str(charge), FLINT_MAX_STEPS=str(rounds))
    t0 = time.time()
    failed = None
    kept = None
    try:
        r = subprocess.run([*FLINT, "-p", prompt, "-m", model, "--yolo", "-C", str(work)],
                           capture_output=True, text=True, timeout=timeout, env=env)
        wall = time.time() - t0
        ok, detail = task["verify"](work), ""
        if not ok:
            detail = "(see output)"
            if retain:
                kept = Path("/tmp/swarm-bench/kept") / model.replace("/", "_") / task["name"]
                kept.parent.mkdir(parents=True, exist_ok=True)
                if kept.exists():
                    shutil.rmtree(kept)
                shutil.copytree(work, kept)
        return {
            "model": model, "task": task["name"],
            "rc": r.returncode, "ok": ok, "detail": detail,
            "secs": round(wall, 1),
            "out_tail": r.stdout.strip()[-200:],
            "err_tail": r.stderr.strip()[-200:],
            "charges": sum(json.loads(l)["usd"] for l in charge.read_text().splitlines())
                       if charge.exists() else 0.0,
            "kept": str(kept) if kept else None,
        }
    except subprocess.TimeoutExpired:
        return {"model": model, "task": task["name"], "rc": None, "ok": False,
                "detail": f"TIMEOUT {timeout}s", "secs": timeout, "out_tail": "",
                "err_tail": "", "charges": None, "kept": None}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma-separated model slugs")
    ap.add_argument("--tasks", default=None, help="comma-separated task indices (1-based)")
    ap.add_argument("--rounds", type=int, default=12, help="FLINT_MAX_STEPS per turn")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--retain", action="store_true",
                    help="keep failed worktrees under /tmp/swarm-bench/kept for inspection")
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    tasks = (TASKS if not a.tasks else [TASKS[int(i) - 1] for i in a.tasks.split(",")])
    if a.dry:
        print("task battery:")
        for i, t in enumerate(tasks, 1):
            print(f"  {i}. {t['name']}: {t['detail']}")
        print("models:", models)
        return 0
    results = []
    for t in tasks:
        for m in models:
            print(f"\n== {m} / {t['name']} ==", flush=True)
            r = run_one(m, t, rounds=a.rounds, timeout=a.timeout, retain=a.retain)
            results.append(r)
            ok = {True: "PASS", False: "FAIL"}[r["ok"]]
            print(f"  {ok}  {r['secs']}s  ${r['charges'] if r['charges'] is not None else 'N/A':.4f}")
            if r.get("out_tail"):
                print(f"  out: {r['out_tail'][-180:]}")
    print("\n=== SUMMARY ===")
    for m in models:
        subset = [r for r in results if r["model"] == m]
        passed = sum(1 for r in subset if r["ok"])
        cost = sum(r["charges"] or 0 for r in subset)
        secs = sum(r["secs"] for r in subset)
        print(f"{m:<42} {passed}/{len(subset)} tasks  ${cost:.3f} total  {secs:.0f}s")
    out = Path("/tmp/swarm-bench/results.jsonl")
    with out.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nresults appended to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())