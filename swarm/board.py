"""Read-only scrum projection. No daemon imports, model calls or queue mutations."""
import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

COLUMNS = ('Backlog', 'Ready', 'In Progress', 'In Review', 'Blocked', 'Done', 'Cancelled')


def read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def rows(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                result.append(row)
        except ValueError:
            pass  # A live append may have an incomplete last line.
    return result


def git(repo, *args):
    p = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, timeout=15)
    return p.returncode, p.stdout.strip()


def committed(repo, trunk, path, default):
    rc, raw = git(repo, 'show', f'{trunk}:{path}')
    try:
        return json.loads(raw) if rc == 0 else default
    except ValueError:
        return default


def verify(repo, trunk, task, attempts):
    """Verify recorded checks and exact integrated tree; never trust a done label alone."""
    result = {'verified': False, 'reason': 'No accepted attempt with passing gates and matching review tree.'}
    for attempt in attempts:
        if attempt.get('task', {}).get('id') != task['id'] or attempt.get('phase') != 'accepted':
            continue
        sha = attempt.get('commit', '')
        if not re.fullmatch('[0-9a-f]{40}', sha):
            continue
        if git(repo, 'merge-base', '--is-ancestor', sha, trunk)[0]:
            result['reason'] = 'Accepted commit is no longer reachable from trunk.'
            continue
        gates = attempt.get('gates') or []
        tree = git(repo, 'rev-parse', sha + '^{tree}')[1]
        folder = Path(attempt['_folder'])
        reviews = [read(p, {}) for p in folder.glob('review-*.json')]
        contract = read(folder / 'contract.json', {})
        criteria = contract.get('acceptance') or []
        ids = {c.get('id') for c in criteria}
        review = next((r for r in reviews if r.get('verdict') == 'approve' and r.get('tree') == tree
                       and ids and {c.get('criterion') for c in r.get('checks', [])} == ids
                       and len(r.get('checks', [])) == len(ids)
                       and all(c.get('passed') is True and c.get('evidence') for c in r['checks'])), None)
        if (not gates or not all(g.get('passed') is True and g.get('command') and
                                (folder / g.get('log', '')).is_file() for g in gates)
                or tree != attempt.get('reviewed_tree') or not review):
            continue
        result = {'verified': True, 'reason': 'Recorded gates pass; approved criteria and integrated tree match.',
                  'commit': sha, 'completed_at': attempt.get('finished'), 'evidence': str(folder),
                  'commands': [g['command'] for g in gates], 'criteria': criteria,
                  'review_summary': review.get('summary', ''),
                  'files': git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', sha)[1].splitlines()}
        break
    return result


def build(repo, state, trunk='swarm/trunk', roadmap_file=None):
    repo, state = Path(repo), Path(state)
    history = {r['id']: r for r in rows(state / 'done.jsonl') if r.get('id')}
    pending = {r['id']: r for r in rows(state / 'queue.jsonl') if r.get('id')}
    now = read(state / 'now.json', {})
    fresh = time.time() - now.get('at', 0) < 180
    workers = {w.get('task'): w for w in now.get('workers', [])} if fresh else {}
    attempts = []
    for p in (state / 'attempts').glob('*/attempt.json'):
        a = read(p, {})
        a['_folder'] = str(p.parent)
        attempts.append(a)
    attempts.sort(key=lambda a: a.get('finished', 0), reverse=True)
    packets = committed(repo, trunk, roadmap_file, {}).get('tasks', []) if roadmap_file else []
    ledger = committed(repo, trunk, 'docs/roadmap/EXECUTION.json', {}) if packets else {}
    by_title = {p['title']: p for p in packets}
    all_tasks = {**history, **pending}
    cards = []
    for tid, t in all_tasks.items():
        card = {'id': tid, 'title': t.get('title', tid), 'status': 'Backlog',
                'reason': t.get('note', ''), 'attempts': t.get('attempts', 0),
                'dependencies': t.get('depends_on', []), 'origin': t.get('origin', 'unknown')}
        if tid in pending:
            missing = [d for d in card['dependencies'] if history.get(d, {}).get('status') != 'done']
            if missing or t.get('not_before', 0) > time.time():
                card.update(status='Blocked', reason='Waiting on ' + ', '.join(missing) if missing else 'Retry cooldown')
            elif t.get('claimed') and tid in workers:
                w = workers[tid]
                card.update(status='In Review' if w.get('role') in ('adversary', 'reviewer', 'judge') else 'In Progress',
                            reason=w.get('doing', ''), owner=w.get('worker'))
            elif t.get('claimed'):
                card.update(status='Blocked', reason='Claim has no fresh matching worker; needs reconciliation.')
            else:
                card['status'] = 'Ready'
        elif t.get('status') == 'done':
            proof = verify(repo, trunk, t, attempts)
            card.update(proof, status='Done' if proof['verified'] else 'In Review')
            if packets and t.get('title') not in by_title:
                card.update(status='In Review', reason='No matching active roadmap packet. Commit evidence cannot prove this broader task completed.')
        elif t.get('dismissed_at'):
            card.update(status='Cancelled', reason='Dismissed by owner; evidence retained.')
        elif t.get('status') in ('parked', 'split'):
            card['status'] = 'Blocked'
        elif t.get('status') in ('removed', 'cancelled', 'superseded'):
            card['status'] = 'Cancelled'
        cards.append(card)
    by_packet = {}
    for p in packets:
        existing = next((c for c in reversed(cards) if c['title'] == p['title']), None)
        if existing:
            existing['packet'] = p['id']
            by_packet[p['id']] = existing
            continue
        e = ledger.get('packets', {}).get(p['id'], {})
        card = {'id': p['id'], 'packet': p['id'], 'title': p['title'], 'status': 'Backlog',
                'dependencies': p.get('depends_on', []), 'reason': '', 'origin': p.get('dispatch')}
        if e.get('status') == 'accepted':
            # Coordinator prose is visible, but is not independently machine verified.
            card.update(status='In Review', reason='Coordinator acceptance recorded; inspect its verification evidence.',
                        commit=e.get('accepted_commit'), review_summary=e.get('evidence', ''))
        cards.append(card)
        by_packet[p['id']] = card
    accepted = {pid for pid, c in by_packet.items() if c['status'] == 'Done'}
    # Coordinator gates remain separate from verified task completion.
    for pid, e in ledger.get('packets', {}).items():
        sha = e.get('accepted_commit')
        if e.get('status') == 'accepted' and e.get('evidence') and (
                (sha and git(repo, 'merge-base', '--is-ancestor', sha, trunk)[0] == 0) or
                (not sha and by_packet.get(pid, {}).get('origin') == 'coordinator')):
            accepted.add(pid)
    # Remove an entire dependent chain if any predecessor/gate is unverified.
    changed = True
    while changed:
        changed = False
        for p in packets:
            if p['id'] not in accepted:
                continue
            missing = any(d not in accepted for d in p.get('depends_on', []))
            missing |= any(not ledger.get('gates', {}).get(g, {}).get('evidence')
                           for g in p.get('coordinator_gates', []))
            if missing:
                accepted.remove(p['id'])
                changed = True
    for p in packets:
        c = by_packet[p['id']]
        missing = [d for d in p.get('depends_on', []) if d not in accepted]
        missing += [g for g in p.get('coordinator_gates', []) if not ledger.get('gates', {}).get(g, {}).get('evidence')]
        if missing and c['status'] not in ('Cancelled', 'In Progress', 'In Review'):
            c.update(status='Blocked', reason='Waiting on ' + ', '.join(missing))
        elif c['status'] == 'Backlog' and p.get('dispatch') == 'swarm':
            c['status'] = 'Ready'
    counts = {s: sum(c['status'] == s for c in cards) for s in COLUMNS}
    stale_tasks = [tid for tid in workers if tid not in pending]
    return {'repo': str(repo), 'as_of': time.time(), 'columns': COLUMNS, 'cards': cards, 'counts': counts,
            'stale_workers': stale_tasks, 'worker_snapshot_fresh': fresh,
            'notice': 'Done verifies recorded gates, review criteria and the integrated commit tree. It does not prove deployment or manual play quality. Coordinator approvals and unmatched roadmap scope stay In Review.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True)
    a = ap.parse_args()
    repo = str(Path(a.repo).expanduser().resolve())
    root = Path(__file__).resolve().parent
    slug = re.sub(r'[^A-Za-z0-9._-]+', '-', Path(repo).name) + '-' + hashlib.sha1(repo.encode()).hexdigest()[:6]
    config = read(root / 'configs' / (slug + '.json'), read(root / 'config.json', {}))
    print(json.dumps(build(repo, root / 'state' / slug, config.get('trunk', 'swarm/trunk'), config.get('roadmap_file'))))


if __name__ == '__main__':
    main()
