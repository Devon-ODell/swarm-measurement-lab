import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'swarm'))
import board


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.state = self.root / 'state'
        self.attempt = self.state / 'attempts' / 't1-a1'
        self.attempt.mkdir(parents=True)
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')
        (self.repo / 'file').write_text('working change')
        self.git('add', '.')
        self.git('commit', '-qm', 'bounded task')
        self.git('branch', 'swarm/trunk')
        self.sha = self.git('rev-parse', 'HEAD')
        self.tree = self.git('rev-parse', 'HEAD^{tree}')
        self.task = {'id': 't1', 'title': 'Bounded task', 'status': 'done'}
        self.a = {'task': self.task, 'phase': 'accepted', 'commit': self.sha,
                  'reviewed_tree': self.tree, 'finished': time.time(),
                  'gates': [{'passed': True, 'command': 'test', 'log': 'gate.log'}]}
        (self.attempt / 'gate.log').write_text('test passed')
        self.write(self.attempt / 'contract.json', {'acceptance': [{'id': 'C1', 'text': 'specific behavior'}]})
        self.review = {'verdict': 'approve', 'tree': self.tree,
                       'checks': [{'criterion': 'C1', 'passed': True, 'evidence': 'observed behavior'}]}
        self.save()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True).strip()

    def write(self, path, data):
        path.write_text(json.dumps(data) + '\n')

    def save(self):
        self.write(self.attempt / 'attempt.json', self.a)
        self.write(self.attempt / 'review-01.json', self.review)
        self.write(self.state / 'done.jsonl', self.task)

    def card(self):
        return board.build(self.repo, self.state)['cards'][0]

    def test_valid_evidence_completes_task(self):
        self.assertEqual(self.card()['status'], 'Done')

    def test_done_label_without_evidence_is_not_done(self):
        (self.attempt / 'attempt.json').unlink()
        self.assertEqual(self.card()['status'], 'In Review')

    def test_failed_gate_or_missing_log_is_not_done(self):
        self.a['gates'][0]['passed'] = False
        self.save()
        self.assertEqual(self.card()['status'], 'In Review')
        self.a['gates'][0]['passed'] = True
        self.save()
        (self.attempt / 'gate.log').unlink()
        self.assertEqual(self.card()['status'], 'In Review')

    def test_review_for_different_tree_is_not_done(self):
        self.review['tree'] = '0' * 40
        self.save()
        self.assertEqual(self.card()['status'], 'In Review')

    def test_missing_or_duplicate_acceptance_check_is_not_done(self):
        self.review['checks'] *= 2
        self.save()
        self.assertEqual(self.card()['status'], 'In Review')
        self.review['checks'] = []
        self.save()
        self.assertEqual(self.card()['status'], 'In Review')

    def test_closed_task_in_worker_snapshot_is_not_in_progress(self):
        self.task['status'] = 'parked'
        self.save()
        self.write(self.state / 'now.json', {'at': time.time(), 'workers': [{'task': 't1'}]})
        result = board.build(self.repo, self.state)
        self.assertEqual(result['cards'][0]['status'], 'Blocked')
        self.assertEqual(result['stale_workers'], ['t1'])

    def test_claim_without_worker_is_blocked(self):
        self.write(self.state / 'queue.jsonl', {**self.task, 'claimed': time.time()})
        self.assertEqual(self.card()['status'], 'Blocked')

    def test_incomplete_append_is_ignored_without_losing_valid_row(self):
        with (self.state / 'done.jsonl').open('a') as f:
            f.write('{"id":')
        self.assertEqual(self.card()['status'], 'Done')

    def test_packet_dependencies_prevent_done_and_unmatched_scope_needs_review(self):
        p = self.repo / 'docs/roadmap'
        p.mkdir(parents=True)
        self.write(p / 'tasks.json', {'tasks': [{'id': 'F02', 'title': 'Bounded task', 'dispatch': 'swarm', 'depends_on': ['F01']}]})
        self.git('add', '.')
        self.git('commit', '-qm', 'roadmap')
        self.git('branch', '-f', 'swarm/trunk', 'HEAD')
        result = board.build(self.repo, self.state, roadmap_file='docs/roadmap/tasks.json')
        self.assertEqual(result['cards'][0]['status'], 'Blocked')
        self.task['title'] = 'Do the entire roadmap'
        self.save()
        result = board.build(self.repo, self.state, roadmap_file='docs/roadmap/tasks.json')
        self.assertEqual(result['cards'][0]['status'], 'In Review')

    def test_unreachable_commit_is_not_done(self):
        self.git('checkout', '--orphan', 'other')
        self.git('commit', '-qm', 'unrelated')
        self.git('branch', '-f', 'swarm/trunk', 'HEAD')
        self.assertEqual(self.card()['status'], 'In Review')


if __name__ == '__main__':
    unittest.main()
