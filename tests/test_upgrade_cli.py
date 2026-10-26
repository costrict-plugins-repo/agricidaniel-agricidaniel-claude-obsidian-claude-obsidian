"""Exercise public local-upgrade commands through their real parser."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from claude_obsidian.cli import build_parser
from claude_obsidian.transaction import BUNDLE_SCHEMA


class UpgradeCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.vault = self.base / 'vault'
        (self.vault / '.obsidian').mkdir(parents=True)
        (self.vault / 'wiki').mkdir()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/claude-obsidian.py'), *args],
                              text=True, capture_output=True, cwd=self.base, timeout=20)

    def test_capacity_has_readonly_status_and_both_vault_argument_positions(self):
        run = self.run_cli('capacity', '--vault', str(self.vault))
        self.assertEqual(0, run.returncode, run.stderr)
        self.assertTrue(json.loads(run.stdout)['ok'])
        self.assertFalse((self.vault / '.vault-meta').exists())
        for args in [('capacity', '--vault', str(self.vault), 'compact', '--operation-id', 'empty'),
                     ('capacity', 'compact', '--vault', str(self.vault), '--operation-id', 'empty')]:
            run = self.run_cli(*args)
            self.assertEqual(0, run.returncode, run.stderr)
            self.assertEqual('noop', json.loads(run.stdout)['status'])

    def test_incomplete_named_workflow_is_a_machine_readable_nonzero_result(self):
        bundle = {'schema': BUNDLE_SCHEMA, 'operation_id': 'incomplete', 'operation_type': 'save',
                  'expected_hashes': {'wiki/Note.md': None},
                  'writes': [{'path': 'wiki/Note.md', 'mode': 'create', 'content': '# Note\n'}]}
        path = self.base / 'bundle.json'
        path.write_text(json.dumps(bundle))
        run = self.run_cli('workflow', 'inspect', str(path), '--vault', str(self.vault))
        self.assertEqual(1, run.returncode, run.stderr)
        self.assertTrue(json.loads(run.stdout)['storage_valid'])
        self.assertFalse(json.loads(run.stdout)['workflow_complete'])
        run = self.run_cli('workflow', 'apply', str(path), '--vault', str(self.vault))
        self.assertEqual(2, run.returncode)
        self.assertIn('PLAN_APPROVAL_REQUIRED', run.stderr)
        self.assertFalse((self.vault / '.vault-meta').exists())

    def test_malformed_bundles_are_controlled_cli_errors(self):
        deep = self.base / 'deep.json'
        deep.write_text('[' * 200000 + ']' * 200000)
        listed = self.base / 'listed.json'
        listed.write_text(json.dumps({'schema': BUNDLE_SCHEMA, 'operation_id': 'listed', 'operation_type': 'save',
                                      'expected_hashes': {'wiki/x.md': None},
                                      'writes': [{'path': 'wiki/x.md', 'mode': ['create'], 'content': 'x\n'}]}))
        for path, code in ((deep, 'INVALID_BUNDLE'), (listed, 'INVALID_WRITE_MODE')):
            for command in ('transaction', 'workflow'):
                with self.subTest(path=path.name, command=command):
                    run = self.run_cli(command, 'inspect', str(path), '--vault', str(self.vault))
                    self.assertEqual(2, run.returncode, run.stderr)
                    self.assertIn(f'ERR {code}:', run.stderr)
                    self.assertNotIn('Traceback', run.stderr)
        self.assertFalse((self.vault / '.vault-meta').exists())

    def test_generated_policy_is_explicit_and_product_root_is_refused(self):
        args = build_parser().parse_args(['lint', '--vault', str(self.vault), '--frontmatter-policy', 'generated'])
        self.assertEqual('generated', args.frontmatter_policy)
        run = self.run_cli('workflow', 'inspect', 'missing.json', '--vault', str(ROOT))
        self.assertEqual(2, run.returncode)
        self.assertTrue(any(code in run.stderr for code in ('PLUGIN_ROOT_IS_NOT_VAULT', 'VAULT_SENTINEL_MISSING')), run.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
