"""Validate native artifact references against the reviewed prospective state."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from claude_obsidian.transaction import BUNDLE_SCHEMA, apply_bundle, inspect_bundle


class ProspectiveCanvasTests(unittest.TestCase):
    def test_both_write_orders_and_planned_canvas_references(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as directory:
                vault = Path(directory)
                (vault / 'wiki').mkdir()
                (vault / '.obsidian').mkdir()
                canvas = {'nodes': [
                    {'id': 'note', 'type': 'file', 'file': 'wiki/New.md', 'x': 0, 'y': 0, 'width': 100, 'height': 100},
                    {'id': 'board', 'type': 'file', 'file': 'wiki/canvases/Other.canvas', 'x': 110, 'y': 0, 'width': 100, 'height': 100},
                ], 'edges': [], 'custom': {'preserve': True}}
                content = json.dumps(canvas)
                writes = [
                    {'path': 'wiki/New.md', 'mode': 'create', 'content': '# New\n'},
                    {'path': 'wiki/canvases/Other.canvas', 'mode': 'create', 'content': '{"nodes":[],"edges":[]}'},
                    {'path': 'wiki/canvases/Board.canvas', 'mode': 'create', 'content': content},
                ]
                if reverse:
                    writes.reverse()
                bundle = {'schema': BUNDLE_SCHEMA, 'operation_id': 'prospective', 'operation_type': 'generic',
                          'expected_hashes': {w['path']: None for w in writes}, 'writes': writes}
                plan = inspect_bundle(vault, bundle)
                apply_bundle(vault, bundle, approved_plan_sha256=plan['approval_sha256'])
                self.assertEqual(content, (vault / 'wiki/canvases/Board.canvas').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
