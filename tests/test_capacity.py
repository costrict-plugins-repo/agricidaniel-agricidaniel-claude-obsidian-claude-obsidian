"""Capacity boundaries and reviewed legacy formatting recovery."""
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from claude_obsidian.capacity import compaction_bundle, inspect_capacity
from claude_obsidian.ledgers import SOURCE_SCHEMA, CLAIM_SCHEMA
from claude_obsidian.transaction import apply_bundle, inspect_bundle, TransactionError


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.vault = Path(self.temporary.name) / 'vault'
        (self.vault / 'wiki/meta/ledgers').mkdir(parents=True)
        (self.vault / '.obsidian').mkdir()

    def test_absent_ledger_is_not_a_zero_byte_measurement(self):
        report = inspect_capacity(self.vault)
        self.assertTrue(report['ok'])
        self.assertTrue(all(x['status'] == 'absent' and x['size_bytes'] is None for x in report['ledgers']))

    def test_lossless_compaction_is_reviewed_and_recoverable(self):
        source = self.vault / 'wiki/meta/ledgers/source-ledger.json'
        claim = self.vault / 'wiki/meta/ledgers/claim-ledger.json'
        source.write_text(json.dumps({'schema': SOURCE_SCHEMA, 'generated_at': '2026-10-17T12:00:00Z', 'sources': {}}) + ' ' * (16 * 1024 * 1024))
        claim.write_text(json.dumps({'schema': CLAIM_SCHEMA, 'generated_at': '2026-10-17T12:00:00Z', 'claims': {}}))
        before = source.read_bytes()
        self.assertFalse(inspect_capacity(self.vault)['ok'])
        bundle = compaction_bundle(self.vault, operation_id='capacity-recovery')
        self.assertEqual(before, source.read_bytes())
        plan = inspect_bundle(self.vault, bundle)
        with self.assertRaises(TransactionError):
            apply_bundle(self.vault, bundle, approved_plan_sha256='0' * 64)
        self.assertEqual(before, source.read_bytes())
        apply_bundle(self.vault, bundle, approved_plan_sha256=plan['approval_sha256'])
        self.assertEqual(json.loads(before), json.loads(source.read_bytes()))
        backup = self.vault / '.vault-meta/transactions/capacity-recovery/backups/0000.original'
        # Backups are bound by the journal, regardless of sorted path position.
        journal = json.loads((backup.parent.parent / 'journal.json').read_text())
        entry = next(x for x in journal['writes'] if x['path'] == 'wiki/meta/ledgers/source-ledger.json')
        self.assertEqual(hashlib.sha256(before).hexdigest(), entry['original_sha256'])
        self.assertEqual(before, (backup.parent / entry['backup']).read_bytes())
        self.assertTrue(inspect_capacity(self.vault)['ok'])

    def test_compaction_never_prunes_records_to_fit(self):
        source = self.vault / 'wiki/meta/ledgers/source-ledger.json'
        original = json.dumps({'schema': SOURCE_SCHEMA, 'sources': {}, 'data': 'x' * 200})
        source.write_text(original)
        with mock.patch('claude_obsidian.capacity.MAX_CANONICAL_LEDGER_BYTES', 100):
            with self.assertRaises(TransactionError) as caught:
                compaction_bundle(self.vault, operation_id='too-large')
        self.assertEqual('LEDGER_MIGRATION_REQUIRED', caught.exception.code)
        self.assertEqual(original, source.read_text())
        self.assertFalse((self.vault / '.vault-meta').exists())

    def test_duplicate_json_keys_and_symlink_are_refused(self):
        source = self.vault / 'wiki/meta/ledgers/source-ledger.json'
        source.write_text('{"sources":{},"sources":{}}')
        with mock.patch('claude_obsidian.capacity.MAX_CANONICAL_LEDGER_BYTES', 30), \
                self.assertRaises(TransactionError) as caught:
            compaction_bundle(self.vault, operation_id='duplicate')
        self.assertEqual('INVALID_PROVENANCE_LEDGER', caught.exception.code)
        source.unlink()
        outside = Path(self.temporary.name) / 'private.json'
        outside.write_text('{}')
        source.symlink_to(outside)
        with self.assertRaises(TransactionError):
            compaction_bundle(self.vault, operation_id='symlink')

    def test_numeric_tokens_and_escaped_string_values_are_preserved_exactly(self):
        source = self.vault / 'wiki/meta/ledgers/source-ledger.json'
        original = '{ "schema": "' + SOURCE_SCHEMA + '", "sources": {}, "custom_measurement": 1.0000000000000001, "custom_string": "a \\"quote\\" \\t text" }'
        source.write_text(original)
        with mock.patch('claude_obsidian.capacity.MAX_CANONICAL_LEDGER_BYTES', len(original)):
            plan = compaction_bundle(self.vault, operation_id='precise')
        text = plan['writes'][0]['content']
        self.assertIn('1.0000000000000001', text)
        self.assertIn('"a \\"quote\\" \\t text"', text)
        self.assertNotIn(':1.0,', text)

    def test_only_ledgers_at_or_above_near_capacity_are_rewritten(self):
        source = self.vault / 'wiki/meta/ledgers/source-ledger.json'
        claim = self.vault / 'wiki/meta/ledgers/claim-ledger.json'
        claim.write_text(json.dumps({'schema': CLAIM_SCHEMA, 'claims': {}}, indent=2) + '\n')
        claim_before = claim.read_bytes()
        record = json.dumps({'schema': SOURCE_SCHEMA, 'sources': {}}, indent=2)
        limit = 1000
        threshold = 850  # the inspect_capacity near_capacity boundary at this limit
        with mock.patch('claude_obsidian.capacity.MAX_CANONICAL_LEDGER_BYTES', limit):
            source.write_text(record + ' ' * (threshold - 1 - len(record)))
            self.assertEqual('within_capacity', inspect_capacity(self.vault)['ledgers'][0]['status'])
            empty = compaction_bundle(self.vault, operation_id='nothing-near')
            self.assertEqual(([], {}), (empty['writes'], empty['expected_hashes']))
            source.write_text(record + ' ' * (threshold - len(record)))
            self.assertEqual('near_capacity', inspect_capacity(self.vault)['ledgers'][0]['status'])
            bundle = compaction_bundle(self.vault, operation_id='near-only')
        self.assertEqual(['wiki/meta/ledgers/source-ledger.json'], [w['path'] for w in bundle['writes']])
        self.assertEqual({'wiki/meta/ledgers/source-ledger.json': hashlib.sha256(source.read_bytes()).hexdigest()},
                         bundle['expected_hashes'])
        self.assertEqual(json.loads(record), json.loads(bundle['writes'][0]['content']))
        self.assertEqual(claim_before, claim.read_bytes())

    def test_unfittable_ledger_is_refused_before_parsing_or_scanning(self):
        source = self.vault / 'wiki/meta/ledgers/source-ledger.json'
        # No JSON whitespace at all: compaction cannot shrink it below the limit.
        original = json.dumps({'schema': SOURCE_SCHEMA, 'sources': {}, 'data': 'x' * 2000}, separators=(',', ':'))
        source.write_text(original)
        with mock.patch('claude_obsidian.capacity.MAX_CANONICAL_LEDGER_BYTES', 1000), \
                mock.patch('claude_obsidian.capacity.strict_json_loads', side_effect=AssertionError('parsed')), \
                mock.patch('claude_obsidian.capacity._compact_json_bytes', side_effect=AssertionError('scanned')):
            with self.assertRaises(TransactionError) as caught:
                compaction_bundle(self.vault, operation_id='bounded-refusal')
        self.assertEqual('LEDGER_MIGRATION_REQUIRED', caught.exception.code)
        self.assertEqual(original, source.read_text())
        # Whitespace inside strings is retained, so a ledger whose cheap bound
        # fits is still refused by the exact compacted size.
        padded = '{"schema":"' + SOURCE_SCHEMA + '","sources":{},"data":"' + ' ' * 2000 + '"}'
        source.write_text(padded)
        with mock.patch('claude_obsidian.capacity.MAX_CANONICAL_LEDGER_BYTES', 1000):
            with self.assertRaises(TransactionError) as caught:
                compaction_bundle(self.vault, operation_id='string-whitespace')
        self.assertEqual('LEDGER_MIGRATION_REQUIRED', caught.exception.code)
        self.assertEqual(padded, source.read_text())

    def test_compaction_matches_the_character_reference_exactly(self):
        def reference(text):
            characters, in_string, escaped = [], False, False
            for character in text:
                if in_string:
                    characters.append(character)
                    if escaped:
                        escaped = False
                    elif character == '\\':
                        escaped = True
                    elif character == '"':
                        in_string = False
                elif character == '"':
                    in_string = True
                    characters.append(character)
                elif character not in ' \t\r\n':
                    characters.append(character)
            return ''.join(characters) + '\n'

        source = self.vault / 'wiki/meta/ledgers/source-ledger.json'
        generator = random.Random(203)
        alphabet = ['a', ' ', '\t', '"', '\\', '/', 'é', ' ', '{', '}', ':', ',', '[', ']', '\n', '\U0001f600']
        for index in range(40):
            values = {f'k {n} \\"': ''.join(generator.choice(alphabet) for _ in range(generator.randrange(30)))
                      for n in range(generator.randrange(1, 6))}
            document = {'schema': SOURCE_SCHEMA, 'sources': {}, 'values': values,
                        'numbers': [1.0000000000000001, -0.0, 12345678901234567890, True, None]}
            text = json.dumps(document, indent=generator.choice([None, 1, 2, '\t']),
                              ensure_ascii=generator.random() < 0.5)
            if generator.random() < 0.5:
                # Raw newlines only occur in formatting; strings escape them.
                text = text.replace('\n', '\r\n')
            source.write_text(text, encoding='utf-8', newline='')
            with self.subTest(index=index), \
                    mock.patch('claude_obsidian.capacity.MAX_CANONICAL_LEDGER_BYTES', len(text.encode('utf-8'))):
                plan = compaction_bundle(self.vault, operation_id=f'oracle-{index}')
                expected = reference(text)
                if expected == text:
                    self.assertEqual([], plan['writes'])
                else:
                    self.assertEqual(expected, plan['writes'][0]['content'])
                    self.assertEqual(json.loads(text), json.loads(plan['writes'][0]['content']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
