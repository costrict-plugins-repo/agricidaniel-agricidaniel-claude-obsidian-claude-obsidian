#!/usr/bin/env python3
"""Independent metric arithmetic, label safety, and offline fixture acceptance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import claude_obsidian.retrieval_eval as evaluation
from claude_obsidian.retrieval_eval import (
    EvaluationError,
    QUERY_SCHEMA,
    load_queries,
    provision_fixture,
    run_quality,
    score_ranked_pages,
)


def labels(path: Path, queries: list[dict]) -> Path:
    path.write_text(json.dumps({"schema": QUERY_SCHEMA, "queries": queries}), encoding="utf-8")
    return path


def snapshot(root: Path) -> dict[str, tuple[str, int]]:
    return {path.relative_to(root).as_posix(): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


class MetricTests(unittest.TestCase):
    def test_hand_calculated_recall_and_reciprocal_rank(self) -> None:
        result = score_ranked_pages(["wiki/X.md", "wiki/A.md", "wiki/A.md", "wiki/B.md"], ["wiki/A.md", "wiki/B.md"])
        self.assertEqual(result["recall_at_1"], 0)
        self.assertEqual(result["recall_at_5"], 1)
        self.assertEqual(result["recall_at_10"], 1)
        self.assertEqual(result["reciprocal_rank"], 0.5)
        self.assertEqual(result["first_relevant_rank"], 2)

    def test_cutoff_misses_and_multiple_gold_denominator(self) -> None:
        result = score_ranked_pages(["wiki/A.md"], ["wiki/A.md", "wiki/B.md"])
        self.assertEqual(result["recall_at_1"], 0.5)
        self.assertEqual(result["recall_at_10"], 0.5)
        self.assertEqual(result["reciprocal_rank"], 1)
        self.assertEqual(score_ranked_pages(["wiki/X.md", "wiki/A.md"], ["wiki/A.md"], mrr_depth=1)["reciprocal_rank"], 0)

    def test_empty_gold_is_separate_from_recall(self) -> None:
        for ranked, correct in (([], True), (["wiki/X.md"], False)):
            result = score_ranked_pages(ranked, [])
            self.assertIsNone(result["recall_at_10"])
            self.assertIsNone(result["reciprocal_rank"])
            self.assertEqual(result["unanswerable_correct"], correct)

    def test_process_arguments_are_offline_and_do_not_build(self) -> None:
        output = subprocess.CompletedProcess([], 0, json.dumps({"strategy": "bm25-only", "candidates": []}), "")
        with mock.patch.object(evaluation.subprocess, "run", return_value=output) as call:
            evaluation._execute_query(ROOT, "-literal-query", ROOT, 20)
        command = call.call_args.args[0]
        self.assertIn("--no-rerank", command)
        self.assertIn("--explain", command)
        self.assertNotIn("--allow-egress", command)
        self.assertNotIn("build", command)
        self.assertEqual(command[-2:], ["--", "-literal-query"])
        self.assertEqual(call.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")


class LabelSafetyTests(unittest.TestCase):
    def test_invalid_labels_are_refused(self) -> None:
        base = {"id": "one", "query": "research", "expected_pages": ["wiki/A.md"]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "queries.json"
            for updates in (
                {"id": "invalid id"}, {"query": ""}, {"query": "x" * 8001},
                {"expected_pages": None}, {"expected_pages": ["../A.md"]},
                {"expected_pages": ["/wiki/A.md"]}, {"expected_pages": ["wiki/../A.md"]},
                {"expected_pages": ["wiki/A.md", "wiki/A.md"]},
                {"expected_pages": [{"path": "wiki/A.md"}]}, {"skip_reason": ""},
                {"query": "invalid\ud800"}, {"expected_pages": ["wiki/invalid\ud800.md"]},
            ):
                labels(path, [{**base, **updates}])
                with self.subTest(updates=updates), self.assertRaises(EvaluationError):
                    load_queries(path)
            labels(path, [base, copy.deepcopy(base)])
            with self.assertRaises(EvaluationError):
                load_queries(path)
            path.write_text('{"schema":"' + QUERY_SCHEMA + '","queries":[],"queries":[]}', encoding="utf-8")
            with self.assertRaises(EvaluationError):
                load_queries(path)

    def test_labels_are_bounded_and_never_follow_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = labels(root / "queries.json", [])
            alias = root / "linked.json"
            alias.symlink_to(path)
            with self.assertRaises(EvaluationError):
                load_queries(alias)
            directory_alias = root / "linked-directory"
            directory_alias.symlink_to(root, target_is_directory=True)
            # A system link such as /home on Fedora Atomic is the usual cause,
            # so the refusal names the link and the path to pass instead.
            with self.assertRaisesRegex(EvaluationError, "pass the resolved path instead"):
                load_queries(directory_alias / "queries.json")
            path.write_bytes(b" " * (evaluation.MAX_QUERY_FILE_BYTES + 1))
            with self.assertRaisesRegex(EvaluationError, "limit"):
                load_queries(path)
            if hasattr(os, "mkfifo"):
                fifo = root / "queries.fifo"
                os.mkfifo(fifo)
                # An empty FIFO also fails JSON parsing, so require the
                # regular-file refusal itself rather than any EvaluationError.
                with self.assertRaisesRegex(EvaluationError, "expected a regular file"):
                    load_queries(fifo)

    def test_missing_index_has_exit_10(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            query_file = labels(root / "queries.json", [])
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/evaluate-retrieval.py"), "--vault", str(root), "--queries", str(query_file)],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 10, result.stderr)
            self.assertIn("RETRIEVAL_INDEX_UNAVAILABLE", result.stderr)
            self.assertFalse(result.stdout)


class FixtureAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="quality-test-")
        cls.root = Path(cls.temporary.name).resolve()
        cls.vault = cls.root / "fixture-vault"
        cls.query_file = provision_fixture(cls.vault, ROOT)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_original_fixture_reports_each_miss_and_preserves_all_bytes(self) -> None:
        before = snapshot(self.vault)
        labels_before = self.query_file.read_bytes()
        report = run_quality(self.vault, self.query_file, ROOT)
        self.assertEqual(snapshot(self.vault), before)
        self.assertEqual(self.query_file.read_bytes(), labels_before)
        self.assertEqual(report["query_count"], 14)
        self.assertEqual(report["scored_answerable_query_count"], 11)
        self.assertEqual(report["scored_unanswerable_query_count"], 3)
        self.assertEqual(report["skipped_query_count"], 0)
        self.assertEqual(report["metrics"]["recall_at_10"]["n"], 11)
        self.assertAlmostEqual(report["metrics"]["recall_at_1"]["value"], 9.5 / 11)
        self.assertAlmostEqual(report["metrics"]["recall_at_10"]["value"], 10 / 11)
        rows = {row["id"]: row for row in report["per_query"]}
        self.assertEqual(rows["semantic-synonym"]["missing_expected_pages"], ["wiki/concepts/Wellbeing.md"])
        self.assertEqual(rows["stale-historical"]["ranked_pages"], [])
        self.assertGreater(rows["stale-historical"]["explain"]["stale_candidates_skipped"], 0)
        self.assertEqual(report["metrics"]["unanswerable_no_result_accuracy"]["value"], 1)
        self.assertEqual(report["index_snapshot_sha256"], hashlib.sha256((self.vault / evaluation.INDEX_PATH).read_bytes()).hexdigest())
        self.assertGreater(report["latency_ms"]["p50"], 0)
        for chunk in (self.vault / ".vault-meta/chunks").rglob("chunk-*.json"):
            self.assertEqual(json.loads(chunk.read_text())["prefix_source"], "synthetic")

    def test_empty_query_set_and_missing_gold_report_no_data(self) -> None:
        path = labels(self.root / "empty.json", [])
        with mock.patch.object(evaluation, "_execute_query", side_effect=AssertionError("empty sets do not run retrieval")):
            report = run_quality(self.vault, path, ROOT)
        self.assertEqual(report["query_count"], 0)
        self.assertEqual(report["metrics"]["mrr"], {"value": None, "n": 0, "status": "no data", "cutoff": 10})
        labels(path, [{"id": "missing", "query": "absent", "expected_pages": ["wiki/Missing.md"]},
                      {"id": "skip", "query": "later", "expected_pages": [], "skip_reason": "explicit fixture exclusion"}])
        report = run_quality(self.vault, path, ROOT)
        self.assertEqual(report["missing_gold_page_count"], 1)
        self.assertEqual(report["missing_gold_query_count"], 1)
        self.assertEqual(report["skipped_query_count"], 2)
        self.assertEqual(report["metrics"]["recall_at_10"]["status"], "no data")

    def test_gold_symlinks_are_refused(self) -> None:
        alias = self.vault / "wiki/Alias.md"
        alias.symlink_to(self.vault / "wiki/concepts/Vault backup.md")
        try:
            path = labels(self.root / "symlink-gold.json", [{"id": "alias", "query": "backup", "expected_pages": ["wiki/Alias.md"]}])
            with self.assertRaises(EvaluationError):
                run_quality(self.vault, path, ROOT)
        finally:
            alias.unlink()

    def test_index_changes_and_corruption_are_refused(self) -> None:
        copy_vault = self.root / "copy-vault"
        shutil.copytree(self.vault, copy_vault)
        path = labels(self.root / "one.json", [{"id": "one", "query": "unrelated", "expected_pages": []}])
        index = copy_vault / evaluation.INDEX_PATH
        original = index.read_bytes()

        def mutate(*args, **kwargs):
            index.write_bytes(original + b"\n")
            return {"strategy": "bm25-only", "candidates": [], "explain": {}}, 1.0

        with mock.patch.object(evaluation, "_execute_query", side_effect=mutate):
            with self.assertRaisesRegex(EvaluationError, "changed during evaluation"):
                run_quality(copy_vault, path, ROOT)
        index.write_text('{"docs":{}}', encoding="utf-8")
        with self.assertRaises(evaluation.IndexNotReadyError):
            run_quality(copy_vault, path, ROOT)

    def test_query_label_mutation_is_refused(self) -> None:
        path = labels(self.root / "changing.json", [{"id": "one", "query": "unrelated", "expected_pages": []}])

        def mutate(*args, **kwargs):
            path.write_bytes(path.read_bytes() + b"\n")
            return {"strategy": "bm25-only", "candidates": [], "explain": {}}, 1.0

        with mock.patch.object(evaluation, "_execute_query", side_effect=mutate):
            with self.assertRaisesRegex(EvaluationError, "query labels changed"):
                run_quality(self.vault, path, ROOT)

    def test_existing_fixture_destination_is_never_overwritten(self) -> None:
        before = snapshot(self.vault)
        with self.assertRaisesRegex(EvaluationError, "existing vault is never overwritten"):
            provision_fixture(self.vault, ROOT)
        self.assertEqual(snapshot(self.vault), before)


if __name__ == "__main__":
    unittest.main()
