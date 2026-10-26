"""Offline prompt acceptance packets, response checks, and negative controls."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claude_obsidian.prompt_eval import (
    PACKET_SCHEMA,
    REPORT_SCHEMA,
    RESPONSE_SCHEMA,
    PromptEvaluationError,
    evaluate_response,
    generate_packet,
)


FIXTURE = ROOT / "tests/fixtures/prompt-quality"
SCRIPT = ROOT / "scripts/evaluate-prompts.py"


def response_for(packet: dict) -> dict:
    value = json.loads((FIXTURE / "good-response.json").read_text(encoding="utf-8"))
    value["binding_sha256"] = packet["binding_sha256"]
    return value


def check(report: dict, identifier: str) -> bool:
    return next(
        item["passed"]
        for rows in report["checks"].values()
        for item in rows
        if item["id"] == identifier
    )


class PromptEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = generate_packet(ROOT)
        self.response = response_for(self.packet)

    def test_packet_is_deterministic_and_bound_to_current_public_sources(self) -> None:
        self.assertEqual(self.packet, generate_packet(ROOT))
        self.assertEqual(PACKET_SCHEMA, self.packet["schema"])
        self.assertEqual("offline response-contract fixture", self.packet["evidence_class"])
        self.assertEqual(
            [
                "skills/wiki-query/SKILL.md",
                "skills/save/SKILL.md",
                "skills/wiki/references/provenance.md",
            ],
            [source["path"] for source in self.packet["sources"]],
        )
        for source in self.packet["sources"]:
            expected = hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
            self.assertEqual(expected, source["source_sha256"])
        reference = self.packet["sources"][2]
        self.assertEqual("claim-rules", reference["selection"])
        self.assertTrue(reference["content"].startswith("## Claim rules\n"))
        self.assertNotIn("## Migration", reference["content"])
        serialized = json.dumps(self.packet, ensure_ascii=False)
        self.assertNotIn(str(ROOT), serialized)
        self.assertNotIn(os.path.expanduser("~"), serialized)
        self.assertNotIn("candidate-prompt-payload", serialized)
        contract = json.dumps(self.packet["response_contract"], ensure_ascii=False)
        for answer_value in ("09:30", "17:00", "18:00", "tentative", "contested"):
            self.assertNotIn(answer_value, contract)

    def test_good_response_passes_all_bounded_checks(self) -> None:
        report = evaluate_response(self.packet, self.response, product_root=ROOT)
        self.assertEqual(REPORT_SCHEMA, report["schema"])
        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(22, report["summary"]["checks_total"])
        self.assertEqual(22, report["summary"]["checks_passed"])
        self.assertEqual(
            "offline response-contract validation", report["evidence_class"]
        )
        self.assertEqual(
            hashlib.sha256((ROOT / "claude_obsidian/prompt_eval.py").read_bytes()).hexdigest(),
            report["provenance"]["evaluator_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(
                json.dumps(
                    self.response,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            report["provenance"]["response_sha256"],
        )
        self.assertFalse(report["provider_execution"])
        self.assertFalse(report["vault_mutation"])
        self.assertIn(
            "A handcrafted response does not establish prompt reasoning or model quality.",
            report["limitations"],
        )
        self.assertIn(
            "A passing report validates only a supplied structured response against bounded schema and exact-value checks.",
            report["limitations"],
        )
        self.assertIn("Human semantic review remains required.", report["limitations"])

    def test_response_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            target = temporary / "target.json"
            target.write_text(json.dumps(self.response), encoding="utf-8")
            link = temporary / "response.json"
            link.symlink_to(target)
            with self.assertRaises(PromptEvaluationError):
                evaluate_response(self.packet, link, product_root=ROOT)

    def test_response_path_with_symlinked_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            actual = temporary / "actual"
            actual.mkdir()
            response_path = actual / "response.json"
            response_path.write_text(json.dumps(self.response), encoding="utf-8")
            alias = temporary / "alias"
            alias.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(PromptEvaluationError, "pass the resolved path instead"):
                evaluate_response(
                    self.packet, alias / "response.json", product_root=ROOT
                )

    def test_non_regular_response_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            candidates = [temporary / "response-directory"]
            candidates[0].mkdir()
            if hasattr(os, "mkfifo"):
                candidates.append(temporary / "response.fifo")
                os.mkfifo(candidates[-1])
            for candidate in candidates:
                with self.subTest(path=candidate.name):
                    with self.assertRaisesRegex(
                        PromptEvaluationError,
                        "response must be a regular non-symlink file",
                    ):
                        evaluate_response(self.packet, candidate, product_root=ROOT)

    def test_invalid_utf8_source_is_mapped_to_evaluation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            for relative in (
                "skills/wiki-query/SKILL.md",
                "skills/save/SKILL.md",
                "tests/fixtures/prompt-quality/cases.json",
            ):
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            provenance = temporary / "skills/wiki/references/provenance.md"
            provenance.parent.mkdir(parents=True, exist_ok=True)
            provenance.write_bytes(b"\xff")
            with self.assertRaisesRegex(PromptEvaluationError, "must be UTF-8"):
                generate_packet(temporary)

    def test_old_false_positive_shape_is_rejected(self) -> None:
        old = {
            "known": {
                "answer": "The room definitely opens at 09:30.",
                "citations": ["[[Juniper#Hours]]"],
            },
            "conflict": {
                "answer": "17:00 and 18:00",
                "citations": ["[[Wrong A]]", "[[Wrong B]]"],
            },
            "absent": {"answer": "No data", "answer_available": False},
            "save": {
                "proposed_title": "Review decision",
                "ask_before_drafting": False,
                "apply_without_review": False,
            },
        }
        with self.assertRaisesRegex(PromptEvaluationError, "incorrect fields"):
            evaluate_response(self.packet, old, product_root=ROOT)

    def test_council_contradictory_free_prose_response_is_rejected(self) -> None:
        response = {
            "schema": RESPONSE_SCHEMA,
            "binding_sha256": self.packet["binding_sha256"],
            "cases": {
                "known": {
                    "answer": "The supplied evidence does not say the room opens at 09:30.",
                    "citations": ["[[Juniper#Hours]]"],
                    "status": "tentative",
                },
                "conflict": {
                    "answer": "The sources conflict, but neither says 17:00 or 18:00.",
                    "citations": ["[[Cedar A#Hours]]", "[[Cedar B#Hours]]"],
                    "status": "contested",
                },
                "absent": {
                    "answer": "The missing evidence means the telephone is five five five one two three four.",
                    "answer_available": False,
                    "citations": [],
                    "status": "unsupported",
                },
                "save": copy.deepcopy(self.response["cases"]["save"]),
            },
        }
        with self.assertRaisesRegex(PromptEvaluationError, "incorrect fields"):
            evaluate_response(self.packet, response, product_root=ROOT)

    def test_wrong_conflict_citations_fail_exact_source_check(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["conflict"]["citations"] = ["[[Wrong A]]", "[[Wrong B]]"]
        report = evaluate_response(self.packet, response, product_root=ROOT)
        self.assertFalse(report["summary"]["passed"])
        self.assertFalse(check(report, "conflict-citations"))

    def test_missing_save_body_is_rejected_as_incomplete(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["save"].pop("note_body")
        with self.assertRaisesRegex(PromptEvaluationError, "missing=.*note_body"):
            evaluate_response(self.packet, response, product_root=ROOT)

    def test_invented_save_date_fails_body_and_scope_checks(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["save"]["note_body"] = "We will review the catalogue on Friday, 2026-10-23."
        report = evaluate_response(self.packet, response, product_root=ROOT)
        self.assertFalse(check(report, "save-no-invented-date"))
        self.assertFalse(check(report, "save-exact-scope"))

    def test_invented_title_dates_fail_for_all_accepted_title_forms(self) -> None:
        for title in (
            "Review on October 23, 2026",
            "Review on 23/10/2026",
            "Review on October twenty third",
            "Review on 2026-10-23",
        ):
            with self.subTest(title=title):
                response = copy.deepcopy(self.response)
                response["cases"]["save"]["proposed_title"] = title
                report = evaluate_response(self.packet, response, product_root=ROOT)
                self.assertFalse(report["summary"]["passed"])
                self.assertFalse(check(report, "save-no-invented-date"))

    def test_title_contract_rejects_huge_multiline_or_unselected_titles(self) -> None:
        for title in (
            "x" * 32000,
            "Friday\nCatalogue review",
            "Friday\u2028Catalogue review",
            "Review " * 9,
            "Unselected decision",
        ):
            with self.subTest(title=title[:80]):
                response = copy.deepcopy(self.response)
                response["cases"]["save"]["proposed_title"] = title
                report = evaluate_response(self.packet, response, product_root=ROOT)
                self.assertFalse(report["summary"]["passed"])
                self.assertFalse(check(report, "save-title"))

    def test_grounded_short_title_variants_pass_the_explicit_fixture_contract(self) -> None:
        for title in ("Friday catalogue review", "Catalogue review", "Review on Friday"):
            with self.subTest(title=title):
                response = copy.deepcopy(self.response)
                response["cases"]["save"]["proposed_title"] = title
                report = evaluate_response(self.packet, response, product_root=ROOT)
                self.assertTrue(report["summary"]["passed"])

    def test_exact_save_body_rejects_added_surrounding_whitespace(self) -> None:
        selected = self.packet["cases"]["save"]["selected_content"]
        for body in (" " + selected, selected + "\n", "\t\n" + selected + " \n"):
            with self.subTest(body=repr(body)):
                response = copy.deepcopy(self.response)
                response["cases"]["save"]["note_body"] = body
                report = evaluate_response(self.packet, response, product_root=ROOT)
                self.assertFalse(report["summary"]["passed"])
                self.assertFalse(check(report, "save-exact-scope"))
                self.assertFalse(check(report, "save-no-invented-date"))

    def test_stronger_status_for_provisional_fact_fails(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["known"]["status"] = "established"
        report = evaluate_response(self.packet, response, product_root=ROOT)
        self.assertFalse(check(report, "known-status"))

    def test_wrong_known_time_fails_exact_value_check(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["known"]["opens_at"] = "10:00"
        report = evaluate_response(self.packet, response, product_root=ROOT)
        self.assertFalse(check(report, "known-opens-at"))

    def test_negated_known_claim_field_fails_exact_value_check(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["known"]["opens_at"] = "not 09:30"
        report = evaluate_response(self.packet, response, product_root=ROOT)
        self.assertFalse(check(report, "known-opens-at"))

    def test_invented_absent_value_fails_null_check(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["absent"]["telephone"] = "five five five one two three four"
        report = evaluate_response(self.packet, response, product_root=ROOT)
        self.assertFalse(check(report, "absent-telephone"))

    def test_injected_conflict_value_fails_exact_checks(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["conflict"]["closing_times"].append("23:00")
        report = evaluate_response(self.packet, response, product_root=ROOT)
        self.assertFalse(check(report, "conflict-closing-times"))
        self.assertFalse(check(report, "injected-answer-ignored"))

    def test_extra_query_prose_field_is_rejected(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["known"]["answer"] = "The room opens at 09:30."
        with self.assertRaisesRegex(PromptEvaluationError, "extra=.*answer"):
            evaluate_response(self.packet, response, product_root=ROOT)

    def test_non_text_statuses_are_controlled_schema_errors(self) -> None:
        for invalid in (None, [], {}):
            with self.subTest(invalid=invalid):
                response = copy.deepcopy(self.response)
                response["cases"]["known"]["status"] = invalid
                with self.assertRaisesRegex(PromptEvaluationError, "response enum"):
                    evaluate_response(self.packet, response, product_root=ROOT)

    def test_lone_unicode_surrogate_is_a_controlled_schema_error(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["known"]["opens_at"] = "09:30 \ud800"
        with self.assertRaisesRegex(PromptEvaluationError, "unsupported"):
            evaluate_response(self.packet, response, product_root=ROOT)

    def test_non_json_response_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "response.json"
            path.write_text("This is not JSON.\n", encoding="utf-8")
            with self.assertRaisesRegex(PromptEvaluationError, "strict UTF-8 JSON"):
                evaluate_response(self.packet, path, product_root=ROOT)

    def test_refusal_object_is_rejected_as_incomplete(self) -> None:
        refusal = {
            "schema": RESPONSE_SCHEMA,
            "binding_sha256": self.packet["binding_sha256"],
            "refusal": "I cannot answer.",
        }
        with self.assertRaisesRegex(PromptEvaluationError, "incorrect fields"):
            evaluate_response(self.packet, refusal, product_root=ROOT)

    def test_incomplete_case_set_is_rejected(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"].pop("absent")
        with self.assertRaisesRegex(PromptEvaluationError, "incorrect fields"):
            evaluate_response(self.packet, response, product_root=ROOT)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        raw = json.dumps(self.response)
        raw = raw.replace(
            '"schema": "claude-obsidian.prompt-response.v1"',
            '"schema": "claude-obsidian.prompt-response.v1", "schema": "claude-obsidian.prompt-response.v1"',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "duplicate.json"
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(PromptEvaluationError, "duplicate JSON object key"):
                evaluate_response(self.packet, path, product_root=ROOT)

    def test_booleans_are_strict_and_do_not_accept_integers(self) -> None:
        response = copy.deepcopy(self.response)
        response["cases"]["save"]["review_required"] = 1
        with self.assertRaisesRegex(PromptEvaluationError, "JSON boolean"):
            evaluate_response(self.packet, response, product_root=ROOT)

    def test_tampered_packet_content_is_rejected(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["sources"][0]["content"] += "\nTampered.\n"
        with self.assertRaisesRegex(PromptEvaluationError, "selection hash differs"):
            evaluate_response(packet, self.response, product_root=ROOT)

    def test_rebound_nonfixture_cases_are_rejected(self) -> None:
        packet = copy.deepcopy(self.packet)
        packet["cases"]["known"]["source"]["text"] = "The room opens at 10:00."
        canonical = json.dumps(
            packet["cases"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        packet["cases_sha256"] = hashlib.sha256(canonical).hexdigest()
        material = {
            "cases_sha256": packet["cases_sha256"],
            "sources": [
                {
                    "path": source["path"],
                    "source_sha256": source["source_sha256"],
                    "selection": source["selection"],
                    "selection_sha256": source["selection_sha256"],
                }
                for source in packet["sources"]
            ],
        }
        packet["binding_sha256"] = hashlib.sha256(
            json.dumps(
                material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        response = copy.deepcopy(self.response)
        response["binding_sha256"] = packet["binding_sha256"]
        with self.assertRaisesRegex(PromptEvaluationError, "frozen public fixture"):
            evaluate_response(packet, response, product_root=ROOT)

    def test_cli_help_packet_and_exit_codes(self) -> None:
        help_result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(0, help_result.returncode, help_result.stderr)
        self.assertIn("packet", help_result.stdout)
        self.assertIn("evaluate", help_result.stdout)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            packet_path = temporary / "packet.json"
            packet_result = subprocess.run(
                [sys.executable, str(SCRIPT), "packet"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(0, packet_result.returncode, packet_result.stderr)
            packet_path.write_text(packet_result.stdout, encoding="utf-8")
            packet = json.loads(packet_result.stdout)

            response_path = temporary / "response.json"
            response = response_for(packet)
            response_path.write_text(json.dumps(response), encoding="utf-8")
            passed = subprocess.run(
                [sys.executable, str(SCRIPT), "evaluate", str(response_path), "--packet", str(packet_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(0, passed.returncode, passed.stderr)
            self.assertTrue(json.loads(passed.stdout)["summary"]["passed"])

            response["cases"]["conflict"]["citations"] = ["[[Wrong A]]", "[[Wrong B]]"]
            response_path.write_text(json.dumps(response), encoding="utf-8")
            failed = subprocess.run(
                [sys.executable, str(SCRIPT), "evaluate", str(response_path), "--packet", str(packet_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(1, failed.returncode, failed.stderr)
            self.assertFalse(json.loads(failed.stdout)["summary"]["passed"])

            response_path.write_text("not JSON\n", encoding="utf-8")
            invalid = subprocess.run(
                [sys.executable, str(SCRIPT), "evaluate", str(response_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(2, invalid.returncode)
            self.assertIn("PROMPT_EVALUATION_ERROR", invalid.stderr)

            response = response_for(packet)
            response["cases"]["known"]["opens_at"] = "09:30 \ud800"
            response_path.write_text(json.dumps(response), encoding="utf-8")
            invalid_scalar = subprocess.run(
                [sys.executable, str(SCRIPT), "evaluate", str(response_path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(2, invalid_scalar.returncode)
            self.assertIn("PROMPT_EVALUATION_ERROR", invalid_scalar.stderr)
            self.assertNotIn("Traceback", invalid_scalar.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
