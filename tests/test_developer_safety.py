#!/usr/bin/env python3
"""Readiness and developer commands must preserve recoverable vault state."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claude_obsidian import cli
from claude_obsidian.paths import VaultSelection
from claude_obsidian.transaction import (
    BUNDLE_SCHEMA,
    MutationLock,
    inspect_bundle,
    recover_incomplete,
    sha256_bytes,
)


def make_vault(root: Path) -> Path:
    root.mkdir()
    for name in (".obsidian", "wiki", ".raw"):
        (root / name).mkdir()
    return root


def snapshot(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): (
            path.lstat().st_mode,
            os.readlink(path) if path.is_symlink() else
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        )
        for path in root.rglob("*")
    }


class DoctorReadinessTests(unittest.TestCase):
    def doctor(self, root: Path, access=None):
        output = []
        with mock.patch.object(cli, "_selection", return_value=VaultSelection(root, "explicit")), \
                mock.patch.object(cli, "_emit", side_effect=output.append):
            if access is None:
                code = cli.command_doctor(SimpleNamespace())
            else:
                with mock.patch.object(cli.os, "access", side_effect=access):
                    code = cli.command_doctor(SimpleNamespace())
        return code, output[0]

    def test_unsafe_metadata_is_unhealthy_without_writes(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            outside = base / "outside"
            (outside / "mutation.lock").mkdir(parents=True)
            for kind in ("file", "symlink", "broken-symlink"):
                with self.subTest(kind=kind):
                    vault = make_vault(base / kind)
                    meta = vault / ".vault-meta"
                    if kind == "file":
                        meta.write_text("not a directory\n", encoding="utf-8")
                    else:
                        meta.symlink_to(outside if kind == "symlink" else base / "missing", target_is_directory=True)
                    before = snapshot(base)
                    code, report = self.doctor(vault)
                    self.assertEqual(1, code)
                    self.assertFalse(report["ok"])
                    self.assertFalse(report["checks"]["meta_writable"])
                    self.assertFalse(report["checks"]["mutation_lock_held"])
                    self.assertEqual(before, snapshot(base))

    def test_unsafe_wiki_and_runtime_entries_are_unhealthy_without_writes(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            outside = base / "outside"
            (outside / "mutation.lock").mkdir(parents=True)
            cases = ("wiki-symlink", "wiki-file", "transactions-symlink", "transactions-file", "lock-symlink")
            for kind in cases:
                with self.subTest(kind=kind):
                    vault = make_vault(base / kind)
                    meta = vault / ".vault-meta"
                    meta.mkdir()
                    if kind == "wiki-symlink":
                        (vault / "wiki").rmdir()
                        (vault / "wiki").symlink_to(outside, target_is_directory=True)
                    elif kind == "wiki-file":
                        (vault / "wiki").rmdir()
                        (vault / "wiki").write_text("not a directory\n", encoding="utf-8")
                    elif kind == "transactions-symlink":
                        (meta / "transactions").symlink_to(outside, target_is_directory=True)
                    elif kind == "transactions-file":
                        (meta / "transactions").write_text("not a directory\n", encoding="utf-8")
                    else:
                        (meta / "mutation.lock").symlink_to(outside / "mutation.lock", target_is_directory=True)
                    before = snapshot(base)
                    code, report = self.doctor(vault)
                    self.assertEqual(1, code, report)
                    self.assertFalse(report["ok"])
                    self.assertFalse(report["checks"]["mutation_lock_held"])
                    if kind.startswith("wiki"):
                        self.assertFalse(report["checks"]["wiki"])
                        self.assertTrue(report["checks"]["runtime_state_safe"])
                    else:
                        self.assertTrue(report["checks"]["wiki"])
                        self.assertFalse(report["checks"]["runtime_state_safe"])
                    self.assertEqual(before, snapshot(base))
            vault = make_vault(base / "held")
            (vault / ".vault-meta/mutation.lock").mkdir(parents=True)
            code, report = self.doctor(vault)
            self.assertEqual(0, code, report)
            self.assertTrue(report["checks"]["mutation_lock_held"])
            self.assertTrue(report["checks"]["runtime_state_safe"])

    def test_write_and_search_access_are_both_required(self):
        with tempfile.TemporaryDirectory() as name:
            vault = make_vault(Path(name) / "vault")
            meta = vault / ".vault-meta"
            for existing in (False, True):
                if existing:
                    meta.mkdir()
                before = snapshot(vault)
                for denied in (os.W_OK, os.X_OK):
                    with self.subTest(existing=existing, denied=denied):
                        seen = []
                        def access(path, mode):
                            seen.append((Path(path), mode))
                            return not mode & denied
                        code, report = self.doctor(vault, access)
                        self.assertEqual(1, code)
                        self.assertFalse(report["checks"]["meta_writable"])
                        self.assertEqual([(meta if existing else vault, os.W_OK | os.X_OK)], seen)
                        self.assertEqual(before, snapshot(vault))

    def test_safe_existing_or_creatable_metadata_preserves_schema(self):
        with tempfile.TemporaryDirectory() as name:
            vault = make_vault(Path(name) / "vault")
            for existing in (False, True):
                if existing:
                    (vault / ".vault-meta").mkdir()
                before = snapshot(vault)
                code, report = self.doctor(vault)
                self.assertEqual(0, code)
                self.assertTrue(report["ok"])
                self.assertEqual("claude-obsidian.doctor.v1", report["schema"])
                self.assertEqual({"vault_exists", "obsidian_config", "wiki", "raw", "meta_writable", "runtime_state_safe", "mutation_lock_held"}, set(report["checks"]))
                self.assertEqual(before, snapshot(vault))


@unittest.skipUnless(shutil.which("make"), "make is required")
class DeveloperRecoveryTests(unittest.TestCase):
    def test_test_cleanup_cannot_erase_an_interrupted_transaction(self):
        with tempfile.TemporaryDirectory() as name:
            vault = make_vault(Path(name) / "vault")
            for leaf in ("A", "B"):
                (vault / f"wiki/{leaf}.md").write_text(f"old {leaf}\n", encoding="utf-8")
            bundle = {
                "schema": BUNDLE_SCHEMA,
                "operation_id": "developer-cleanup-recovery",
                "operation_type": "generic",
                "expected_hashes": {f"wiki/{leaf}.md": sha256_bytes((vault / f"wiki/{leaf}.md").read_bytes()) for leaf in ("A", "B")},
                "writes": [{"path": f"wiki/{leaf}.md", "mode": "replace", "content": f"new {leaf}\n"} for leaf in ("A", "B")],
            }
            approval = inspect_bundle(vault, bundle)["approval_sha256"]
            program = """import json, os, sys
from pathlib import Path
from claude_obsidian.transaction import apply_bundle
def interrupt(path, index):
    if index == 1:
        os._exit(91)
apply_bundle(Path(sys.argv[1]), json.loads(sys.argv[2]), approved_plan_sha256=sys.argv[3], progress=interrupt)
"""
            child = subprocess.run([sys.executable, "-B", "-c", program, str(vault), json.dumps(bundle), approval], cwd=ROOT, capture_output=True, text=True, timeout=15)
            self.assertEqual(91, child.returncode, child.stderr)
            self.assertEqual("new A\n", (vault / "wiki/A.md").read_text())
            self.assertEqual("old B\n", (vault / "wiki/B.md").read_text())
            before = snapshot(vault)
            cleanup = subprocess.run(["make", "-f", str(ROOT / "Makefile"), "clean-test-state"], cwd=vault, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(0, cleanup.returncode)
            self.assertEqual(before, snapshot(vault))
            with MutationLock(vault, timeout=2, force_stale_lock=True) as lock:
                self.assertEqual([bundle["operation_id"]], recover_incomplete(vault, mutation_lock=lock))
            for leaf in ("A", "B"):
                self.assertEqual(f"old {leaf}\n", (vault / f"wiki/{leaf}.md").read_text())


if __name__ == "__main__":
    unittest.main()
