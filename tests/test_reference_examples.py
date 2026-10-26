#!/usr/bin/env python3
"""Exercise shipped reference examples without optional plugins or a network."""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from claude_obsidian.lint_engine import _parse_frontmatter, lint_vault
from claude_obsidian.page_schema import PAGE_TYPES, route_rejection


def blocks(path: str, language: str) -> list[str]:
    text = (ROOT / path).read_text(encoding="utf-8")
    return re.findall(r"(?m)^```" + re.escape(language) + r"\n(.*?)^```\s*$", text, re.S)


def state(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class ReferenceExampleTests(unittest.TestCase):
    def test_all_six_rendered_methodology_templates_pass_generated_properties(self) -> None:
        roles = {
            "lyt/atomic-template.md": ("concept", "atomic"),
            "lyt/moc-template.md": ("meta", "moc"),
            "para/area-template.md": ("concept", "area"),
            "para/project-template.md": ("entity", "project"),
            "para/resource-template.md": ("concept", "resource"),
            "zettel/atomic-template.md": ("concept", "zettel"),
        }
        template_root = ROOT / "skills/wiki-mode/templates"
        self.assertEqual(set(roles), {p.relative_to(template_root).as_posix()
                                    for p in template_root.glob("*/*.md")})
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            (vault / "wiki").mkdir()
            for relative, (kind, role) in roles.items():
                with self.subTest(template=relative):
                    text = (template_root / relative).read_text(encoding="utf-8")
                    rendered = re.sub(r"\{\{([^}]+)\}\}",
                                      lambda m: "2026-10-17" if m[1] == "date" else "Fixture " + m[1], text)
                    parsed = _parse_frontmatter(rendered)
                    self.assertEqual(kind, parsed.values["type"])
                    self.assertEqual(role, parsed.values["methodology_role"])
                    self.assertIn(kind, PAGE_TYPES)
                    if role != "moc":
                        self.assertIsNone(route_rejection(kind))
                    (vault / "wiki/index.md").write_text(rendered, encoding="utf-8")
                    before = state(vault)
                    report = lint_vault(vault, as_of=date(2026, 10, 17),
                                        frontmatter_policy="generated")
                    self.assertEqual([], report["invalid_frontmatter"], report)
                    self.assertEqual([], report["missing_frontmatter"], report)
                    self.assertEqual(before, state(vault))

    def test_shipped_markdown_properties_have_valid_tags_and_reject_numeric_tags(self) -> None:
        examples = blocks("skills/obsidian-markdown/SKILL.md", "yaml")
        self.assertTrue(examples)
        for example in examples:
            parsed = _parse_frontmatter(example)
            if not parsed.present:
                continue
            for tag in parsed.values.get("tags", []):
                self.assertIsInstance(tag, str)
                self.assertFalse(tag.isdecimal(), tag)
        self.assertTrue(any("retrieval" in _parse_frontmatter(example).values.get("tags", [])
                            for example in examples))
        # Quoting a year only affects YAML type; generated-note validation must
        # still distinguish an invalid numeric-only tag from semantic year tags.
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            (vault / "wiki").mkdir()
            valid = examples[0].replace("  - retrieval", "  - year/2026\n  - 知識")
            (vault / "wiki/index.md").write_text(valid, encoding="utf-8")
            report = lint_vault(vault, frontmatter_policy="generated")
            self.assertEqual([], report["invalid_frontmatter"], report)
            (vault / "wiki/index.md").write_text(valid.replace("year/2026", "2026"), encoding="utf-8")
            report = lint_vault(vault, frontmatter_policy="generated")
            self.assertTrue(any(item["field"] == "tags" for item in report["invalid_frontmatter"]))

    def test_native_base_embed_examples_resolve_to_shipped_view_names(self) -> None:
        examples = blocks("skills/obsidian-bases/SKILL.md", "yaml")
        views = [example for example in examples if re.search(r"(?m)^views:", example)]
        self.assertEqual(3, len(views))
        kinds: set[str] = set()
        with tempfile.TemporaryDirectory() as directory:
            vault = Path(directory)
            (vault / "wiki").mkdir()
            for example in views:
                names = re.findall(r'(?m)^    name:\s*"([^"]+)"$', example)
                self.assertTrue(names)
                kinds.update(re.findall(r"(?m)^  - type:\s*(\w+)$", example))
                (vault / "wiki/Dashboard.base").write_text(example, encoding="utf-8")
                content = "\n".join(f"![[Dashboard.base#{name}]]" for name in names)
                (vault / "wiki/index.md").write_text(content, encoding="utf-8")
                before = state(vault)
                report = lint_vault(vault)
                self.assertEqual([], report["dead_links"], report)
                self.assertEqual([], report["unsupported_fragments"], report)
                self.assertEqual(before, state(vault))
                references = set(re.findall(r"formula\.([A-Za-z_][A-Za-z_0-9]*)", example))
                definitions = set(re.findall(r"(?m)^  ([A-Za-z_][A-Za-z_0-9]*):", example))
                self.assertTrue(references <= definitions, references - definitions)
        self.assertEqual({"table", "cards", "list", "kanban"}, kinds)

    def test_dataview_example_declares_vault_root_scope_and_existing_properties(self) -> None:
        # This checks query structure/data compatibility, not the optional DQL
        # engine. Native rendering evidence is explicitly outside this suite.
        examples = blocks("skills/wiki/references/dataview.md", "dataview")
        self.assertEqual(1, len(examples))
        lines = [line.strip() for line in examples[0].splitlines() if line.strip()]
        self.assertEqual(["TABLE", "FROM", "WHERE", "SORT"],
                         [line.split(maxsplit=1)[0] for line in lines])
        columns = [column.strip() for column in lines[0].removeprefix("TABLE ").split(",")]
        source = shlex.split(lines[1])[1]
        predicate = shlex.split(lines[2])
        self.assertEqual("=", predicate[2])
        vault = ROOT / "examples/sample-vault"
        # The folder is relative to the vault root even if the dashboard is deep.
        source_path = vault / source
        self.assertEqual(vault / "wiki", source_path)
        records = [(p, _parse_frontmatter(p.read_text(encoding="utf-8")).values)
                   for p in source_path.rglob("*.md")]
        matching = [(p, fields) for p, fields in records
                    if fields.get(predicate[1]) == predicate[3]]
        self.assertEqual(["wiki/concepts/Source-grounded notes.md"],
                         sorted(p.relative_to(vault).as_posix() for p, _ in matching))
        for _, fields in matching:
            self.assertTrue(set(columns) <= fields.keys(), (columns, fields))
        self.assertEqual(["SORT", "file.name", "ASC"], shlex.split(lines[3]))

    def test_defuddle_shipped_external_plan_is_inert(self) -> None:
        snippets = blocks("skills/defuddle/SKILL.md", "bash")
        commands = []
        for snippet in snippets:
            for line in snippet.replace("\\\n", " ").splitlines():
                if line.startswith("python3 "):
                    commands.append(shlex.split(line))
        plans = [argv for argv in commands if argv[2:5] == ["capture", "external-plan", "url"]]
        self.assertEqual(1, len(plans))
        argv = [str(ROOT / "scripts/claude-obsidian.py") if token == "$CORE"
                else "https://example.com/reference" if token == "HTTPS_URL" else token
                for token in plans[0][1:]]
        result = subprocess.run([sys.executable, *argv], cwd=ROOT, check=True,
                                capture_output=True, text=True)
        plan = json.loads(result.stdout)
        self.assertFalse(plan["execute"])
        self.assertTrue(all(command[0] == "python3" for command in commands))
        self.assertTrue(all("defuddle" not in command[0] for command in commands))

    def test_manifest_pins_reviewed_sources_and_matches_actual_fallback_bytes(self) -> None:
        document = json.loads((ROOT / "config/upstream-references.json").read_text(encoding="utf-8"))
        self.assertEqual("claude-obsidian.upstream-references.v1", document["schema"])
        self.assertEqual(10, len(document["references"]))
        for reference in document["references"]:
            self.assertRegex(reference["revision"], r"^[0-9a-f]{40}$")
            date.fromisoformat(reference["checked_on"])
            self.assertTrue(reference["license"])
            self.assertTrue(reference["intentional_divergences"])
            self.assertEqual([], reference["native_versions_tested"])
            for source in reference["inspected_files"]:
                self.assertIn(reference["revision"], source["url"])
                if "sha256" in source:
                    self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
            for relative in reference["local_paths"]:
                self.assertTrue((ROOT / relative).is_file(), relative)
        for fallback in document["local_fallbacks"]:
            data = (ROOT / fallback["path"]).read_bytes()
            self.assertEqual(fallback["sha256"], hashlib.sha256(data).hexdigest(), fallback["path"])


if __name__ == "__main__":
    unittest.main()
