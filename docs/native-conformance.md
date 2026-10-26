# Native Obsidian conformance procedure

Offline parser checks do not establish application behavior. Use a disposable
fixture vault and a declared supported Obsidian version for native acceptance.
No real vault, plugin installation, or account change is needed for static work.
Do not install a runtime or plugin merely to make an unverified check look green.

Record app version, desktop/mobile platform, timezone, enabled core/community
plugins and versions, fixture hashes, date, and evidence class. Classify each
observation as source/static, official CLI, or actual application. Mark anything
not exercised as unverified. Retain screenshots or exact CLI output and a small
machine-readable result per case; do not infer a passing render from valid YAML.

## Original fixture cases

| Case | Fixture and expected observation to record |
|---|---|
| Markdown paths | Space and percent-encoded filenames, alias labels, duplicate basenames in different folders, same-note links, nested headings and block IDs. Record the resolved target or native unresolved state. |
| Protected regions | Indented code, nested fences, comments, escaped syntax, math, callouts and footnotes. Record which apparent links are actual links and preserve bytes during unrelated edits. |
| Properties | Flat typed properties, quoted wikilinks and link lists, semantic year tags and Unicode tags. Numeric-only values belong in a number property, not tags. |
| Embeds | Named `.base` view, PDF page/options, image size, and Canvas file nodes. Record named-view selection and render-time network behavior separately. |
| Bases | Table/cards on 1.9+, list on 1.10+, kanban on 1.14+, groupOrder absent/empty/explicit/null, and read-only formula grouping. Record result rows and group visibility. |
| Date formulas | Fixed one-day, multi-month, negative, missing-date and timezone cases. Compare subtraction/division, Duration day field, and explicit number conversion. Record actual result type/value and errors without choosing a winner in advance. |
| Optional syntax | Existing DQL, DataviewJS, Templater, Tasks recurrence/indentation and trailing spaces. Normal read/ingest/lint must not execute code; unrelated prose edits preserve those bytes. |
| CLI selection | Two open fixture vaults. Run the read-only allowlist through explicit selected-vault identity; prove the other vault is unchanged. |

Use original fixture text and synthetic records. Do not copy a third party's test
corpus without its license and attribution. A portable lint result is compared
to the recorded native target graph only for the supported subset. Keep ambiguous
or unsupported cases visible. Native app observations do not establish Windows
write support, cloud-sync convergence, provider quality, or every plugin version.

When a discrepancy appears, retain the failing case, app evidence, exact parser
result and intentional divergence before changing an implementation or source
pin. Never update expected fixtures automatically on a normal test run.
