# Knowledge workflow acceptance

Storage validity and knowledge completeness are separate checks. A recoverable
transaction can omit the note's catalog entry or operation history. Save and
Ingest use the named workflow gate:

```bash
python3 "$CORE" workflow inspect "$BUNDLE" --vault "$VAULT"
# Read storage_valid, workflow_complete, findings, and approval_sha256.
python3 "$CORE" workflow apply "$BUNDLE" --vault "$VAULT" \
  --approved-plan-sha256 "$APPROVAL_SHA256"
```

Set `APPROVAL_SHA256` only from the reviewed inspect result. Generic
`transaction inspect/apply` remains the storage interface for other authorized
operations; it does not substitute for this workflow acceptance check.

Save and Ingest content operations couple the note/page writes, a hash-bound active
catalog write, a log entry, and hot context of at most 500 words and 32 KiB (the
SessionStart context bound). The default catalog is `wiki/index.md`. To declare
another active methodology MOC, place its exact vault-relative path in the bundle:

```json
{
  "workflow": {
    "catalogs": ["wiki/maps/Research.md"]
  }
}
```

This is a field inside a complete `claude-obsidian.transaction.v1` bundle, not a
standalone applyable document. Include expected hashes and writes for declared
catalogs. Every selected canonical content page, whether created or replaced,
requires a visible reference in the prospective content of a declared catalog
write. Use a path-qualified wikilink or relative Markdown link from that catalog
to each selected canonical page. An already indexed note can keep the catalog
bytes unchanged: supply its existing content as a `replace` write and bind its
current SHA-256 in `expected_hashes`. No meaningless catalog text edit is needed.
An unchanged catalog that omits a selected page fails the gate. `changed_paths`
lists transaction targets; its presence alone does not prove that bytes changed.
The catalog write and hash remain required for existing pages, keeping navigation
and the note update in the same approved, recoverable operation and replay.
Catalogs and selected content pages must use visible canonical paths: a
dot-prefixed directory or filename cannot satisfy this navigation obligation.
An active catalog is a catalog declared in `workflow.catalogs` that is either
`wiki/index.md` or already exists (a hash-bound `replace` write). A catalog that
the same bundle creates must be linked from one of those active catalog writes;
otherwise inspection reports `INACTIVE_CATALOG`, so a new throwaway page cannot
stand in for navigation. To link a new MOC from the root index, list
`wiki/index.md` beside the MOC so it is also declared in `workflow.catalogs`; an
undeclared index write does not count. The gate cannot judge whether an
existing declared page is a useful MOC.

Ingest includes both source and claim ledger writes; Save includes ledgers when
evidence records changed. Ingest also needs a source: a create-only `.raw/`
payload in the bundle, or a recorded source in the prospective source ledger
(`MISSING_INGEST_SOURCE`). At least one ledger must change; byte-identical
rewrites of both report `UNCHANGED_PROVENANCE`. The new first log entry, the
first `## ` heading of the log body, uses `## [YYYY-MM-DD] save|...` or
`## [YYYY-MM-DD] ingest|...`. It may sit directly below the log title and
introduction, as in the log that `init` seeds. Every existing entry stays
unchanged after the new one, and every other existing line keeps its order;
otherwise inspection reports `LOG_HISTORY_NOT_PRESERVED`. The date must be a
real calendar date, not only a matching digit pattern. Do not manufacture claims
or source records to satisfy a validator.

Workflow inspection materializes hash-bound UTF-8 `content_file` drafts. After
a successful apply, replay can use the committed target bytes when a disposable
draft no longer exists. For managed address assignment, it can reverse only
the exact generated second-line insertion, and only when those recovered bytes
match the declared original hash. The declared content hash, original materialized bundle,
approval and completed result still have to match. An available draft whose
bytes changed is refused, as are changed targets, modes or expected hashes.
Moving a draft with identical declared bytes does not change the materialized
operation. Completed replay also requires the retained journal and expanded
bundle: their identities, write coverage and content hashes must correlate
with the result under the transaction lock. A receipt alone cannot certify a
completed operation. Historical receipt-only records remain untouched during
unrelated operations, but cannot authorize replay. This equivalence does not
relax the transaction engine's replay or result-drift checks.

Malformed retained result JSON, invalid UTF-8, excessive JSON nesting, null or
incomplete result metadata, non-text modes or operation types, and malformed
replay writes return controlled transaction errors. A retained journal still in
the `prepared` state cannot certify completion. A FIFO or other non-regular
file at a receipt, log, or target path is refused without blocking. Completed
replay does not read the current `wiki/log.md`; the retained result binds it.
Reusing a completed operation ID with another operation type reports
`OPERATION_ID_REUSED`, as `transaction apply` does for another bundle.
Rejected replay does not authorize a new operation or modify the vault.
An existing log with invalid UTF-8 returns `WORKFLOW_LOG_NOT_TEXT` before
inspection can approve its replacement. Retain its original bytes and review
a separate repair rather than silently discarding unparseable history.

Lint and workflow navigation share protection for HTML and Obsidian `%%` comments,
matching-length inline code, backtick or tilde fences (closed by an equal or longer
run), indented code, escaped opening brackets, and `$` or `$$` math. Masking preserves
source offsets and line numbers. List continuation links remain visible when their
indentation is relative to a list item rather than a code block. A list-contained
fence ends when its list container ends, including an unclosed fence followed by
an outside paragraph or sibling item. Literal destination text is not parsed as
prose math or comments; enclosing code or comments still hide the whole link.
Inline Markdown links support single-line labels with balanced brackets, angled
destinations, balanced parentheses, and optional quoted or parenthesized titles.
Escapes and inline code inside a label do not create extra label brackets.
Decoded destinations still undergo vault confinement checks.
Heading labels render supported inline links outside code; requested fragments
name the resulting text and are not reinterpreted as new links. A literal
wikilink inside inline code therefore remains a distinct heading label.
This is a static navigation subset, not a full CommonMark or Obsidian renderer:
reference-style links, HTML anchors, blockquote container parsing, plugin syntax,
and native rendering are outside its coverage.

The deterministic gate checks declared structure and coupling. It cannot decide
whether prose is useful, a citation supports a claim, or an answer is complete.
Use the frozen Save/Ingest/Query scenarios under `tests/fixtures/forward/` for
fresh-context host acceptance. Retain host/model versions, prompts and outputs,
before/after hashes, operation receipts, reviewer findings, and exact missing
checks. Query must leave every vault byte unchanged and abstain when support is
missing. Test both a clean vault and a representative legacy copy.

Mocked/offline fixtures, actual model runs, native Obsidian rendering, and human
acceptance are distinct evidence classes. Do not promote one into another.
The [offline prompt evaluator](prompt-evaluation.md) checks explicit response
contracts and negative controls before any separately approved model trial;
it does not replace the disposable-vault workflow acceptance above.
