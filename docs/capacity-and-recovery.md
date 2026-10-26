# Capacity and recovery

Inspect the selected vault before a large operation:

```bash
python3 "$CORE" capacity --vault "$VAULT"
```

This reads status without changing the vault. Keep ledger bytes, raw archive
bytes, page count, and retrieval chunk count distinct when measuring capacity.
No page-count promise or constant latency follows from a passing small fixture.
Benchmark representative small, medium, and large copies on the target machine;
retain elapsed time, memory, index size, cancellation outcome, and recovery time.

Canonical source and claim ledgers must fit the 16 MiB reader/writer limit.
Transactions preflight them and reject growth with `LEDGER_CAPACITY_EXCEEDED`.
Do not remove evidence, lower fidelity, or disable validation to get under the
limit. A formatting-heavy legacy ledger may be compacted losslessly:

```bash
python3 "$CORE" capacity compact --vault "$VAULT" \
  --operation-id capacity-reviewed
# Set APPROVAL_SHA256 to the reviewed approved_plan_sha256.
python3 "$CORE" capacity compact --vault "$VAULT" \
  --operation-id capacity-reviewed \
  --approved-plan-sha256 "$APPROVAL_SHA256" --apply
```

Keep a complete recoverable vault backup before repair. Review the operation's
paths and hashes. This repair reads legacy inputs only up to 64 MiB and changes
JSON formatting, preserving every record and source identity. It rewrites only
ledgers that `capacity` reports as `near_capacity` (at least 85% of the limit)
or `over_capacity`; smaller ledgers are left byte-for-byte unchanged and a vault
with none is a `noop`. If the compact canonical representation still exceeds
16 MiB, it fails with `LEDGER_MIGRATION_REQUIRED`, refusing early when even
removing all whitespace could not fit. Stop and retain the data; automated
sharding and record eviction are not implemented. An arbitrary size increase
would need reader, writer, journal, memory, and rollback evidence together.

The compaction is an ordinary transaction, so inspection runs the same semantic
provenance validation as any ledger write. If the vault drifted out of band,
for example a `.raw/` source edited after ingest so its recorded
`content_sha256` no longer matches, inspection refuses the plan with
`INVALID_PROVENANCE_LEDGER` even though only formatting would change. Do not
bypass validation. Recover through a normal reviewed transaction that writes a
corrected ledger in compact form (one JSON line with no insignificant
whitespace), with the expected hash of the current ledger and an approved plan.

Writers determine how long the savings last. `migration_bundle` (`migrate`)
now writes canonical ledgers in that compact form. The `init`/`adopt` planner
in `claude_obsidian/vault_ops.py` and hand-written bundles may still write
indented JSON; that is valid, and compaction becomes relevant only once such a
ledger approaches the limit.

One content/predecessor file is bounded at 64 MiB; one operation's new content or
recovery backups at 128 MiB; writes at 1,024. These storage ceilings are distinct
from stricter ledger limits and from practical operating capacity. Split large
work at a logical boundary and inspect each bounded operation. Never split a
single coupled workflow into unreviewed per-file applies.

For an interrupted operation, stop competing writers and retain the selected
vault with its `.vault-meta/` journals and backups. Then use:

```bash
python3 "$CORE" transaction recover --vault "$VAULT"
```

Recovery may roll back an incomplete apply. Do not delete journals to hide an
error or overwrite changed files by hand. Exit 75 requires a fresh read and plan.
If an existing replacement target was deleted after an interrupted apply,
recovery leaves it absent, retains the journal as `rollback-failed`, and returns
`ROLLBACK_FAILED`. Review the external deletion and retained backup before
authorizing a separate restoration; recovery does not silently recreate it.
Use `--force-stale-lock` only after independently establishing that no writer
is active. The capture queue has a separate explicit recovery interface; it is
not a way to bypass the transaction lock. See
[coordination.md](../skills/wiki/references/coordination.md) for external edits
and multi-machine limits.
