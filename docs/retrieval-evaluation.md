# Offline retrieval evaluation

Use explicit relevance labels to measure a prebuilt local BM25 index. Evaluation
reads selected pages and caches without rebuilding them. It always passes
`--no-rerank`, makes no model or provider calls, and never provisions a user
vault. Reported results describe the selected snapshot and query set.

Use physical paths for query-label files. The reader rejects symlinks in every
path component, including macOS aliases such as `/tmp` and `/var`. Resolve the
trusted containing directory before creating ordinary label files there;
explicit symlink inputs remain invalid.

```bash
python3 scripts/evaluate-retrieval.py --vault /absolute/vault \
  --queries /absolute/queries.json
```

The core API is `run_quality(vault, query_file, product_root, top_k=10)`. The
portable CLI also exposes the same evaluation through `bench --queries FILE
--vault VAULT`. Both require an existing index. Missing or corrupt derived
state exits 10 with a rebuild diagnostic. Invalid labels, unsafe paths, or a
changing snapshot exit 2. Rebuilding an index is a separate explicit operation.

## Write useful labels

Each query needs an ID, exact query text, and an explicit list of relevant
vault-relative Markdown page paths. Empty lists mean that no answer is
expected from this corpus. They are evaluated as negative cases; they never
inflate recall or MRR by dividing by zero or counting a free success.

```json
{
  "schema": "claude-obsidian.retrieval-queries.v1",
  "queries": [
    {
      "id": "journal-recovery",
      "query": "recover interrupted writes",
      "expected_pages": ["wiki/concepts/Recovery.md"],
      "category": "recovery"
    },
    {
      "id": "unanswerable",
      "query": "a topic absent from the vault",
      "expected_pages": [],
      "category": "negative"
    }
  ]
}
```

Write labels from the information need before tuning retrieval. Include exact
identifiers, natural-language formulations, synonyms, Unicode names, several
relevant pages, stale material, conflicting records, and genuinely absent
topics. Keep a separate holdout set for future changes. Do not rewrite labels
to fit the candidate ranking.

An optional `skip_reason` marks an explicit exclusion. The report records it.
A query with a missing gold page is also excluded from aggregate metrics and
listed with its missing paths. Unsafe symlinks, junctions, traversal, absolute
paths, duplicate labels, and nonregular files fail closed. Label files are
bounded at 1 MiB, query sets at 1,000 entries, each query at 8,000 normalized
characters, and each gold set at 64 pages. Reads use no-follow descriptors on
platforms that support them and reject link components on other platforms.

## Interpret the report

Recall at 1, 5, and 10 is the fraction of a query's gold pages present in the
first 1, 5, or 10 distinct returned pages. The aggregate is the mean across
scored queries with non-empty gold lists. A query with two relevant pages and
one correct first result therefore has recall at 1 of 0.5.

MRR is the mean reciprocal rank of the first relevant page. A first relevant
page at rank 2 contributes 0.5; an absent relevant page contributes 0. The
default cutoff is 10. `--top` can increase the measured depth; the minimum
depth remains 10 so all three recall cutoffs have comparable observations.

Every metric includes its own `n` and status. `n = 0` is rendered as `no data`
with a null value. Empty-gold queries have a separate no-result accuracy with
their own denominator. Missing-gold and explicitly skipped queries have
separate counts. Per-query rows expose rankings, misses at each cutoff,
reciprocal rank, process latency, and retrieval diagnostics.

The candidate pool is at least 20 BM25 chunks and grows with requested depth.
Usable-candidate recall is reported separately after stale filtering and page
deduplication. Raw BM25 candidate recall is unavailable because `retrieve.py
--explain` exposes raw candidate counts without their page identities. This
scope is recorded directly in the report. Latency includes process startup,
index loading, and query work. P50 and P95 use nearest-rank percentiles.

Exact label and index byte hashes bind a run to its inputs. Changes to either
during evaluation invalidate the run. Page hash checking in retrieval rejects
stale chunks. A benchmark does not hold the vault-wide mutation lock; run it
while other writers are idle. The report is returned to stdout and can be
saved outside the user vault when a durable comparison is needed.

Malformed or undecodable chunk records are counted as skipped stale candidates;
healthy candidates remain available. Page freshness requires strict UTF-8, so
corrupt page bytes cannot match a cached literal replacement character through
lossy decoding. Rejected chunks and pages are not repaired during a query.

## Run the original fixture

The source checkout includes 14 original labeled queries over 10 invented
pages. It contains French, Japanese, Romanian, exact-address lookup,
multiple-gold retrieval, an intentional lexical synonym miss, stale conflicting
history, and explicit unanswerable cases.

```bash
python3 scripts/evaluate-retrieval.py --fixture
python3 tests/test_retrieval_eval.py
```

Fixture mode creates only a temporary directory, copies the invented pages,
runs `contextual-prefix.py --all --no-llm`, builds BM25, and changes one fixture
page after indexing to test stale-cache rejection. It then runs the same
read-only evaluator and deletes the temporary directory. It refuses an
existing destination through its provisioning API. The report includes hashes
for all original fixture files. Index hashes can vary between builds because
derived records include build metadata; fixture content hashes are stable.

Distributions that exclude source test fixtures still support evaluation with
user-supplied labels. In that case `--fixture` reports that its source fixture
is unavailable. It never silently switches to a different corpus.

This fixture establishes executable behavior and a repeatable local baseline.
It does not establish real-vault quality, native Obsidian behavior, semantic
model quality, or a quality improvement over a previous release. Compare
candidates on the same unchanged labels and corpus and retain every query's
result, including misses and exclusions.
