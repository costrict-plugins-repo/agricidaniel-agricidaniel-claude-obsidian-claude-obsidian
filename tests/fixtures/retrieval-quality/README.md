# Original retrieval quality fixture

This small invented corpus was authored for offline regression measurement.
Its labels were chosen from the page meanings before running the evaluator.
It contains exact identifiers, Unicode text, lexical queries, semantic synonyms,
multiple relevant pages, unrelated terms, and a stale conflicting cache entry.
It contains no user vault content or external passages.

The corpus is a functional evaluation fixture. It cannot establish quality on
real vaults or comparative model performance. The synonym query intentionally
exposes a lexical limitation instead of requiring a perfect aggregate score.

`pages/` is copied only into an absent temporary fixture vault. After synthetic
prefixing and BM25 indexing, `post-index/` replaces the named page only in that
temporary vault. This deliberately leaves one stale chunk for rejection tests.
`queries.json` always remains outside the generated vault and is never changed
by evaluation. Reproducible hashes are reported for its exact bytes and index.
