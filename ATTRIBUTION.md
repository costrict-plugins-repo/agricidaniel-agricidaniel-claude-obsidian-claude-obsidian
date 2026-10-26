# Attributions

claude-obsidian is an original work. The following third-party patterns, tools, and creators informed its design.

---

## LLM Wiki Pattern

**Author:** Andrej Karpathy
**Source:** https://github.com/karpathy
**Use:** The core architecture of claude-obsidian (using an LLM to build and maintain a structured wiki from raw sources) is based on the LLM Wiki pattern Karpathy described publicly. claude-obsidian is an independent implementation; no code or content from Karpathy's repositories was copied.

---

## ITS CSS Snippets

**Author:** SlRvb
**Source:** https://github.com/SlRvb/Obsidian--ITS-Theme
**License:** GPL-2.0
**Files:**
- `.obsidian/snippets/ITS-Dataview-Cards.css`
- `.obsidian/snippets/ITS-Image-Adjustments.css`

These snippets exist only in the private contributor-vault state. They are not
selected into the deterministic public product artifact and are not installed
by the vault template. The project-owned `templates/vault/.obsidian/snippets/vault-colors.css`
is the only snippet distributed by claude-obsidian.

---

## Historical contributor-vault plugins

The source repository has contained the following Obsidian community plugins in
its contributor-vault state. They are not included in the deterministic public
product artifact or installed by the vault template. Install and review them
through Obsidian under their respective licenses.

| Plugin | Author | Repository |
|--------|--------|-----------|
| Calendar | Liam Cain | https://github.com/liamcain/obsidian-calendar-plugin |
| Thino | Boninall (Quorafind) | https://github.com/Quorafind/Obsidian-Thino |
| Obsidian Excalidraw | Zsolt Viczian | https://github.com/zsviczian/obsidian-excalidraw-plugin |
| Obsidian Banners | Danny Hernandez | https://github.com/noatpad/obsidian-banners |

`scripts/setup-vault.sh` does not download unverified plugin binaries. It prints
manual installation guidance instead.

---

## Adopted contributor designs

Two community pull requests diagnosed retrieval bugs and proposed fixes that
the v2.0.0 refoundation implemented in near-identical form. The pull requests
were not merged as submitted because the surrounding files were rewritten, but
the designs are theirs.

| Change | Contributor | Pull request |
|--------|-------------|--------------|
| Nomic `search_query` and `search_document` task prefixes in the reranker, including the `with_task_prefix` helper and cache-scheme invalidation | maartengoet | https://github.com/AgriciDaniel/claude-obsidian/pull/77 |
| BM25 fallback order preserved in reranker no-op paths (`bm25_score` before `score`) | vinsocci | https://github.com/AgriciDaniel/claude-obsidian/pull/62 |

Version 3.0.0 also draws on the following diagnosed fixes and proposals. These
credits do not imply that a pull request was merged or accepted without
revision.

| Change or design input | Contributor | Pull request |
|--------|-------------|--------------|
| Split encoded Markdown destinations before decoding their path | Ychris12138 | https://github.com/AgriciDaniel/claude-obsidian/pull/194 |
| Canvas `_index.md`, title-preserving routes and legacy journal compatibility | Marcos64s | https://github.com/AgriciDaniel/claude-obsidian/pull/199 |
| Haiku 5.5 model and cache-floor migration, revised here with explicit thinking and output limits | hphsu | https://github.com/AgriciDaniel/claude-obsidian/pull/195 |
| Optional local Ollama prefix tier, independently reimplemented here with bounded transport and local-model preflight | vinsocci | https://github.com/AgriciDaniel/claude-obsidian/pull/63 |
| Dataview queries use vault-relative content paths | miraclemanza | https://github.com/AgriciDaniel/claude-obsidian/pull/46 |

These community pull requests were merged as submitted.

| Change | Contributor | Pull request |
|--------|-------------|--------------|
| ZCode host adapter for portable skill discovery | Lion-1209 | https://github.com/AgriciDaniel/claude-obsidian/pull/146 |
| One page-type vocabulary, so the router files every documented type | HectorBernstorff | https://github.com/AgriciDaniel/claude-obsidian/pull/149 |
| Lint ignores gitignored candidates when resolving ambiguous wikilinks | Packetslave | https://github.com/AgriciDaniel/claude-obsidian/pull/159 |
| Ingest prefers Markdown links for citable URLs | CooperSheroy | https://github.com/AgriciDaniel/claude-obsidian/pull/176 |
| UTF-8 test output so Windows test runs do not crash | yotamleo | https://github.com/AgriciDaniel/claude-obsidian/pull/178 |
| Contextual prefixes parse CRLF frontmatter | ketpatil77 | https://github.com/AgriciDaniel/claude-obsidian/pull/180 |
| Scaffolded vaults pinned to the absolute wikilink format | tilusnet | https://github.com/AgriciDaniel/claude-obsidian/pull/182 |

Version 2.2.0 also draws on these pull requests, implemented here independently.

| Change | Contributor | Pull request |
|--------|-------------|--------------|
| The repository root ignores `.mcp.json`, a possible credential carrier | GPezzuti | https://github.com/AgriciDaniel/claude-obsidian/pull/44 |
| `stop_status` reads large valid journals instead of reporting them unreadable | STiFLeR7 | https://github.com/AgriciDaniel/claude-obsidian/pull/172 |
| `checkpoint` honors Git's false values for `core.filemode` | STiFLeR7 | https://github.com/AgriciDaniel/claude-obsidian/pull/173 |

These issue reports, several with a diagnosis or a tested patch, led to fixes
in 3.0.0.

| Report | Reporter | Issue |
|--------|----------|-------|
| Lint passed a flat vault clean with 0 pages scanned | RoelKluin | https://github.com/AgriciDaniel/claude-obsidian/issues/200 |
| Decoding before the fragment split broke `%23`-encoded links | threnard | https://github.com/AgriciDaniel/claude-obsidian/issues/193 |
| Lint reported existing hidden `.raw` targets as missing files | szrenwei | https://github.com/AgriciDaniel/claude-obsidian/issues/191 |
| SessionStart and Stop hooks timed out on Windows | ppuri2 | https://github.com/AgriciDaniel/claude-obsidian/issues/168 |
| CLI transport detection ignored whether this vault was reachable | cosmos75 | https://github.com/AgriciDaniel/claude-obsidian/issues/141 |
| Glossary proposal | simonc | https://github.com/AgriciDaniel/claude-obsidian/issues/25 |

Pinned ecosystem references, licenses and intentional differences are recorded
in [config/upstream-references.json](config/upstream-references.json). The
3.0.0 original examples do not incorporate restricted third-party plugin
implementations or personal contributor vaults.

---

## claude-obsidian

**Author:** AgriciDaniel / AI Marketing Hub
**License:** MIT (see [LICENSE](LICENSE))
**Repository (public canonical):** https://github.com/AgriciDaniel/claude-obsidian
**Community early-access mirror (Pro):** https://github.com/AI-Marketing-Hub
