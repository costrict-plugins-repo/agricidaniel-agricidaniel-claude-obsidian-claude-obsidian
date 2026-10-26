# Reconciling upstream references

`config/upstream-references.json` records the primary repository revisions
reviewed on 2026-10-17, inspected paths, license boundaries, local references,
and intentional divergences. These are dated source observations, not installed
runtime versions or current-head claims. This update imports no upstream code,
fixture corpus, branding, or substantial prose.

Check local fallback hashes before reviewing a change. A hash mismatch means
local text needs reconciliation; it is not permission to restore old text.
Compare a proposed upstream revision against the pinned revision using an
explicitly requested network read. Review behavior, source authority, license
changes and application compatibility before moving a pin. Never download and
replace skills automatically. Keep the 15-skill interface and transactional
vault boundary unless a separately reviewed design changes them.

When updating a reference, record the candidate SHA, exact inspected paths,
retrieval date, native evidence when available, and new intentional divergences.
Then update affected local hashes and run the reference-example tests plus
package/contracts checks. Moving a pin without reviewing content is not an
upstream refresh. No scheduler or installation is implied by this manifest.

MIT references still require their notices for copied substantial portions.
Templater is an interoperability target under AGPLv3; do not copy its code into
this MIT product without a separate license decision. Smart Connections uses a
custom restricted license, so its implementation is excluded from copying.
Web Clipper's marketing assets are outside its source-code MIT grant. A skill
mentioning Knap grants no license to Knap itself, which was not inspected.

Known disagreement: official Bases Help describes date differences in
milliseconds while the 1.9.3 changelog and upstream skill describe a Duration.
The recorded `obsidian.d.ts` excerpts do not include a Duration definition and
are not cited for either side. Preserve this as unresolved until supported
native app tests exist.
The local Defuddle skill intentionally stops at inert plans/filesystem capture,
even when an upstream recipe would run the CLI. Local transaction, no automatic
installation, selected-vault and network-consent rules remain authoritative.
