# Optional contextual prefix providers

Synthetic prefixes remain the default. Preview with `--peek`; use `--no-llm`
to keep a build deterministic and provider-free. Selecting a provider changes
derived caches only, and requires an explicit build. Canonical pages remain
unchanged.

## An installed local Ollama model

Pass `--local-ollama --ollama-model name:tag` to contextual-prefix.py. The
adapter never starts a server, pulls a model or falls back to a cloud provider.
It requires a cloud-disabled status response, an exact installed model tag and
digest, and local model metadata before transmitting page text. Older servers
without these fields fall back to synthetic prefixes. A changing model digest
invalidates cached prefixes on the next explicit build.

When that preflight is unavailable, for example during a transient outage, the
cause is reported once for the operation and no page text is sent. Existing
records produced by the same endpoint and model under the current prompt
revision are kept for unchanged chunks, because their digest cannot be
re-proved until the server returns. Only new or changed chunks receive a
synthetic prefix. If no chunk changes, chunk records and the ready BM25 index
remain byte identical.

The default endpoint is the numeric loopback `127.0.0.1`. The exact hostname
`localhost` is normalized to `127.0.0.1` before any request, and literal `::1`
remains available. This prevents local mode from relying on host resolver
configuration. An explicit non-loopback endpoint additionally requires the
user's approval and `--allow-remote-ollama`. Select HTTPS for a remote server
when page contents need transport encryption. Standard certificate hostname
validation applies to the normalized numeric endpoint, so a certificate for
only `localhost` does not cover `https://127.0.0.1`. Proxies and redirects are
disabled. JSON requests and responses are bounded, as are output and timeouts.
Each timeout is one wall-clock deadline for the complete request: connection,
upload, response headers and every body read. A server that drips bytes cannot
extend it. Responses are parsed strictly, and a duplicate object key at any
depth rejects the response.
Unsupported responses, refusal, truncation and mismatched model identity
produce synthetic prefixes and open the operation circuit. Later chunks and
pages then fall back without another generation request. A user-supplied
server can misrepresent its configuration; the preflight does not attest the
remote machine.

The optional Ollama embedding endpoints used by reranking and semantic tiling
apply the same exact `localhost` to `127.0.0.1` normalization before their
first transport request. They preserve an explicit port and base path, keep a
literal `::1`, and retain the separate remote endpoint consent requirement.
Before any query, page, chunk, or embedding cache work, all Ollama callers
require server status to report `cloud.disabled` as the Boolean value `true`.
They also require a selected installed model with a valid digest and nonempty
local `model_info`. Cloud tags, remote model fields, malformed identity, and
conflicting alias digests fail this proof. Remote fields are decisive in the
status and model metadata responses, in the model list envelope, and in the
list entries for the selected model or its permitted alias. Another installed
cloud model never receives the request and does not block a local model. An
explicit tag matches exactly. An
untagged embedding model matches only its exact name or `:latest`, preferring
the exact name. `--allow-remote-ollama` approves the server endpoint only. It
does not approve cloud delegation by that server.
Every embedding GET and POST disables ambient proxies, refuses redirects,
requires the final response URL to match the requested URL, and reads a bounded
JSON response within the same per-request wall-clock deadline and strict
duplicate-key parsing. A malformed HTTP status line or other protocol error is
treated as an unavailable server: reranking falls back to BM25 order and
tiling exits 10, or 12 once local proof has begun. Reranking validates a nonempty query within 8,000 NFKC-normalized
characters and at most 1,000 object candidates with safe relative paths and
finite scores before model, endpoint, cache, or embedding work. Semantic tiling
validates threshold schema version 1 and finite bands satisfying
`0 <= review < error <= 1` before provider probes or cache reads and writes.
Malformed input exits through the documented controlled interface and sends no
query, candidate, or page body to Ollama.
Embedding responses and semantic tiling cache vectors must be bounded lists of
finite, nonboolean numbers with one consistent dimension per model. Malformed
cache documents exit through the cache corruption contract before any cache
publication. Rerank cache keys include the verified model digest. Semantic
tiling cache schema version 2 binds reusable entries to the resolved model name
and digest. A valid version 1 cache is treated as a miss, while a malformed
legacy cache remains corruption. The first semantic tiling embedding failure,
including an HTTP error, remote identity in the response, or a dimension
change, opens a circuit for the run: no further page body is sent, and cached
vectors still contribute to the report. Partial or total embedding failure
preserves the previous cache, prints a provisional incomplete report, and
exits 12.
Malformed rerank chunk JSON,
including non-text selected content, preserves the whole candidate set through
the documented BM25 fallback.
JSON parser conversion or nesting limits follow the same cache, threshold and
candidate error or fallback contracts as other malformed JSON. They do not
escape as raw parsing exceptions.

The source contract was reviewed against Ollama revision
`d1ed3fd13ac73f60cff00bbead8af24a68c4200d`, including its
[status handler](https://github.com/ollama/ollama/blob/d1ed3fd13ac73f60cff00bbead8af24a68c4200d/server/routes.go)
and [response types](https://github.com/ollama/ollama/blob/d1ed3fd13ac73f60cff00bbead8af24a68c4200d/api/types.go),
checked 2026-10-17. Mocked protocol tests are separate from live server
compatibility. No minimum released Ollama version has been established.

## An explicitly approved cloud provider

`--allow-egress` permits direct Anthropic API requests when
`ANTHROPIC_API_KEY` is present. An API key never counts as consent. State which
provider will receive the selected page bodies before an approved build.

The API defaults to `--anthropic-model claude-haiku-5-5`, using low effort,
adaptive thinking and a bounded 2,048-token output budget. The explicit legacy
profile is `--anthropic-model claude-haiku-4-5-20251001`. Unsupported model
identifiers are refused. Refused, truncated, empty or unusable output falls
back directly to a synthetic prefix and opens the operation circuit. Remaining
chunks and pages use synthetic prefixes without repeating a failed generation
request. Local Ollama mode never selects the API.

Generation request bodies have three configurable operation guards. Defaults
are 128 requests, 16,777,216 serialized request-body bytes for the complete
operation, and 1,048,576 serialized UTF-8 request-body bytes for one request.
Use `--provider-max-requests`, `--provider-max-bytes`, and
`--provider-max-request-bytes` to change those values within their hard caps.
The hard caps are 512 requests, 134,217,728 operation bytes, and 4,194,304
bytes for one request.
Local Ollama keeps its smaller 131,072-byte request limit. The complete set of
planned requests for one page is measured before that page's first generation
call. A page that cannot fit the remaining count or byte budget uses synthetic
prefixes without partial provider execution. Counts and bytes are shared
across every selected page in one command. Synthetic and preview paths consume
none of this generation budget, including explicit `--no-llm` runs.

The 512-token cache floor is model-specific. The character-count estimate is
only a heuristic; actual cache reads and writes require provider telemetry.
Changing the model or generation profile refreshes derived prefixes on the
next explicit build. These
profiles were checked against current [model guidance](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide)
and [caching guidance](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
on 2026-10-17. Mocked payload tests establish request behavior, not live model
quality, cache savings or billing.

The ambient Claude CLI generation tier has been removed. No `claude -p`
subprocess, subscription fallback, host tools, hooks or loaded host settings
participate in prefix generation. With no usable API key or selected local
Ollama provider, the result is synthetic. Older `claude-cli` records and records
from previous prompt revisions regenerate on the next explicit build, even
when their text happens to match the synthetic result. They are never retained
as an equivalent safe fallback.

## Source envelopes and output validation

Every current wiki page is decoded as strict UTF-8 during the address
preflight, before optional provider preflight, page envelope construction,
provider requests, or derived cache and index mutation. Malformed UTF-8 exits
with the controlled page error and preserves existing chunks and index bytes.
The direct single-page processing entry point applies the same strict decode
before it inspects derived state.

Both generators use the same system trust policy. Each processed page creates
one fresh random identifier shared by all of that page's page and chunk tags.
The identifier is named in each request's system instruction. Titles, bodies
and chunks are JSON encoded and HTML escaped inside those tags, so embedded
quotes, matching closing tags, control characters and fake role text remain
data. Request identifiers do not enter the derived
prefix profile, so randomness alone never causes a rebuild. Reusing the same
identifier within one page keeps its system and page blocks byte stable across
chunk requests. Eligibility markers are retained, but only provider telemetry
establishes actual cache use.

A model response must be UTF-8 encodable, nonempty, a single line, at most 34
whitespace-separated words and at most 280 Unicode characters. The character
bound also applies to writing without spaces. Validation allows non-Latin
scripts and the joining characters used by some languages. It rejects control
characters, line separators, directional formatting controls and markup
syntax. Invalid responses fall back as a whole; the adapter never silently
keeps the first line or truncates an invalid model response. The deterministic
synthetic prefix retains its separate 1,000-character bound.

Every replacement record for a page is serialized before its ready BM25 index
is invalidated. An encoding failure therefore preserves that page's previous
chunks and index. If a provider fails and its synthetic fallback exactly matches
an existing generation with the current safe synthetic profile and matching
source identity, the original chunk and ready index remain byte identical.
Fallback records identify the effective provider as synthetic. A changed chunk
set still invalidates the old index before publication and needs the usual
explicit BM25 rebuild. Filesystem publication failures after invalidation also
leave retrieval unavailable until rebuilt, preventing a mixed-generation read.

An existing chunk is skipped only when its schema, source hashes, raw text,
character count, contextualized text, effective provider, and complete prefix
profile agree with the current page and requested generation profile. Unknown
schemas, duplicate JSON keys, non-object JSON values, and malformed or
internally inconsistent derived records regenerate. A synthetic record must
also match the deterministic prefix recomputed from the current page and chunk.

These checks establish structural boundaries and bounded output, not semantic
sentence correctness, instruction detection or prompt-injection resistance.
The tests use mocked responses and temporary vaults; live provider behavior,
retrieval quality and adversarial model acceptance remain unverified.
