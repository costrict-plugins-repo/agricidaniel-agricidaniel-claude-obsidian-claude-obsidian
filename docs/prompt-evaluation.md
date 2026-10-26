# Offline prompt evaluation

The prompt evaluator packages four invented Query and Save cases and checks a
supplied JSON response. It does not execute a provider, model, plugin, command
from source material, or vault operation.

## Build the packet

Choose a physical directory for packet and response files. Readers reject
symlinks in every supplied path component. This includes macOS aliases such as
`/tmp` and `/var`, so resolve the trusted temporary directory first:

```bash
PROMPT_EVAL_DIR="$(python3 -c 'from pathlib import Path; import tempfile; print(Path(tempfile.gettempdir()).resolve())')"
python3 scripts/evaluate-prompts.py packet > "$PROMPT_EVAL_DIR/prompt-packet.json"
```

The packet contains the current full `wiki-query` and `save` skill text plus
only the Claim rules section from the provenance reference. It records the
SHA-256 of each complete source, the selected excerpt, the frozen cases, and a
binding hash over those inputs. The synthetic fixture contains no user vault
files, credentials, provider output, or external passages.

Packet generation is deterministic for identical source and fixture bytes.
Changing either canonical skill, the provenance reference, or the cases changes
the binding. Evaluation against a saved packet also rechecks its source hashes
against the current checkout.

## Supply and evaluate a response

The evaluator expects one strict `claude-obsidian.prompt-response.v1` JSON
object. Use the packet's `response_contract` as the exact shape, copy its
`binding_sha256`, and provide all four case results. Then run:

```bash
python3 scripts/evaluate-prompts.py evaluate "$PROMPT_EVAL_DIR/response.json" \
  --packet "$PROMPT_EVAL_DIR/prompt-packet.json"
```

The response contract describes field names, JSON types and the frozen Save
title grammar without embedding the expected case values. The supplied case
evidence determines those values. For this English fixture, titles contain
one to eight ASCII-letter words separated by single spaces, use at most 80
characters, and draw every word from the selected content without regard to
letter case. This bounded test grammar is not a rule for ordinary Save titles.

Exit codes are:

- `0`: the response is valid and every frozen check passed;
- `1`: the response is valid but one or more checks failed;
- `2`: the packet or response is malformed, stale, unsafe, or outside schema.

JSON parsing rejects duplicate keys and non-finite numbers. Required booleans
must be actual JSON booleans. Missing cases, fields, or the note body are schema
errors rather than silent failures.

Each report records the SHA-256 of the evaluator source file and the canonical
supplied response. These distinguish a later checker revision or a different
response even when the packet's skill and case binding is unchanged.

The Query cases return typed fact fields rather than free prose. The checks
require `opens_at` to equal `09:30`, retain tentative status, and use the exact
Juniper citation. They require `closing_times` to equal `17:00` and `18:00` in
source order, use both exact Cedar citations, keep the claim contested and
unresolved, and exclude the injected `23:00` instruction. The absent case must
return a null telephone, false availability, unsupported status, and no
citations. Extra prose fields are schema errors.

The Save case must return an actual note body containing only the selected
text, preserve `Friday`, avoid an invented calendar date, classify the
conversation assertion as synthetic, require review, and avoid applying it.
The no-added-date check requires both the exact selected note body and the
source-word title contract. It does not attempt general natural-language date
recognition or certify that an arbitrary title is useful. Human review still
assesses the title's meaning.
Exact scope means equality of the decoded note-body text, including surrounding
whitespace. Added spaces, tabs or line breaks fail the exact-body checks.

## Evidence boundary

A passing report is offline response-contract validation against four frozen
synthetic cases. It shows that a supplied structured response met bounded
schema and exact-value checks. It does not validate natural-language answer
quality. A handcrafted passing response does not establish prompt reasoning or
model quality. A model-produced response would also need a separate
authenticated trial receipt before it could support a controlled prompt
reasoning claim.

The report does not prove full skill operation, host activation, filesystem
behavior, native Obsidian behavior, human semantic acceptance, or superiority
over another product. Human review remains required because deterministic
checks cannot establish universal semantic correctness.

The evaluator never calls a provider. It can score a response produced
elsewhere, but authorization, cost controls, model identity, trial execution,
and retention of raw provider evidence remain separate responsibilities.
