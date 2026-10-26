# Synthetic prompt quality fixture

These four invented cases exercise a bounded structured Query response and Save
draft contract. They cover a provisional known fact, conflicting evidence with
an embedded instruction, an absent fact, and a selected Save draft that
contains a relative weekday but no calendar date.

The fixture contains no user vault content, credentials, provider output, or
external passages. `good-response.json` uses a binding placeholder because the
packet generator binds every run to the current Query and Save source hashes.

This supports offline structured response-contract validation. It does not
validate natural-language answer quality. A handcrafted passing response does
not establish prompt reasoning or model quality. A model-produced response also
needs a separate authenticated trial receipt before it can support a controlled
prompt reasoning claim. This is not provider execution, full skill operation,
host acceptance, native Obsidian verification, human semantic acceptance, or a
competitive benchmark.
