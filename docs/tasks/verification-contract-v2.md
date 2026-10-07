# Verification contracts, revision 2

[日本語](verification-contract-v2_ja.md)

`scope-operations-v2` retained the 72 task IDs of catalog 7.0, the two blind judges and the
minimum of two committed turns. It changed public operation conditions and
verification schemas. Use a new run ID; retain older runs with their original
code and catalog. A CPU fixture pass does not establish natural-image accuracy.

Catalog 8.0 (`scope-operations-v3`) renames and merges task IDs and keeps the rules
below; see the [catalog guide](README.md#changes-from-catalog-70).

## Public conditions and independent evidence

- Chart ranking binds `rank_mode` (`all`, `max`, `min`, `max_min`) and
  `rank_order` before question generation. Both source decoders receive the same
  fixed public choices. `max_min` returns maximum then minimum, retaining ties;
  `rank_order` controls complete ranking. All relevant marks must be visible.
  The source decoder prevents ticks on an unmarked axis and requires explicit
  series/category operands for a complete reading. `UNKNOWN` may leave them empty.
  Chart and graph inventories use `tasks.source_max_tokens`, as table inventories do,
  instead of first truncating a large closed series at a fixed 2,048 tokens.
  Separated intervals can establish an extremum without establishing an exact
  number. Overlapping intervals cannot prove which mark is greater.
- Ranking compares category order without an unused answer digit limit. An
  unmarked axis may omit its unit when every exact numeric mark visibly prints
  the same `%` unit and that printed number equals the extracted value. This
  narrow correspondence does not apply to numeric lookup or reconstruction;
  missing labels, conflicting units and different values remain unresolved.
  Chart retries name invalid region indices without copying raw model content
  or inventing corrected coordinates.
- Registered positional descriptions for unlabeled chart marks resolve only
  within a closed set of horizontally separated marks. Both independent
  readings must agree on values, scope and corresponding overlapping regions.
  Ambiguous positions and arbitrary synonyms do not establish correspondence.
- Extractive document QA reads the complete minimal answer-bearing span twice,
  without the answer. It preserves source language, qualifiers and negation.
  Whole-answer comparison permits sentence-initial case and a final prose
  period to differ. It does not permit a matching substring, translation or
  omitted qualifier to pass. Verbatim text/code transcription remains exact.
- Formula transcription accepts whole LaTeX display containers (`\[...\]`,
  `\(...\)`, `$...$`, `$$...$$`) around the registered mathematical grammar.
  It still compares literal structure, not algebraic equivalence. Matrices,
  chemical bond layouts and unregistered commands remain unsupported.
- Text transcription binds the already declared enclosing scope before
  generating its question. It does not expand a region after seeing an answer.
  A whole requested unit must remain contained and independently readable.
- Graph neighbor/path/edge checks compare topology without requiring an unused
  node subtype. Displayed labels, endpoints, direction, completeness and region
  correspondence remain required. An explicit parenthesized visible label can
  identify a bilingual neighbor name; guessed translations cannot.

Every answer-blind source stage also rejects nested `candidate_answer` fields.
`MET` is never filled in for absent or malformed evidence.

Chart, graph, table and document source decoders receive coordinate bounds from the
already declared public scope. Transcription and extractive QA use the declared
target region when present. Region containment and positive extent are still
validated after decoding; these bounds do not prove visibility or completeness.
Routing distinguishes a document title or axis name from an observed object
category and keeps `object_label` null without supported entity evidence.
An incomplete document source cannot emit certified fields or nodes. Graph
edge regions enclose visible strokes with positive extent, including horizontal
and vertical strokes; line endpoints are not bounding rectangles.

## Static table output

The operation exposes its output grammar to generation and evaluation. HTML is
parsed without rendering or executing it, with content and format verdicts
kept separate. Supported tags are balanced `table`, `thead`, `tbody`, `tfoot`,
`tr`, `th`, `td`, and `br` for cell line breaks.

Allowed attributes are `rowspan`/`colspan` on cells, `scope` on headers, and
numeric `border`/`cellpadding`/`cellspacing` on the table. Optional `style` has
only the declared `border-collapse`, `text-align` and `vertical-align` choices.
Unknown CSS, active markup, event handlers and external assets are rejected.
Text, blanks, header roles, row/column positions and merged spans must match
both independent readings. JSON fields and the simple Markdown limitation are
also published in the operation contract.

## Runtime and accounting

Bounded JSON whitespace requires the verified vLLM 0.29.0 XGrammar patch,
explicit `xgrammar`, and ordinary whitespace enabled. Configuration loading
now rejects an incompatible generator runtime before starting image calls.
Do not invent a runtime manifest to satisfy this check.

Terminal stop records use the actual attempted turn index. Failures, invalid
outputs and retries remain saved. Production vLLM usage outside the request's
token bounds remains an error; missing usage is never zero cost.
A JSON response missing terminal braces or brackets is rejected without repair.
Fully complete structured JSON with trailing whitespace retains its existing
validation path. A native
research CLI's wrapped-output token count is not a demonstrated count of the
domain JSON alone, and its requested generation limits are not enforced.
These limitations must be retained in ceiling-experiment interpretation.

## Small performance comparisons

Compare the same fixed images, model roles, planned depth and seed in one
server session. `evidence_format: compact` assigns immutable IDs in the
controller; `initial_binding_batch_size: 2` binds a prefix and tries subsequent
batches only when no admissible new fact remains. The candidate pool remains
eight. Measure accepted multi-turn conversations, task coverage, reference
ratings, output tokens, retries and wall time together. Summed overlapping HTTP
durations are not wall time. Neither option relaxes a quality gate.
