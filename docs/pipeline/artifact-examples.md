# Intermediate and final artifact examples

One turn can pass both judges and still belong to a conversation that stops early. Another conversation can reach `QUALITY_CANDIDATE` and still be barred from training output. These are different boundaries, and the saved records make the difference visible.

[Previous: select and export](selection-and-export.md) · [Back to contents](README.md) · [Japanese version](artifact-examples_ja.md)

## How to read these examples

Each JSON block has the field structure that the current code writes. Except for the image identity in section 1, the values are illustrative: they were written for this guide, not copied from a measured run. Placeholders such as `conv-example-001` and `<sha256>` stand for long identifiers. Each record below is stored as canonical JSON under `artifacts/<kind>/<first two hex digits>/<sha256>` in the run store; a section title in code font names its `<kind>`.

The examples follow one planned three-turn conversation about an evaluation-only street photo. Turn 1 passes after a full-view tie-break, turn 2 passes after one answer repair, and turn 3 fails an operation check, so the conversation keeps its two-turn prefix.

## 1. Prepared image

This image identity was recorded for a generated evaluation fixture that contains five red squares:

```json
{
  "image_id": "a96e0109e4d1f21e661ddbd4ef28d8ba6aaf0d149d77006a791f5b349f3b0caa",
  "purpose": "evaluation",
  "dataset": "Pixelogue procedural fixtures",
  "dataset_split": "development",
  "canonical_pixel_sha256": "1e3a955fe64c866075e76bb9a250300b869073bf4357820f7d1e626cb0487302",
  "visual_group_id": "ab7efa8dd4295b3afb992ae39e7e1d5d8a203e3e8105336f8000eb5ad3627ac1",
  "full_view": {
    "relative_path": "images/21/21e9a7efd94d6b56d32fe0751db4a0791e1c62f4ca784381fb4e03b216a77ae4.png",
    "width": 640,
    "height": 480,
    "media_type": "image/png"
  }
}
```

The source-and-pixel-derived image ID, canonical pixel hash, and visual group answer different questions. They identify this source record, its normalized visual content, and the group used to prevent copy leakage.

## 2. Image profile and route

The router's response is kept with its model call:

```json
{
  "image_kind": "photo",
  "readable_text": "some",
  "supported_families": ["visual_description", "reference_spatial", "text_reading", "set_logic"],
  "reason": "Street scene with people at a bus stop and a readable shop sign."
}
```

For turn 1, an anchor turn, the controller offers only lightly verified operations. The route is stored in the `turn_route` table and repeated in `question-drafts`:

```json
{
  "primary_family": "text_reading",
  "primary_task_ids": ["text_transcription", "label_value_linking", "text_reading_order", "text_visual_binding"],
  "secondary_family": "visual_description",
  "secondary_task_ids": ["attribute_lookup", "object_identification"],
  "basis": "profile"
}
```

`basis` becomes `fallback` when the profile is missing or names no feasible family. A profile that stays malformed leaves an `image-profile-abstentions` record with `image_id`, `reason`, and `message`.

## 3. `question-drafts`

The conversation's generator returned two drafts for that route:

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 1,
  "call_index": 0,
  "route": {
    "primary_family": "text_reading",
    "primary_task_ids": ["text_transcription", "label_value_linking", "text_reading_order", "text_visual_binding"],
    "secondary_family": "visual_description",
    "secondary_task_ids": ["attribute_lookup", "object_identification"],
    "basis": "profile"
  },
  "batch": {
    "drafts": [
      {
        "task_id": "text_transcription",
        "question": "What does the sign above the shop entrance say?",
        "target": "the sign above the shop entrance",
        "public_parameters": [],
        "scope_region": {"left": 0.08, "top": 0.05, "right": 0.46, "bottom": 0.21},
        "target_region": null,
        "fact_key": {"subject": "shop sign", "dimension": "text"}
      },
      {
        "task_id": "attribute_lookup",
        "question": "What color is the jacket worn by the person standing next to the bus?",
        "target": "the person standing next to the bus",
        "public_parameters": [{"name": "attribute", "value": "jacket color"}],
        "scope_region": {"left": 0.40, "top": 0.15, "right": 0.75, "bottom": 0.98},
        "target_region": {"left": 0.46, "top": 0.20, "right": 0.64, "bottom": 0.95},
        "fact_key": {"subject": "person next to the bus", "dimension": "jacket colour"}
      }
    ],
    "reason": null
  }
}
```

The second draft's private `request_key` becomes `person next to bus|jacket color`: articles are removed and `colour` is normalized, so a later draft cannot ask for the same fact with different spelling.

## 4. `public-text-rejections` and `draft-rejections`

The first draft locates its text relatively, which the transcript evidence cannot verify, so it never reaches a judge:

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 1,
  "field": "question",
  "reason": "TEXT_RELATION_UNVERIFIED",
  "content": "What does the sign above the shop entrance say?"
}
```

A draft that breaks its operation contract is recorded separately. In turn 3, an `attribute_grouping` draft omitted its required `return` choice:

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 3,
  "call_index": 0,
  "draft_index": 0,
  "reason": "DRAFT_PARAMETER_MISSING",
  "message": "Missing parameters ['return']"
}
```

## 5. `question-gate-decisions`

The second draft of turn 1 went to both judges. Their crops showed the person; both labels named the drafted operation:

```json
{
  "question_message_id": "conv-example-001:q:1",
  "candidate_id": "cand-example-1",
  "drafted_task_id": "attribute_lookup",
  "votes": [
    {
      "local_anchor": "MET",
      "operation_coherent": "MET",
      "useful_request": "MET",
      "reason": "Asks for the jacket color of the bound person without stating it.",
      "realized_task_id": "attribute_lookup"
    },
    {
      "local_anchor": "MET",
      "operation_coherent": "MET",
      "useful_request": "MET",
      "reason": "The person is visible in the crop and the color is not given.",
      "realized_task_id": "attribute_lookup"
    }
  ],
  "label_rule": "exact",
  "fit": "MET",
  "decided_task_id": "attribute_lookup",
  "verdict": "MET"
}
```

`decided_task_id` differs from `drafted_task_id` only after a `relabel` under the `same_contract` policy.

## 6. `rating-decisions`

The answer was “The jacket is yellow.” One judge returned `UNKNOWN` because the crop did not show the bus, so it reviewed the complete image once more:

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 1,
  "template_id": "Q_HOLISTIC",
  "subject": null,
  "votes": [
    {"verdict": "MET", "reason": "The person beside the bus wears a yellow jacket."},
    {"verdict": "UNKNOWN", "reason": "The crop does not show whether this person stands next to the bus."}
  ],
  "tiebreak_vote": {"verdict": "MET", "reason": "The full image shows the person next to the bus in a yellow jacket."},
  "repair_objection": null,
  "controller_reason": null,
  "verdict": "MET"
}
```

`votes` keeps the first-round votes in judge order. `controller_reason` is `INVALID_PUBLIC_TEXT` or `EMPTY_FOCUS_VIEW` when the controller decided without judges. `attribute_lookup` has no verification contract beyond the two reviews, so turn 1 was committed.

## 7. `answer-attempts`

In turn 2, “What is the person on the left doing?” first received “The person on the left is reading a newspaper.” One judge returned `MET` and the other `NOT_MET`. The `rating-decisions` record of that review carries the objection in `repair_objection`, and the replaced answer is kept here:

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 2,
  "attempt": 0,
  "answer": {
    "message_id": "conv-example-001:a:2",
    "turn_index": 2,
    "role": "assistant",
    "content": "The person on the left is reading a newspaper."
  },
  "rating": {
    "items": [
      {
        "item_id": "<sha256>",
        "template_id": "Q_HOLISTIC",
        "axis": "factual_correctness",
        "verdict": "UNKNOWN",
        "reason": "generator_a:The person on the left is reading. | generator_b:The person on the left holds a phone, not a newspaper.",
        "actor": "dual-consensus",
        "history_hash": "<sha256>"
      }
    ],
    "aggregate": "ABSTAIN"
  },
  "objection": "The person on the left holds a phone, not a newspaper."
}
```

Besides the image, history, and question, the generator received only the original answer and the objection. Its replacement, “The person on the left is looking at a phone.”, passed a fresh review by both judges, and turn 2 was committed.

## 8. `operation-checks`

Turn 3 asked “How many people are waiting at the bus stop?” and received “There are three people waiting at the bus stop.” Both holistic reviews returned `MET`, but `entity_count` also requires `closed_set_check`. Both readers independently counted four:

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 3,
  "contract": "closed_set_check",
  "candidate_id": "cand-example-3",
  "question_hash": "<sha256>",
  "answer_hash": "<sha256>",
  "verdict": "NOT_MET",
  "independent_evidence": [
    {
      "coverage": "MET",
      "mode": "count",
      "counts": [{"scope": "all", "expected": 4, "reported": 3}],
      "expected_members": [],
      "reported_members": [],
      "empty_scope_is_explicit": false,
      "reason": "Four people wait at the stop; the answer reports three."
    },
    {
      "coverage": "MET",
      "mode": "count",
      "counts": [{"scope": "all", "expected": 4, "reported": 3}],
      "expected_members": [],
      "reported_members": [],
      "empty_scope_is_explicit": false,
      "reason": "One person behind the shelter pole is also waiting."
    }
  ]
}
```

The two readings agree, so the controller compares their expected and reported counts. The turn's aggregate becomes `FAIL`.

## 9. Stop record and retained prefix

The failed turn ends the loop:

```json
{
  "conversation_id": "conv-example-001",
  "turn_index": 3,
  "status": "REJECTED",
  "stage": "answer_verification",
  "reason": "RATING_FAIL"
}
```

This is a `conversation-stop-reasons` record. Because two turns were already committed and `evaluation.retain_accepted_prefix` is true, the complete three-turn conversation is kept privately in `conversation-stops` with `"retained_turns": 2`, and the final `conversations` row contains only the prefix. The abbreviated, flattened row below omits message IDs, operation contracts, history hashes, and rating items:

```json
{
  "conversation_id": "conv-example-001",
  "generation_model": "Qwen/Qwen3.8-27B-FP8",
  "target_language": "en",
  "status": "QUALITY_CANDIDATE",
  "image": {
    "image_id": "image-example-001",
    "purpose": "evaluation",
    "dataset": "Example evaluation images"
  },
  "turns": [
    {
      "turn_index": 1,
      "task_id": "attribute_lookup",
      "question": "What color is the jacket worn by the person standing next to the bus?",
      "answer": "The jacket is yellow.",
      "rating": "PASS",
      "status": "COMMITTED"
    },
    {
      "turn_index": 2,
      "task_id": "visible_action_relation",
      "question": "What is the person on the left doing?",
      "answer": "The person on the left is looking at a phone.",
      "rating": "PASS",
      "status": "COMMITTED"
    }
  ]
}
```

`QUALITY_CANDIDATE` describes dialogue quality and turn count. The image still has `purpose: evaluation`, so source eligibility keeps this record out of training selection and export.

## 10. Shape of an eligible training record

No evaluation-only row should be presented as released training data. The following valid JSON has the exact `TrainingRecord` field structure for an eligible two-turn source. Its values are illustrative:

```json
{
  "record_id": "conversation-example-001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "text": null, "image": "images/ab/abcdef.png"},
        {"type": "text", "text": "What color are the squares in this image?", "image": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "They are red.", "image": null}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "How are the squares arranged?", "image": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "They form a single horizontal row.", "image": null}
      ]
    }
  ]
}
```

The image appears once. Later questions rely on the preserved public history. Operation contracts, drafts, votes, and rating records are absent.

The corresponding provenance row remains separate:

```json
{
  "conversation_id": "conversation-example-001",
  "source_id": "local:example-001",
  "image_id": "image-example-001",
  "visual_group_id": "visual-group-example-001",
  "generation_model": "Qwen/Qwen3.8-27B-FP8",
  "student_processor_lock": {
    "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
    "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "min_pixels": 16384,
    "max_pixels": 4194304
  },
  "operation_ids": ["attribute_lookup", "grounded_description"],
  "catalog_versions": ["7.0", "7.0"],
  "primary_operation_id": "grounded_description"
}
```

Only a standard-profile conversation with training permission, a passing selection audit, and the same public-only structure can reach `training.jsonl`.

## 11. Private record kinds

| Kind | Written when | Main fields |
|---|---|---|
| `question-drafts` | A drafting call returns a valid batch | `conversation_id`, `turn_index`, `call_index`, `route`, `batch` |
| `draft-rejections` | A draft breaks its operation contract | `call_index`, `draft_index`, `reason`, `message` |
| `draft-abstentions` | A drafting call stays malformed after retries | `call_index`, `reason`, `message` |
| `public-text-rejections` | A deterministic check rejects question or answer text | `field`, `reason`, `content` |
| `question-gate-decisions` | Both judges have gated a question | `votes`, `label_rule`, `fit`, `drafted_task_id`, `decided_task_id`, `verdict` |
| `image-profile-abstentions` | The router profile stays malformed after retries | `image_id`, `reason`, `message` |
| `focus-views`, `focus-images` | A judge crop is cut | crop size, `source_view_id`, `source_left` to `source_bottom`; the PNG bytes |
| `focus-view-abstentions` | A bound region contains no whole pixel | `candidate_id`, `region` |
| `rating-decisions` | A holistic review is decided | `votes`, `tiebreak_vote`, `repair_objection`, `controller_reason`, `verdict` |
| `answer-attempts` | A repair replaces an answer | `answer`, `rating`, `objection` |
| `operation-checks` | A validator finishes | `contract`, `verdict`, `independent_evidence` |
| `structured-output-failures` | A model output breaks its contract | `stage`, `attempt`, `reason`, `next_retry_feedback` |
| `turns` | A turn is committed | the complete turn artifact |
| `conversation-stop-reasons` | A conversation stops before its planned length | `turn_index`, `status`, `stage`, `reason` |
| `conversation-stops` | A stopped conversation keeps an accepted prefix | `conversation`, `retained_turns` |
| `conversations` | A conversation is final | the complete conversation artifact |
| `model-output-abstentions`, `errors` | An image, or a re-rated turn, ends on a model-output or execution failure | `conversation_id`, `image_id`, `reason`, `message` |
| `requests`, `responses`, `request-images` | Every model call | request envelope, raw response, deduplicated image data |
| `transport-errors`, `cache-accesses` | HTTP failures and reused saved responses | status or error type, attempt, request hash |

Stores written by the retired scoped planner may also contain evidence, binding, intent, and requirement records. The current code writes none of these kinds, and the turns saved by that planner remain readable.
