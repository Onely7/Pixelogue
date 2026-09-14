# Intermediate and final artifact examples

One generated answer passes its rubric, yet the conversation containing it is rejected. Another conversation reaches `QUALITY_CANDIDATE` but still cannot enter training output. These are different boundaries, and the saved records make the difference visible.

[Previous: select and export](selection-and-export.md) · [Back to contents](README.md) · [Japanese version](artifact-examples_ja.md)

## How these examples were obtained

The values below came from the temporary BF16 Qwen3.5-9B/Qwen3.5-2B one-GPU pilot run on 15 September 2026. Qwen3.5-9B substitutes for both standard generator and evaluator models in this validation run; these are not Qwen3.8-27B/Gemma 4 31B results. Long hashes preserve the identity chain. The JSON blocks contain the fields needed for each explanation; the complete content-addressed records retain the remaining fields.

The images are generated evaluation fixtures. They are safe for pipeline validation and remain ineligible for training export.

## 1. Prepared image

The first image contains five red squares. `ingest` produced this identity:

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

## 2. Visible-capability inventory

Qwen3.5-9B reported a complete countable scope:

```json
{
  "image_id": "a96e0109e4d1f21e661ddbd4ef28d8ba6aaf0d149d77006a791f5b349f3b0caa",
  "capabilities": [
    "bounded_complete_scope",
    "countable_entities",
    "multiple_entities",
    "visible_attribute",
    "visible_entity"
  ],
  "visible_scopes": [
    "The entire image is visible and contains exactly five red squares arranged horizontally in a single row."
  ],
  "scope_limited": false
}
```

This is private model evidence. It helps construct candidates but is not copied into the training conversation.

## 3. Candidate and selector result

The controller offered six compatible operations. One was the count task:

```json
{
  "candidate_id": "88225d6b078ceef51d30756f4b22d97f315a6b2fc3ad80ff9c94ea92c111b50d",
  "task_id": "visible_count",
  "family": "visible_count",
  "instruction_summary": "Count all qualifying visible instances within a complete bounded scope.",
  "required_capabilities": ["bounded_complete_scope", "countable_entities"]
}
```

Qwen3.5-2B selected that ID:

```json
{
  "candidate_id": "88225d6b078ceef51d30756f4b22d97f315a6b2fc3ad80ff9c94ea92c111b50d",
  "reason": "The image contains exactly five red squares arranged horizontally. The candidate 'visible_count' with task_id 'visible_count' is the most appropriate as it specifically counts all qualifying visible instances within a complete bounded scope"
}
```

This is the exact selection text returned by the model. The content-addressed response artifact also retains the surrounding vLLM response and token usage.

## 4. Generated question, requirement, and answer

The generation and pre-answer stages produced:

```json
{
  "question": "How many red squares are there in the image?",
  "requirement": {
    "kind": "content",
    "description": "How many red squares are there in the image?",
    "start": 0,
    "end": 44,
    "lifetime": "current_turn"
  },
  "answer": "There are 5 red squares in the image."
}
```

Both evaluator calls extracted the answer as one factual claim. They also bound five expected and five reported set members, and the turn's full rubric aggregate was `PASS`.

## 5. A passing turn inside a rejected conversation

The first turn passed, but the next plan did not produce another acceptable turn. Since a quality candidate needs at least two committed turns, the saved conversation ended as `REJECTED`:

```json
{
  "conversation_id": "5769049f502ce1c43f87a09ef2d82c088663ee552238c7c18201b8413385d0c5",
  "generation_model": "Qwen/Qwen3.5-9B",
  "target_language": "en",
  "status": "REJECTED",
  "turns": [
    {
      "turn_index": 1,
      "question": {"role": "user", "content": "How many red squares are there in the image?"},
      "answer": {"role": "assistant", "content": "There are 5 red squares in the image."},
      "rating": {"aggregate": "PASS"},
      "status": "COMMITTED"
    }
  ]
}
```

The complete JSONL row also stores message IDs, instruction details, requirements, history hashes, and each rubric item. A good first answer is preserved without being mislabelled as a complete multi-turn training candidate.

## 6. A real quality-candidate conversation

The same pilot completed this three-turn conversation. Every displayed turn was committed with a `PASS` aggregate:

```json
{
  "conversation_id": "0400eaad5f5e73ff5be1fca146e5bea27ddd0dea7a8233a94e1e573ee4ead54f",
  "generation_model": "Qwen/Qwen3.5-9B",
  "target_language": "en",
  "status": "QUALITY_CANDIDATE",
  "image": {
    "image_id": "80287bd0bc2a5882ee5c3ebdea624ddbc816fcda95710b1055bd8a858ac22b85",
    "purpose": "evaluation",
    "dataset": "Pixelogue procedural fixtures"
  },
  "turns": [
    {
      "turn_index": 1,
      "task_id": "visible_count",
      "question": "How many objects are visible?",
      "answer": "There are 4 objects visible.",
      "rating": "PASS",
      "status": "COMMITTED"
    },
    {
      "turn_index": 2,
      "task_id": "object_identification",
      "question": "What are the values shown for A and B?",
      "answer": "The values shown are A = 3 and B = 1.",
      "rating": "PASS",
      "status": "COMMITTED"
    },
    {
      "turn_index": 3,
      "task_id": "visible_count",
      "question": "How many green squares are visible?",
      "answer": "There are 4 green squares visible.",
      "rating": "PASS",
      "status": "COMMITTED"
    }
  ]
}
```

`QUALITY_CANDIDATE` describes dialogue quality and turn count. The image still has `purpose: evaluation`, so source eligibility keeps this record out of training selection and export.

## 7. Shape of an eligible training record

No evaluation-only pilot row should be presented as released training data. The following valid JSON has the exact `TrainingRecord` field structure for an eligible two-turn source. Its values are illustrative:

```json
{
  "record_id": "conversation-example-001",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "text": null, "image": "images/ab/abcdef.png"},
        {"type": "text", "text": "How many red squares are there in the image?", "image": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "There are five red squares.", "image": null}
      ]
    },
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What color are they?", "image": null}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "They are red.", "image": null}
      ]
    }
  ]
}
```

The image appears once. Later questions rely on the preserved public history. Internal instruction, requirement, claim, and rating records are absent.

The corresponding provenance row remains separate:

```json
{
  "conversation_id": "conversation-example-001",
  "source_id": "local:example-001",
  "image_id": "image-example-001",
  "visual_group_id": "visual-group-example-001",
  "generation_model": "Qwen/Qwen3.8-27B",
  "student_processor_lock": {
    "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
    "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    "min_pixels": 16384,
    "max_pixels": 4194304
  }
}
```

The two real conversations above remain diagnostic data. Only a standard-profile conversation with training permission, a passing selection audit, and the same public-only structure can reach `training.jsonl`.
