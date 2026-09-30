# Synthesis validation on 2,000 images and examples by task

The pinned Open Images V7 validation cohort contains 2,000 evaluation-only images.
Selection excludes the earlier 20 images and follows SHA-256 ranks of `20261001:ImageID`.
Only download/ingestion failures and exact canonical-pixel duplicates were replaced.
Annotations, titles and boxes never enter model payloads. Near-copy detection was not performed.
[The manifest](../../validation/open_images_v7_eval_2000_manifest.jsonl) records identities,
attributions and byte hashes; downloaded pixels and generated results stay outside Git.

```sh
uv run --locked python validation/fetch_open_images_eval_2000.py
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/open-images-v7-eval-2000/sources.jsonl \
  --rights data/open-images-v7-eval-2000/rights.jsonl \
  --image-root data/open-images-v7-eval-2000 \
  --artifact-root artifacts/prepared-open-images-v7-eval-2000 --workers 8
```

## Frozen synthesis conditions

Copy `configs/paired-one-gpu-pilot.yaml` into `artifacts/open-images-2000/config.yaml`.
Set `profile: standard`, `data.pilot: false`, `data.target_dialogues: 2000`, and `seed: 20261001`.
Set `data.open_images.enabled: false` and `image_ids_manifest: null` to use the independently
prepared manifest rather than the unchanged 20-image `prepare` route.
Keep the primary Qwen/Gemma pair, default Qwen3.5-2B selector, pinned quantization and bfloat16.
The already checked one-device server configuration is specific to zao01's 96 GiB GPU.
Keep eight candidates, two concurrent images, at least two committed turns and both judges.
The seven uncalibrated specialist extensions remain outside normal selection.

```sh
uv run --locked pixelogue synthesize \
  --config artifacts/open-images-2000/config.yaml \
  --images artifacts/prepared-open-images-v7-eval-2000/images.jsonl \
  --artifact-root artifacts/prepared-open-images-v7-eval-2000 \
  --run-id open-images-2000-20261001 --workers 2 \
  --output artifacts/open-images-2000/conversations.jsonl
```

Freeze code, configuration, image order, model allocation and planned turn counts before inference.
Resume the same trial with the same run ID and manifest; changed contracts require a new run.
Keep WAL on zao01's local `/var/tmp/pixelogue` and back up complete snapshots to shared storage.
Record GPU, endpoint health and output progress every ten minutes, and restart reservation
immediately after success or failure.

## Errors, yield and diversity

```sh
uv run --locked pixelogue run-diagnostics \
  --config artifacts/open-images-2000/config.yaml \
  --run-id open-images-2000-20261001 \
  --conversations artifacts/open-images-2000/conversations.jsonl \
  --output-stem artifacts/open-images-2000/diagnostics
uv run --locked python -m pixelogue.synthesis_campaign_report \
  --images artifacts/prepared-open-images-v7-eval-2000/images.jsonl \
  --conversations artifacts/open-images-2000/conversations.jsonl \
  --artifact-root artifacts/prepared-open-images-v7-eval-2000 \
  --output-dir artifacts/open-images-2000/report
```

Count 2–6 committed-turn `QUALITY_CANDIDATE` conversations separately from committed prefixes
inside rejected, abstained or failed conversations. Preserve failed attempts and pre-turn stops.
`ERROR=0` does not imply zero malformed model outputs. Check all 2,000 terminal records alongside
stage reach, retries and stop reasons.

Reports include candidate task/family counts, concentration, exact-question repetitions,
conversation depth, model outcomes and languages. Human quality and semantic duplication remain
unmeasured without independent ballots. Photographs alone do not establish all-72-task coverage.

## Browse actual image/instruction/answer examples

`report/examples.html` lists all 72 tasks, with text and family filtering. The report also writes
`examples.md`, `examples.jsonl`, `task-coverage.csv` and `report.json`. Each task shows up to three
actual examples with the original Q/A, preceding public history, source and conversation IDs,
and thumbnails of the actual canonical model input.

Automatic candidates and diagnostic prefixes are labeled separately. Tasks without examples stay
visible; catalog example questions and invented answers are never substituted. This gallery is
neither a human gold-label audit nor a training export of evaluation images.

[日本語ガイド](synthesis-campaign_ja.md)
