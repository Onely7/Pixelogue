# Prepare images

An image file alone is not enough input. Before Pixelogue decodes its pixels, it needs a stable source identity and a rights decision. That order prevents an unreviewed file from quietly entering generation.

[Back to contents](README.md) · [Next: generate and evaluate](generation-and-evaluation.md)

## 1. Validate the configuration

Run `compile` before downloading models or images:

```sh
uv run --locked pixelogue compile \
  --config configs/pilot.yaml \
  --output artifacts/compiled-plan.json
```

The command resolves the configuration, exact language quotas, task and rubric catalogs, and JSON Schemas. It writes `compiled_hash`, a digest of that resolved plan. A schema error here is cheaper to fix than an error halfway through GPU inference.

The pilot profile is a diagnostic profile. It has a small target, a four GPU-hour limit, and cannot be exported as training data. The standard profile is reserved for a reviewed production run.

## 2. Obtain input images

Pixelogue supports two entry paths.

### Fixed Open Images V7 validation sample

```sh
uv run --locked pixelogue prepare \
  --config configs/pilot.yaml \
  --destination data/open-images-v7
```

The checked-in manifest fixes 20 validation image IDs and their expected source metadata and hashes. `prepare` downloads only those images. They remain evaluation-only.

The directory then contains:

```text
data/open-images-v7/
├── images/                              # downloaded bytes; ignored by Git
├── sources.jsonl                        # model-safe source records
├── rights.jsonl                         # rights decisions
├── open_images_private_metadata.jsonl   # source URLs and operator-only metadata
└── open_images_failures.jsonl           # deterministic download failures
```

Labels, bounding boxes, titles, and descriptions are not copied into `sources.jsonl` and are not sent to any model. They also are not treated as answers to generated questions.

### Local or separately acquired images

Create one JSON object per line in `sources.jsonl` and `rights.jsonl`. The records join through `rights_record_id`.

```json
{"source_id":"local:blue-sign-001","image_path":"images/blue-sign.png","source_group_ids":[],"rights_record_id":"rights:blue-sign-001","purpose":"training","dataset":"local-reviewed","dataset_split":"source"}
```

```json
{"rights_record_id":"rights:blue-sign-001","license_uri":"https://example.org/licence","attribution":"Example author","processing_allowed":true,"qa_redistribution_allowed":true,"image_redistribution_allowed":false,"training_allowed":true,"valid_from":"2026-09-15T00:00:00Z"}
```

The example shows the shape, not a permission decision for a real image. The operator must verify the actual licence. A path is relative to `--image-root`; absolute paths and `..` escapes are rejected.

For a training source, `processing_allowed`, `qa_redistribution_allowed`, and `training_allowed` must be true at the evaluation time. An evaluation source needs processing permission but remains blocked from training export.

## 3. Create deterministic fixtures when testing the pipeline

```sh
uv run --locked pixelogue make-fixtures \
  --destination data/fixtures \
  --pairs-per-stratum 2
```

Fixtures are generated images with known positive and single-error negative cases. They test wiring and evaluator behavior without claiming that synthetic examples measure natural-image accuracy.

`evaluate-capabilities` can run both evaluator roles over those answer-keyed cases. Expected labels stay in the controller and are not included in model prompts.

## 4. Normalize and group images

```sh
uv run --locked pixelogue ingest \
  --config configs/pilot.yaml \
  --sources data/open-images-v7/sources.jsonl \
  --rights data/open-images-v7/rights.jsonl \
  --image-root data/open-images-v7 \
  --artifact-root artifacts/prepared-open-images
```

`ingest` performs these steps in order:

1. Reject duplicate source IDs, duplicate rights IDs, and missing rights references.
2. Check the rights validity window and the requested source purpose.
3. Resolve the path inside `--image-root` and read the file safely.
4. Reject unsupported formats, animation, files over 20 MiB, decoded images over 40 million pixels, and images with a short edge below 128 pixels.
5. Apply EXIF orientation and the declared Open Images rotation.
6. Convert an embedded colour profile to sRGB when present, place transparency on white, and write a deterministic RGB PNG view.
7. Hash both the encoded view and canonical pixels.
8. Group exact pixels and declared source groups. Optionally add SSCD near-copy grouping.
9. If any member of a visual group is evaluation-only, mark the whole group evaluation-only.
10. Assign training sources to a stable split from the visual group and configured seed.

One invalid image becomes a row in `failures.jsonl`; it does not erase valid rows. Duplicate identities and dangling rights references reject the manifest because continuing would make resume behavior ambiguous.

## 5. Inspect the prepared result

```text
artifacts/prepared-open-images/
├── images/
│   └── ab/abcdef...png
├── images.jsonl
├── failures.jsonl
└── manifest.json
```

`images.jsonl` is the direct input to `synthesize`. Each row contains source identity, purpose, dataset, byte and pixel hashes, visual group, dimensions, media type, and the relative path of the normalized view.

Check the counts before using a GPU:

```sh
wc -l artifacts/prepared-open-images/images.jsonl
wc -l artifacts/prepared-open-images/failures.jsonl
uv run --locked pixelogue doctor \
  --config configs/pilot.yaml \
  --check-servers
```

Read every failure reason. An empty failure file is expected for a fully reproduced fixed sample; it is not a requirement for arbitrary input collections.

## 6. Keep active storage local

The repository may be on NFS, while SQLite WAL requires a local filesystem. The checked-in configuration therefore places active run state under `/var/tmp/pixelogue`. Prepared image artifacts may live elsewhere, but the live database and its WAL files must remain local.

Image preparation is complete when `images.jsonl` parses, rights and grouping checks pass, and the configured model servers are healthy.
