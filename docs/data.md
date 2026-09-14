# Images and Open Images V7

Pixelogue reads image bytes only after it has a source record and a matching rights record. A
source record says where the image came from and whether it is for training or evaluation. A rights
record says what processing and redistribution are allowed. The two records are joined by
`rights_record_id`.

## Download the fixed Open Images validation sample

The committed [validation manifest](../validation/open_images_v7_manifest.jsonl) fixes 20 image IDs,
attributions, licence URLs, landing pages, and raw SHA-256 hashes. The image files remain local
because their redistribution terms are separate from this repository.

```sh
uv run --locked pixelogue prepare \
  --config configs/pilot.yaml \
  --destination data/open-images-v7
```

The command reads the official validation metadata and downloads pixels from the official Open
Images S3 bucket. It verifies the committed provenance and byte hash. Expected files are:

- `sources.jsonl`: model-safe source records;
- `rights.jsonl`: evaluation-only rights records;
- `open_images_private_metadata.jsonl`: source URLs, titles, and operational metadata;
- `open_images_failures.jsonl`: deterministic failure reasons, empty after a complete run;
- `images/`: local image bytes.

Titles, labels, bounding boxes, localized narratives, and other annotations are never included in a
model payload. Open Images annotations are not treated as answers to free-form questions.

## Validate and canonicalize images

```sh
uv run --locked pixelogue ingest \
  --config configs/pilot.yaml \
  --sources data/open-images-v7/sources.jsonl \
  --rights data/open-images-v7/rights.jsonl \
  --image-root data/open-images-v7 \
  --artifact-root artifacts/prepared-open-images
```

Pixelogue rejects path escapes, unavailable rights, unsupported formats, animation, files over 20
MiB, decoded images over 40 million pixels, and images whose short edge is below 128 pixels. It
applies EXIF and Open Images counterclockwise rotation, converts colour to sRGB when an ICC profile
is present, composites transparency on white, and writes a deterministic RGB PNG view.

Exact pixels and declared source groups are always grouped. To add SSCD near-copy grouping, install
the optional visual environment and supply a local TorchScript model with its hash:

```sh
uv sync --locked --extra cpu --extra visual --group dev
uv run --locked pixelogue ingest \
  --config configs/pilot.yaml \
  --sources path/to/sources.jsonl \
  --rights path/to/rights.jsonl \
  --image-root path/to/images \
  --artifact-root artifacts/prepared \
  --sscd-model models/sscd.torchscript.pt \
  --sscd-sha256 FULL_64_CHARACTER_SHA256
```

If any member of a visual group is evaluation-only, every member of that group becomes
evaluation-only. `export` also rejects evaluation images directly, giving two independent barriers.

## Add another source

Create one JSON object per line in both manifests. Paths in a source record are relative to
`--image-root`; absolute paths and `..` escapes are rejected. Use an ISO 8601 timestamp for
`valid_from`. For training sources, both `training_allowed` and `qa_redistribution_allowed` must be
true. Keep documentary evidence for the rights decision outside model prompts.

The Open Images site documents the [V7 validation split and manual download
path](https://storage.googleapis.com/openimages/web/download_v7.html). Its [rotation
note](https://storage.googleapis.com/openimages/web/2018-05-17-rotation-information.html) defines
the metadata values as counterclockwise degrees.
