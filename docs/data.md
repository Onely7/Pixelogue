# Images and Open Images V7

Pixelogue reads image bytes only after it has a source record and a matching rights record. A source record says where the image came from and whether it is for training or evaluation. A rights record says what processing and redistribution are allowed. The two records are joined by `rights_record_id`.

## Download the fixed Open Images validation sample

The committed [validation manifest](../validation/open_images_v7_manifest.jsonl) fixes 20 image IDs, attributions, licence URLs, landing pages, and raw SHA-256 hashes. The image files remain local because their redistribution terms are separate from this repository.

```sh
uv run --locked pixelogue prepare \
  --config configs/pilot.yaml \
  --destination data/open-images-v7
```

The command reads the official validation metadata and downloads pixels from the official Open Images S3 bucket. It verifies the committed provenance and byte hash. Expected files are:

- `sources.jsonl`: model-safe source records;
- `rights.jsonl`: evaluation-only rights records;
- `open_images_private_metadata.jsonl`: source URLs, titles, and operational metadata;
- `open_images_failures.jsonl`: deterministic failure reasons, empty after a complete run;
- `images/`: local image bytes.

Titles, labels, bounding boxes, localized narratives, and other annotations are never included in a model payload. Open Images annotations are not treated as answers to free-form questions.

## Validate and canonicalize images

```sh
uv run --locked pixelogue ingest \
  --config configs/pilot.yaml \
  --sources data/open-images-v7/sources.jsonl \
  --rights data/open-images-v7/rights.jsonl \
  --image-root data/open-images-v7 \
  --artifact-root artifacts/prepared-open-images
```

Pixelogue rejects path escapes, unavailable rights, unsupported formats, animation, files over 20 MiB, decoded images over 40 million pixels, and images whose short edge is below 128 pixels. It applies EXIF and Open Images counterclockwise rotation, converts colour to sRGB when an ICC profile is present, composites transparency on white, and writes a deterministic RGB PNG view.

Exact pixels and declared source groups are always grouped. To add SSCD near-copy grouping, install the optional visual environment and supply a local TorchScript model with its hash:

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

If any member of a visual group is evaluation-only, every member of that group becomes evaluation-only. `export` also rejects evaluation images directly, giving two independent barriers.

## Add another source

Create one JSON object per line in both manifests. Paths in a source record are relative to `--image-root`; absolute paths and `..` escapes are rejected. Use an ISO 8601 timestamp for `valid_from`. For training sources, both `training_allowed` and `qa_redistribution_allowed` must be true. Keep documentary evidence for the rights decision outside model prompts.

The Open Images site documents the [V7 validation split and manual download path](https://storage.googleapis.com/openimages/web/download_v7.html). Its [rotation note](https://storage.googleapis.com/openimages/web/2018-05-17-rotation-information.html) defines the metadata values as counterclockwise degrees.

## Prepare existing CVDF training images

`prepare-local-train` is a CPU-only adapter for an existing flat directory of
unmodified CVDF `<ImageID>.jpg` files and the official image metadata CSV. It does
not download images. The fixed validation `prepare` command remains evaluation-only.

```sh
uv run --locked pixelogue prepare-local-train \
  --metadata /work/datasets/openimages_v7/metadata/images.csv \
  --image-root /work/datasets/openimages_v7/images/cvdf/train \
  --destination /work/outputs/pixelogue/oi-train-input \
  --count 32 \
  --seed 20260916 \
  --evaluation-images artifacts/prepared-open-images/images.jsonl
```

Run from the repository root. `--evaluation-images` is optional and repeatable;
provide all available prepared evaluation manifests. The committed validation IDs
and raw hashes are always excluded using `--validation-manifest` (default:
`validation/open_images_v7_manifest.jsonl`). Reference manifests must contain only
evaluation-purpose records. Canonical pixel hashes from those manifests also exclude
exact copies with different IDs. **Near-copy detection is not performed** by this
command; `manifest.json` records that limitation. Before production export, perform
joint visual grouping with the evaluation collection using the pinned SSCD workflow
above, or supply an independently verified exclusion list through filtered metadata.
Separate `ingest` runs cannot discover cross-run visual groups automatically.

The command streams the CSV twice, with memory proportional to `--count`, and
selects the lowest SHA-256 ranks of `seed:ImageID` among eligible local images.
The second pass checks metadata stability and uniqueness of selected IDs. Selection
does not depend on CSV order. Missing files are probed only when their rank could
enter the sample; the missing-file counter is not a complete directory audit.
The output directory must not already exist.

Eligibility requires `Subset=train`, a 16-digit hexadecimal ID, an explicit CC BY
2.0/2.5/3.0/4.0 URL, author and HTTPS landing-page attribution, and known rotation.
Other licenses and missing rotation are excluded. Both integer and `.0` rotation
strings are accepted. CVDF removed EXIF without rotating pixels, so metadata rotation
is applied once during canonicalization; do not use this adapter on already rotated
images. The rights policy permits processing, training and QA redistribution while
preserving attribution, and does not grant image redistribution. License metadata
and titles are retained privately; they are never inference inputs.

Outputs include `sources.jsonl`, `rights.jsonl`, canonical `images/`, ready-to-use
`images.jsonl`, `failures.jsonl`, `selected-ids.json`, `private-metadata.jsonl`, and
`manifest.json` with input hashes, seed, counts, split assignments and exclusion scope.
The timestamp records preparation, so reruns have the same selected IDs and image
hashes, not byte-identical rights files. `--count` bounds candidates before image
validation; rejected images are recorded without silent replacement. Inspect the
accepted count before synthesis. No second `ingest` is needed.

Use the output directory as `synthesize --artifact-root` and its `images.jsonl` as
`--images`, with `configs/standard.yaml` after server checks. The synthesis limit
`data.target_dialogues` counts attempted images, not accepted dialogues; prepare
additional disjoint batches with new run IDs when more quality candidates are needed.
The current standard configuration generates English. Pilot output remains barred
from training export. Export writes image references, not copied image bytes; retain
the prepared image directory and private attribution records alongside the bundle.

Progress is shown on stderr by default: stage changes, CSV row counts about every
two seconds, image counts including rejections, and total elapsed seconds. The second
CSV pass and image processing include totals. No extra pass is made to count rows.
The final JSON report remains on stdout, so `> report.json` keeps it machine-readable;
use `2> progress.log` to save progress or `--quiet` to disable it. Updates are emitted
between rows/images; a single slow file operation may delay an update.
