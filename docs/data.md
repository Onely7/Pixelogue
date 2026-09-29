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

## Diverse web evaluation sample

The [pinned Commons manifest](../validation/diverse_web_eval_manifest.jsonl) records one visually checked raster for each of 60 broad image categories, including documents, tables, charts, diagrams, medical images, screenshots, and photographs. Image bytes stay under ignored `data/`; the committed file records titles, source pages, thumbnail hashes, dimensions, and attribution. These are **evaluation-only inputs**, with no independent answer labels. A model's generated question or answer does not become a ground-truth label.

```sh
uv run --locked python validation/fetch_diverse_eval.py
uv run --locked pixelogue ingest \
  --config configs/gpu-watch-diverse.yaml \
  --sources data/diverse-web-eval/sources.jsonl \
  --rights data/diverse-web-eval/rights.jsonl \
  --image-root data/diverse-web-eval \
  --artifact-root artifacts/prepared-diverse-web-eval
```

The fetcher verifies the pinned Commons page ID, thumbnail SHA-256, and recorded licence URI. `ingest` still applies the normal image and permission checks; inspect `failures.jsonl` and require 60 accepted images before synthesis. Category 40 is a *static contact sheet* representing multiple frames because animated images are deliberately unsupported by ingestion. The category names and Commons descriptions are retained outside model inputs. A separate calibration set with independent human labels is still required to estimate verifier accuracy.

The diverse report lists and counts only `COMMITTED` turn tasks. A rejected or abstained terminal attempt remains in the diagnostic stop record and is excluded from committed task counts.

## Separate six-image holdout smoke set

The [holdout manifest](../validation/holdout_web_eval_manifest.jsonl) pins six other Commons pages: a photograph, document, table, chart, flowchart, and app screen. Their page IDs, pixel hashes, and visual groups differ from the 60 development images and the fixed Open Images sample. The fetcher restores evaluation-only source and rights records; image bytes stay out of Git.

```sh
uv run --locked python validation/fetch_holdout_eval.py
uv run --locked pixelogue ingest \
  --config configs/gpu-watch-diverse.yaml \
  --sources data/holdout-web-eval/sources.jsonl \
  --rights data/holdout-web-eval/rights.jsonl \
  --image-root data/holdout-web-eval \
  --artifact-root artifacts/prepared-holdout-web-eval
```

All six images must be accepted before synthesis. These few images check behavior across formats; they do not estimate a general success rate or provide human answer labels. The initial six were inspected to diagnose pipeline failures, so subsequent changes treat them as development examples.

The [second holdout manifest](../validation/holdout_web_eval_v2_manifest.jsonl) fixes six different Commons pages in the same six categories. Its page IDs and thumbnail hashes are checked against both earlier Commons sets. Keep this cohort untouched until implementation choices are frozen; do not use its model outputs to tune the same comparison.

```sh
uv run --locked python validation/fetch_holdout_eval.py --cohort v2
uv run --locked pixelogue ingest \
  --config configs/gpu-watch-diverse.yaml \
  --sources data/holdout-web-eval-v2/sources.jsonl \
  --rights data/holdout-web-eval-v2/rights.jsonl \
  --image-root data/holdout-web-eval-v2 \
  --artifact-root artifacts/prepared-holdout-web-eval-v2
```

These images are evaluation-only; image bytes and generated records remain outside Git. This six-image smoke set gives format-specific observations, not a population success rate or independent human labels.

Three additional score, geometry, and circuit diagrams are pinned in the [specialist evaluation manifest](../validation/specialist_web_eval_manifest.jsonl). Restore their source and rights records with `uv run --locked python validation/fetch_specialist_eval.py`, then ingest `data/specialist-web-eval` and `data/specialist-extra-eval` as separate evaluation-only roots. The fetcher verifies each byte hash and image size. These development images and alternate candidate answers do not constitute independent calibration samples.

## PubChem structure evaluation sample

The [pinned PubChem manifest](../validation/pubchem_2d_eval_manifest.jsonl) identifies 89 distinct, neutral, acyclic 2D depictions with three to five non-hydrogen atoms from C, N, O, F, Cl, and Br. Ten images form a development split; 79 separate molecular structures form the confirmation split. The source's SMILES property supplies a private controller-side label for each depiction. This is a limited digital-depiction domain, not evidence about hand-drawn or arbitrary chemical images. The image bytes and answer labels remain outside Git.

```sh
uv sync --locked --directory runtime/validators
runtime/validators/.venv/bin/python validation/fetch_pubchem_chemical_eval.py
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/pubchem-2d-simple-eval/sources.jsonl \
  --rights data/pubchem-2d-simple-eval/rights.jsonl \
  --image-root data/pubchem-2d-simple-eval \
  --artifact-root artifacts/prepared-pubchem-2d-simple-eval
uv run --locked python validation/build_pubchem_chemical_cases.py
```

The fetcher checks the committed CID, image SHA-256, dimensions, source group and SMILES hash before restoring local files. It writes `private_labels.jsonl` only under `data/`; the case builder uses those labels to freeze 20 correct and 59 incorrect confirmation answers. Model requests carry the image, public operation and question, but never a CID, the source SMILES or the expected verdict. Calibration is still conditional on actual model results and the stated confidence bounds.

## ScreenSpot UI click evaluation sample

The [pinned ScreenSpot manifest](../validation/screenspot_ui_eval_manifest.jsonl) identifies 89 distinct screenshots with human annotated text targets: 10 for development and 79 for confirmation. The [ScreenSpot test split](https://huggingface.co/datasets/bevaya/ScreenSpot) supplies the screenshot, instruction and target box. Target boxes stay in the private local label file and are never included in model requests. This sample assesses a single specified click on a visible text target, not arbitrary GUI actions or live interaction.

```sh
uv run --locked python validation/fetch_screenspot_ui_eval.py
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/screenspot-ui-eval/sources.jsonl \
  --rights data/screenspot-ui-eval/rights.jsonl \
  --image-root data/screenspot-ui-eval \
  --artifact-root artifacts/prepared-screenspot-ui-eval
uv run --locked python validation/build_screenspot_ui_cases.py
```

The fetcher checks image and annotation hashes against the committed manifest. Selection also excluded screenshots with a 64-bit difference-hash distance at most five; this screen is not a complete near-copy audit. The case builder freezes 20 annotated-center clicks and 59 off-target clicks on separate confirmation screenshots. ScreenSpot images remain evaluation-only and outside Git. Model extraction and verifier outcomes must still satisfy the stated calibration bounds before normal selection.

The separate [ScreenSpot holdout manifest](../validation/screenspot_ui_holdout_manifest.jsonl) pins 150 additional confirmation screenshots after the first confirmation sample was inspected during development. It excludes matching pixels and difference hashes within distance five across the two sets. Freeze it only once with `--freeze --holdout`; subsequent restores use `--holdout` alone. Its private case builder fixes 50 annotated-center clicks and 100 off-target clicks before evaluation.

```sh
uv run --locked python validation/fetch_screenspot_ui_eval.py --holdout
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/screenspot-ui-holdout/sources.jsonl \
  --rights data/screenspot-ui-holdout/rights.jsonl \
  --image-root data/screenspot-ui-holdout \
  --artifact-root artifacts/prepared-screenspot-ui-holdout
uv run --locked python validation/build_screenspot_ui_cases.py --holdout
```

## PrIMuS music notation evaluation sample

The [pinned PrIMuS manifest](../validation/primus_music_eval_manifest.jsonl) identifies 89 incipits from different work IDs in the [official PrIMuS archive](https://grfia.dlsi.ua.es/primus/). Each has a printed score and an independently generated MEI label. This first calibration domain covers only a complete first measure, one voice, treble or bass clef, no key signature, and supported note/rest durations without accidentals, ties or tuplets. The score strip is enlarged exactly twofold before ingestion. The original archive, images and MEI labels remain outside Git and never enter model requests.

```sh
uv sync --locked --directory runtime/validators
uv run --locked python validation/fetch_primus_music_eval.py
uv run --locked pixelogue ingest \
  --config configs/paired-one-gpu-pilot.yaml \
  --sources data/primus-music-eval/sources.jsonl \
  --rights data/primus-music-eval/rights.jsonl \
  --image-root data/primus-music-eval \
  --artifact-root artifacts/prepared-primus-music-eval
uv run --locked python validation/build_primus_music_cases.py
```

The fetcher checks the source archive, MEI, original PNG, transformed PNG and derived label hashes. The case builder freezes 20 exact transcriptions and 59 adjacent-note errors on separate confirmation works. Its label consistency check does not measure visual reading accuracy; independent model readings and the calibration bounds determine normal selection. The rights record points to the source page under the operator's evaluation-only testing authorization and does not assert a reuse license for the scores.

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
  --workers 4 \
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

Use `--workers 4` to process up to four images concurrently (default: 1, maximum: 64).
Only image reading, decoding, rotation, colour conversion and PNG writing are parallel;
CSV sampling/verification and visual grouping remain serial. Result ordering, grouping,
and image hashes match the serial path. Progress counts completed images, including
rejections, even if an earlier image is still running. More workers use more memory and
storage bandwidth; throughput depends on the filesystem and image sizes.

Both preparation and synthesis use tqdm bars on stderr. Preparation displays one bar
per stage with counts, elapsed time, rate and estimated remaining time when the total
is known. The first CSV pass has no known total; the second pass uses its row count.
The final JSON report remains on stdout: use `> report.json`, `2> progress.log`, or
`--quiet` to disable the bars. No extra CSV pass is made to count rows.

Canonical PNG output now drops inherited EXIF and ICC metadata after conversion to
sRGB, avoiding timestamps in generated profiles. Encoded hashes for previously prepared
images with such metadata can change; use a new preparation directory and run ID.
A single CPU check on 31 local images took 6.65 s with one worker and 1.87 s with four,
with identical manifests. This excludes CSV scans and is not a repeated benchmark.
