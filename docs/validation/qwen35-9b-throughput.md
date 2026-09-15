# Qwen3.5-9B throughput validation

Pixelogue processed the same eight images with one worker and four workers on one GPU. Four-image concurrency reduced the interval from the first saved model response to the last from 745 to 296 seconds, a 2.52× speedup and a 60.3% reduction. This result supports bounded image-level concurrency for the diagnostic pilot. It does not establish performance for the standard Qwen3.8-27B and Gemma 4 31B configuration.

## Experiment status

Both benchmark runs completed on 15 September 2026. Each run wrote eight final conversation artifacts. Replay verification still succeeds for both run stores. No Pixelogue model server or synthesis process remained active when the progress was checked later that day; the tmux session contained only an idle control shell.

The implementation measured here was introduced in commit `749ff15`:

- up to `runtime.max_concurrent_images` independent images can be in flight;
- turns within one conversation remain sequential;
- only a bounded number of futures are submitted;
- completed conversations are returned in input order;
- SQLite and artifact mutations use short serialized sections, while model HTTP requests run outside the store lock;
- each model call records its HTTP duration;
- `pixelogue profile` summarizes calls, tokens, and duration by stage and model.

The intended effect is to give vLLM several independent requests that it can process with continuous batching. The generation and evaluation contracts, repair limit, and acceptance rules are unchanged by the worker count.

## Setup

| Item | Value |
|---|---|
| Input | The same eight evaluation-only procedural images in both runs |
| Configuration hash | `3d43f3295a9d2c984ed32a6e9437c87724050f7e3df12955126ff74082c77e93` |
| Instruction selector | `Qwen/Qwen3.5-2B` |
| Generator and both logical evaluator roles | `Qwen/Qwen3.5-9B` through one shared endpoint |
| GPU | One NVIDIA RTX 6000 Ada Generation |
| Weight format | BF16, without quantization |
| Serial run | `qwen35-9b-perf-serial-8-20260915`, `--workers 1` |
| Concurrent run | `qwen35-9b-perf-parallel-8-20260915`, `--workers 4` |
| Timing source | Response-artifact timestamps and recorded HTTP duration |

The model servers were already loaded. The primary 745- and 296-second intervals run from the first to the last saved response artifact, so they exclude model download, model loading, fixture creation, and server startup. Both roles A and B made separate blind requests, but they used the same Qwen3.5-9B checkpoint. This tests pipeline execution and scheduling rather than evaluator-model diversity.

Pixelogue's budget ledger counted every HTTP attempt, while the `model_call` table retained one row for each model-lock, stage, and request-hash identity. Identical repeated requests therefore share a row. The `profile` command reported 528 and 529 completed identities and ended the concurrent interval at 286 seconds, before a repeated response saved 10 seconds later. The all-response interval is the primary result here; the profile's 2.60× result remains useful for comparing its stage records.

## Results

| Measurement | One worker | Four workers |
|---|---:|---:|
| All-response interval | 745 s | 296 s |
| Relative speed | 1.00× | 2.52× |
| HTTP response artifacts | 550 | 544 |
| Response artifacts per minute | 44.30 | 110.27 |
| Images per minute | 0.64 | 1.62 |
| Measured seconds per image | 93.1 | 37.0 |
| Prompt tokens in all responses | 309,011 | 308,216 |
| Completion tokens in all responses | 37,947 | 37,940 |
| Profiled completed call identities | 528 | 529 |
| Profile interval | 745 s | 286 s |
| Profiled call identities per minute | 42.52 | 110.98 |
| Profiled summed HTTP duration | 717.5 s | 866.9 s |
| Profiled rubric identities | 406 (76.9%) | 402 (76.0%) |
| Committed turns | 7 | 8 |

Profiled summed HTTP duration can exceed elapsed time when requests overlap. Its increase in the four-worker run shows that individual requests experienced more contention, while the much shorter elapsed interval shows that overlap more than compensated for it. Token and response counts in the upper part of the table include repeated and invalid responses; profile rows and their stage breakdown include only retained `COMPLETE` identities.

The completed profile-identity counts differ by one, or about 0.2%, and the total HTTP attempt counts differ by six. The models followed different generation and evaluation paths even though the image set and configuration hash matched. The serial run ended with one `QUALITY_CANDIDATE`, five `REJECTED`, and two `ABSTAINED` conversations. The concurrent run ended with one `QUALITY_CANDIDATE`, four `REJECTED`, two `ABSTAINED`, and one `ERROR` conversation. The comparison therefore measures two closely matched operational runs, not a bit-for-bit identical request trace.

## Bottleneck

Rubric evaluation remained the main cost in the retained profile. It represented about 76% of completed call identities in both runs. In the serial run, its 406 identities consumed 432.1 seconds of profiled HTTP time. Requirement extraction, claim extraction, question fit, evidence extraction, and complete-set extraction were the next largest stages.

The earlier 25-image observation showed the same pattern: vLLM usually had only zero or one running request and no queued request, while the GPU was busy. The limiting factor was the controller waiting for many small, sequential evaluator calls. Image decoding, SQLite work, and final JSONL writing were not dominant at pilot scale.

This explains why image-level concurrency helped. Pixelogue cannot reorder dependent turns within one conversation, but it can advance another image while the first image waits for its next evaluator response.

## Failure and integrity checks

The parallel run recorded two invalid structured responses and one terminal `ERROR`. The model omitted a required rubric reason twice; no HTTP transport failure or GPU-memory failure caused it. A later bounded structured-output retry change was tested separately on the eight failed pilot images: three invalid calls recovered and no conversation ended in `ERROR`. That reliability rerun is not part of this timing comparison.

Replay produced the following results:

| Run | Hashed artifacts | Model-call rows | Committed turns |
|---|---:|---:|---:|
| One worker | 1,094 | 528 | 7 |
| Four workers | 1,092 | 531, including 2 invalid calls | 8 |

SQLite `integrity_check` returned `ok`, and `foreign_key_check` returned no rows for both databases. Automated tests cover overlapping requests, input-order output, model-call duration recording, and serialized run-store access. The coordinator itself limits the submitted work to the configured worker count.

Both `run` rows still say `CREATED`; this version does not update a run-level terminal status. Completion was established from eight final conversation artifacts and successful replay rather than that field. A future run-status implementation should make this progress check direct.

## Improvements after this benchmark

The filtering review in commit `4c43dc9` removed evaluator calls whose rubric conditions did not apply. Applied to the final artifacts of the separate 50-image pilot, the corrected conditions avoid at least 78 irrelevant evaluator calls. Four normalized exact question repeats would also stop before eight question-fit calls. These changes should improve throughput further, but they were not rerun in this eight-image benchmark and are excluded from the reported speedup.

Rubric request batching remains a possible next experiment. It must group only criteria with identical permitted input scopes; otherwise a batched request could expose an answer, image, claim, or private context to a criterion that is not allowed to receive it. At 30,000-image scale, rewriting the growing conversation JSONL after every image should also be profiled again.

## Limits of the result

- Each worker setting was measured once, so there is no confidence interval or run-to-run variance estimate.
- The sample contains eight procedural images and no Open Images examples.
- One Qwen3.5-9B checkpoint filled both evaluator roles, so the result does not validate the standard model pair or evaluator diversity.
- The measurement excludes model loading and server startup.
- The two runs produced different terminal outcomes, HTTP attempt counts, and completed profile identities.
- Stage timings omit identical repeated requests because this schema retains one `model_call` row per request identity.
- The timestamps have one-second resolution, and no repeated measurements were made.
- Faster execution does not establish conversation quality. The separate [pilot validation report](qwen35-9b-pilot.md) records the manual quality findings.

## Reproduce the comparison

Long GPU work should run in tmux. First inspect GPU use and run `doctor` as described in the [model and GPU guide](../models-and-gpu.md). Use fresh run IDs and output paths for each run.

```sh
nvidia-smi
tmux new-session -s pixelogue-throughput

uv run --locked pixelogue synthesize \
  --config configs/pilot.yaml \
  --images PATH_TO_EIGHT_IMAGE_MANIFEST.jsonl \
  --artifact-root PATH_TO_PREPARED_ROOT \
  --run-id throughput-serial-YYYYMMDD \
  --output artifacts/throughput-serial/conversations.jsonl \
  --workers 1

uv run --locked pixelogue synthesize \
  --config configs/pilot.yaml \
  --images PATH_TO_EIGHT_IMAGE_MANIFEST.jsonl \
  --artifact-root PATH_TO_PREPARED_ROOT \
  --run-id throughput-parallel-YYYYMMDD \
  --output artifacts/throughput-parallel/conversations.jsonl \
  --workers 4
```

Generate the machine-readable profiles and verify both stores after the runs finish:

```sh
uv run --locked pixelogue profile \
  --database /var/tmp/pixelogue/throughput-serial-YYYYMMDD/run.sqlite3 \
  --output artifacts/throughput-serial/profile.json
uv run --locked pixelogue profile \
  --database /var/tmp/pixelogue/throughput-parallel-YYYYMMDD/run.sqlite3 \
  --output artifacts/throughput-parallel/profile.json

uv run --locked pixelogue replay \
  --config configs/pilot.yaml \
  --run-id throughput-serial-YYYYMMDD
uv run --locked pixelogue replay \
  --config configs/pilot.yaml \
  --run-id throughput-parallel-YYYYMMDD
```

Do not reuse the two historical run IDs. Preserve each store as the record of its original configuration and responses.
