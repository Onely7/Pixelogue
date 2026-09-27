"""Command-line boundary for Pixelogue operations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Annotated, Literal, cast

import typer

from pixelogue.capabilities import evaluate_capabilities
from pixelogue.config import load_config
from pixelogue.contracts import (
    ConversationArtifact,
    ImageArtifact,
    RightsRecord,
    SelectionManifest,
    SourceRecord,
)
from pixelogue.doctor import diagnose
from pixelogue.errors import (
    CapabilityError,
    ExternalInputError,
    PixelogueError,
    ShortfallError,
    SolverUnknownError,
)
from pixelogue.export import export_bundle
from pixelogue.fixtures import FixtureRecord, make_fixtures
from pixelogue.io import read_json, read_jsonl, write_json, write_jsonl
from pixelogue.local_train import prepare_local_train
from pixelogue.open_images import OpenImagesDownloader, OpenImagesPinnedRecord
from pixelogue.operations import (
    FrozenPool,
    PreparedDataset,
    compile_configuration,
    make_frozen_pool,
    prepare_sources,
    summarize_conversations,
    summarize_operations,
)
from pixelogue.pipeline import SynthesisCoordinator, SynthesisJob
from pixelogue.planner import exact_schedule
from pixelogue.profiling import profile_database
from pixelogue.progress import preparation_progress, synthesis_progress
from pixelogue.research_ablation import (
    AblationCase,
    AblationPlan,
    build_ablation_plan,
    run_ablation_plan,
    write_ablation_reports,
)
from pixelogue.research_audit import (
    Adjudication,
    AnswerBallot,
    AuditCase,
    AuditPack,
    QuestionBallot,
    build_audit_pack,
    resolve_audit,
    write_audit_pack,
    write_audit_resolution,
)
from pixelogue.research_exposure import (
    ExposureCase,
    ExposurePlan,
    build_exposure_plan,
    freeze_exposure_plan,
    run_exposure_trials,
    write_exposure_reports,
)
from pixelogue.research_history import HistoryStudyCase, run_history_study
from pixelogue.selection import (
    SelectionPolicy,
    audit_selection,
    bind_audit,
    select_candidates,
)
from pixelogue.serialization import canonical_hash
from pixelogue.serving import VllmClient
from pixelogue.specialist_evaluation import (
    SpecialistEvaluationCase,
    SpecialistEvaluationResult,
    build_calibration_manifest,
    run_specialist_evaluation,
)
from pixelogue.sscd import SscdEmbedder
from pixelogue.store import RunStore
from pixelogue.task_status import task_status_report, write_task_status_reports

app = typer.Typer(
    no_args_is_help=True,
    help="Build, evaluate, select, and export image-grounded dialogues.",
)

ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


@app.command("research-exposure")
def research_exposure_command(
    cases: Annotated[Path, typer.Option("--cases", exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option("--artifact-root", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    evaluator: Literal["generator_a", "generator_b"] = "generator_a",
    repetitions: Annotated[int, typer.Option(min=1, max=20)] = 1,
    seed: int = 20260915,
    plan_only: bool = False,
    retry_failed: bool = False,
) -> None:
    """Freeze and run research-only hidden, plausible and incorrect answer trials."""
    config = load_config(config_path)
    plan = build_exposure_plan(
        tuple(read_jsonl(cases, ExposureCase)), config, evaluator, repetitions, seed
    )
    freeze_exposure_plan(output_dir / "plan.json", plan)
    if plan_only:
        typer.echo(json.dumps({"plan_hash": plan.plan_hash, "trials": len(plan.trials)}))
        return
    stats = run_exposure_trials(plan, config, artifact_root, output_dir, retry_failed=retry_failed)
    report = write_exposure_reports(plan, output_dir)
    typer.echo(
        json.dumps(
            {
                "plan_hash": plan.plan_hash,
                "run": stats,
                "paired": report["paired_plausible_minus_hidden"],
            }
        )
    )


@app.command("research-exposure-report")
def research_exposure_report_command(
    output_dir: Annotated[Path, typer.Option("--output-dir", exists=True, file_okay=False)],
) -> None:
    """Rebuild the research report from saved trial records without model calls."""
    plan = read_json(output_dir / "plan.json", ExposurePlan)
    typer.echo(json.dumps(write_exposure_reports(plan, output_dir)))


@app.command("audit-pack")
def audit_pack_command(
    cases: Annotated[Path, typer.Option("--cases", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    rate: Annotated[float, typer.Option(min=0.001, max=1.0)] = 0.1,
    seed: int = 20260915,
) -> None:
    """Sample accepted, rejected and abstained outputs into blind local audit views."""
    pack = build_audit_pack(tuple(read_jsonl(cases, AuditCase)), rate=rate, seed=seed)
    write_audit_pack(pack, output_dir)
    typer.echo(
        json.dumps(
            {
                "pack_hash": pack.pack_hash,
                "frame_counts": pack.frame_counts,
                "selected_counts": pack.selected_counts,
            }
        )
    )


@app.command("audit-resolve")
def audit_resolve_command(
    pack_path: Annotated[Path, typer.Option("--pack", exists=True, dir_okay=False)],
    question_votes: Annotated[Path, typer.Option("--question-votes", exists=True, dir_okay=False)],
    answer_votes: Annotated[Path, typer.Option("--answer-votes", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    adjudications: Annotated[
        Path | None, typer.Option("--adjudications", exists=True, dir_okay=False)
    ] = None,
    required_raters: Annotated[int, typer.Option(min=2, max=10)] = 3,
) -> None:
    """Resolve independent human votes while preserving originals and unknowns."""
    pack = read_json(pack_path, AuditPack)
    report = resolve_audit(
        pack,
        tuple(read_jsonl(question_votes, QuestionBallot)),
        tuple(read_jsonl(answer_votes, AnswerBallot)),
        tuple(read_jsonl(adjudications, Adjudication)) if adjudications else (),
        required_raters=required_raters,
    )
    write_audit_resolution(report, output)
    typer.echo(json.dumps({"output": str(output), "items": len(pack.items)}))


@app.command("research-history")
def research_history_command(
    cases: Annotated[Path, typer.Option("--cases", exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option("--artifact-root", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    retry_failed: bool = False,
) -> None:
    """Check exact committed spans and two-stage counterfactual history witnesses."""
    report = run_history_study(
        tuple(read_jsonl(cases, HistoryStudyCase)),
        load_config(config_path),
        artifact_root,
        output_dir,
        retry_failed=retry_failed,
    )
    typer.echo(
        json.dumps(
            {
                "study_hash": report["study_hash"],
                "strong_dependency": report["strong_dependency"],
                "cases": report["cases"],
            }
        )
    )


@app.command("evaluate-specialist")
def evaluate_specialist_command(
    cases: Annotated[Path, typer.Option("--cases", exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option("--artifact-root", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    run_id: Annotated[str, typer.Option("--run-id")] = "specialist-evaluation",
    retry_failed: bool = False,
) -> None:
    """Exercise uncalibrated specialists on held-out inputs outside normal selection."""
    stats = run_specialist_evaluation(
        tuple(read_jsonl(cases, SpecialistEvaluationCase)),
        load_config(config_path),
        artifact_root,
        output_dir,
        run_id,
        retry_failed=retry_failed,
    )
    typer.echo(json.dumps(stats))


@app.command("calibration-build")
def calibration_build_command(
    results_dir: Annotated[Path, typer.Option("--results-dir", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", dir_okay=False)],
    report_path: Annotated[Path, typer.Option("--report", dir_okay=False)],
) -> None:
    """Compute exact model-bound confirmation certificates and pending-case counts."""
    results = tuple(
        read_json(path, SpecialistEvaluationResult) for path in sorted(results_dir.glob("*.json"))
    )
    manifest, report = build_calibration_manifest(results)
    write_json(output, manifest)
    write_json(report_path, report)
    typer.echo(json.dumps(report))


@app.command("research-ablation")
def research_ablation_command(
    cases: Annotated[Path, typer.Option("--cases", exists=True, dir_okay=False)],
    artifact_root: Annotated[Path, typer.Option("--artifact-root", exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir", file_okay=False)],
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    seed: int = 20260915,
    run_id: Annotated[str, typer.Option("--run-id")] = "research-ablation",
    plan_only: bool = False,
    retry_failed: bool = False,
) -> None:
    """Run the 18 fixed-question comparison cells outside training export."""
    config = load_config(config_path)
    plan = build_ablation_plan(tuple(read_jsonl(cases, AblationCase)), config, seed)
    plan_path = output_dir / "plan.json"
    if plan_path.exists() and read_json(plan_path, AblationPlan) != plan:
        raise ExternalInputError("ABLATION_PLAN_CHANGED", "Existing plan differs")
    write_json(plan_path, plan)
    if plan_only:
        typer.echo(json.dumps({"plan_hash": plan.plan_hash, "trials": len(plan.trials)}))
        return
    stats = run_ablation_plan(
        plan, config, artifact_root, output_dir, run_id=run_id, retry_failed=retry_failed
    )
    write_ablation_reports(plan, output_dir)
    typer.echo(json.dumps({"plan_hash": plan.plan_hash, "run": stats}))


@app.command("research-ablation-report")
def research_ablation_report_command(
    output_dir: Annotated[Path, typer.Option("--output-dir", exists=True, file_okay=False)],
) -> None:
    """Rebuild depth-aware factorial reports from completed research trials."""
    plan = read_json(output_dir / "plan.json", AblationPlan)
    typer.echo(json.dumps(write_ablation_reports(plan, output_dir)))


def _clients(
    config_path: Path, run_id: str, store: RunStore | None = None
) -> tuple[VllmClient, ...]:
    config = load_config(config_path)
    return (
        VllmClient(
            config.models.active_selector_endpoint,
            config.runtime,
            run_id=run_id,
            store=store,
        ),
        VllmClient(config.models.generator_a, config.runtime, run_id=run_id, store=store),
        VllmClient(config.models.generator_b, config.runtime, run_id=run_id, store=store),
    )


@app.command("compile")
def compile_command(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    output: Annotated[Path, typer.Option("--output", dir_okay=False)] = Path(
        "artifacts/compiled-plan.json"
    ),
) -> None:
    """Validate configuration and emit resolved catalogs and JSON Schemas."""
    compiled = compile_configuration(load_config(config_path))
    write_json(output, compiled)
    typer.echo(json.dumps({"output": str(output), "compiled_hash": compiled["compiled_hash"]}))


@app.command("task-status")
def task_status_command(
    config_path: ConfigOption = Path("configs/specialist-pilot.yaml"),
    junit: Annotated[Path | None, typer.Option("--junit", exists=True, dir_okay=False)] = None,
    output_stem: Annotated[Path, typer.Option("--output-stem")] = Path("artifacts/task-status"),
) -> None:
    """Report each task's validator, CPU evidence, environment and selection state."""
    report = task_status_report(load_config(config_path), junit)
    write_task_status_reports(report, output_stem)
    typer.echo(json.dumps({"output_stem": str(output_stem), "summary": report["summary"]}))


@app.command()
def prepare(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    destination: Annotated[Path, typer.Option("--destination", file_okay=False)] = Path(
        "data/open-images-v7"
    ),
) -> None:
    """Acquire the configured deterministic Open Images V7 validation sample."""
    config = load_config(config_path)
    settings = config.data.open_images
    if not settings.enabled or settings.groups == 0:
        typer.echo(json.dumps({"downloaded": 0, "reason": "open_images_disabled"}))
        return
    pinned = (
        tuple(read_jsonl(settings.image_ids_manifest, OpenImagesPinnedRecord))
        if settings.image_ids_manifest is not None
        else ()
    )
    sources, rights = OpenImagesDownloader().download_sample(
        str(settings.metadata_url),
        destination,
        count=settings.groups,
        seed=settings.seed,
        pinned=pinned,
    )
    write_jsonl(destination / "sources.jsonl", sources)
    write_jsonl(destination / "rights.jsonl", rights)
    typer.echo(json.dumps({"downloaded": len(sources), "destination": str(destination)}))


@app.command("prepare-local-train")
def prepare_local_train_command(
    metadata: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    image_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    destination: Annotated[Path, typer.Option(file_okay=False)],
    count: Annotated[int, typer.Option(min=1)] = 32,
    workers: Annotated[int, typer.Option(min=1, max=64)] = 1,
    quiet: Annotated[bool, typer.Option("--quiet", help="Hide progress on stderr.")] = False,
    seed: int = 20260915,
    validation_manifest: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "validation/open_images_v7_manifest.jsonl"
    ),
    evaluation_images: Annotated[
        list[Path] | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Prepare a bounded training sample from local, unmodified CVDF JPEGs."""
    with preparation_progress(sys.stderr, enabled=not quiet) as progress:
        report = prepare_local_train(
            metadata,
            image_root,
            destination,
            count=count,
            seed=seed,
            validation_manifest=validation_manifest,
            evaluation_images=evaluation_images or (),
            progress=progress,
            workers=workers,
        )
    typer.echo(json.dumps(report))


@app.command()
def doctor(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    check_servers: Annotated[bool, typer.Option("--check-servers")] = False,
    output: Annotated[Path | None, typer.Option("--output", dir_okay=False)] = None,
) -> None:
    """Inspect model locks, idle GPU capacity, and optional local servers."""
    report = diagnose(load_config(config_path), check_servers=check_servers)
    if output is not None:
        write_json(output, report)
    typer.echo(report.model_dump_json(indent=2))
    if not report.ready:
        raise CapabilityError("DOCTOR_NOT_READY", "One or more required checks failed")


@app.command("profile")
def profile_command(
    database: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
) -> None:
    """Summarize model-call latency and token use from a run database."""
    report = profile_database(database)
    if output is not None:
        write_json(output, report)
    typer.echo(report.model_dump_json(indent=2))


@app.command("make-fixtures")
def make_fixtures_command(
    destination: Annotated[Path, typer.Option("--destination", file_okay=False)] = Path(
        "data/fixtures"
    ),
    pairs_per_stratum: Annotated[int, typer.Option(min=1)] = 2,
    seed: int = 20260915,
) -> None:
    """Create reproducible positive and single-error negative image fixtures."""
    records = make_fixtures(destination, pairs_per_stratum=pairs_per_stratum, seed=seed)
    typer.echo(json.dumps({"records": len(records), "destination": str(destination)}))


@app.command("evaluate-capabilities")
def evaluate_capabilities_command(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    fixtures: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/fixtures/fixtures.jsonl"
    ),
    fixture_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/fixtures"
    ),
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/capability-report.json"
    ),
    run_id: Annotated[str, typer.Option("--run-id")] = "capability-evaluation",
) -> None:
    """Run both independent judges over answer-keyed capability fixtures."""
    config = load_config(config_path)
    with RunStore(
        config.storage.run_root,
        run_id,
        require_local_wal=config.storage.require_local_wal,
    ) as store:
        store.initialize_run(run_id, config.config_hash, config.profile)
        _, judge_a, judge_b = _clients(config_path, run_id, store)
        report = evaluate_capabilities(
            read_jsonl(fixtures, FixtureRecord),
            fixture_root,
            (judge_a, judge_b),
            seed=config.seed,
        )
    write_json(output, report)
    typer.echo(json.dumps({"outcomes": len(report.outcomes), "output": str(output)}))


@app.command()
def ingest(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    sources: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/open-images-v7/sources.jsonl"
    ),
    rights: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "data/open-images-v7/rights.jsonl"
    ),
    image_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "data/open-images-v7"
    ),
    artifact_root: Annotated[Path, typer.Option(file_okay=False)] = Path("artifacts/prepared"),
    sscd_model: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    sscd_sha256: Annotated[str | None, typer.Option()] = None,
    sscd_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.95,
) -> None:
    """Validate images and rights, then write canonical grouped image artifacts."""
    config = load_config(config_path)
    if (sscd_model is None) != (sscd_sha256 is None):
        raise typer.BadParameter("--sscd-model and --sscd-sha256 must be supplied together")
    embedder = None
    if sscd_model is not None and sscd_sha256 is not None:
        actual_hash = hashlib.sha256(sscd_model.read_bytes()).hexdigest()
        if actual_hash != sscd_sha256:
            raise typer.BadParameter("SSCD model SHA-256 does not match --sscd-sha256")
        embedder = SscdEmbedder(sscd_model)
    report = prepare_sources(
        read_jsonl(sources, SourceRecord),
        read_jsonl(rights, RightsRecord),
        image_root,
        artifact_root,
        seed=config.seed,
        embedder=embedder,
        similarity_threshold=sscd_threshold,
    )
    write_jsonl(artifact_root / "images.jsonl", report.images)
    write_jsonl(artifact_root / "failures.jsonl", report.failures)
    write_json(artifact_root / "manifest.json", report)
    typer.echo(
        json.dumps(
            {
                "accepted": len(report.images),
                "rejected": len(report.failures),
                "manifest_hash": report.manifest_hash,
            }
        )
    )


@app.command()
def synthesize(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    images: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/prepared/images.jsonl"
    ),
    artifact_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "artifacts/prepared"
    ),
    run_id: Annotated[str, typer.Option("--run-id")] = "pilot",
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/pilot/conversations.jsonl"
    ),
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers",
            min=1,
            max=64,
            help="Maximum images processed concurrently; defaults to runtime configuration.",
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Hide progress on stderr.")] = False,
) -> None:
    """Generate and independently rate bounded multi-turn conversations."""
    config = load_config(config_path)
    image_records = read_jsonl(images, ImageArtifact)
    prepared = read_json(artifact_root / "manifest.json", PreparedDataset)
    if tuple(image_records) != prepared.images:
        raise ExternalInputError(
            "PREPARED_INPUT_CHANGED", "Image list differs from rights-checked manifest"
        )
    run_contract_hash = canonical_hash(
        {
            "config_hash": config.config_hash,
            "prepared_manifest_hash": prepared.manifest_hash,
            "image_records": [image.model_dump(mode="json") for image in image_records],
        }
    )
    with RunStore(
        config.storage.run_root,
        run_id,
        require_local_wal=config.storage.require_local_wal,
    ) as store:
        store.initialize_run(run_id, run_contract_hash, config.profile)
        selector, generator_a, generator_b = _clients(config_path, run_id, store)
        coordinator = SynthesisCoordinator(
            config,
            run_id,
            store,
            selector,
            generator_a,
            generator_b,
        )
        scheduled_images = image_records[: config.data.target_dialogues]
        language_schedule = exact_schedule(
            len(scheduled_images),
            {target.tag: target.weight for target in config.data.languages},
            seed=config.seed,
            namespace="languages",
        )
        generator_schedule = exact_schedule(
            len(scheduled_images),
            {str(role): weight for role, weight in config.models.generation_allocation.items()},
            seed=config.seed,
            namespace="generators",
        )
        jobs = (
            SynthesisJob(
                image=image,
                target_language=cast(Literal["en", "ja", "zh-Hans"], language),
                generator_role=cast(Literal["generator_a", "generator_b"], generator_role),
            )
            for image, language, generator_role in zip(
                scheduled_images,
                language_schedule,
                generator_schedule,
                strict=True,
            )
        )
        worker_count = workers or config.runtime.max_concurrent_images
        conversations: list[ConversationArtifact] = []
        output.parent.mkdir(parents=True, exist_ok=True)
        with (
            output.open("w", encoding="utf-8") as stream,
            synthesis_progress(
                len(scheduled_images), worker_count, sys.stderr, enabled=not quiet
            ) as record_progress,
        ):
            try:
                for conversation in coordinator.synthesize_batch(
                    jobs,
                    artifact_root,
                    max_workers=worker_count,
                ):
                    stream.write(conversation.model_dump_json() + "\n")
                    stream.flush()
                    conversations.append(conversation)
                    if len(conversations) == 1 or len(conversations) % 100 == 0:
                        write_json(
                            output.with_suffix(".summary.json"),
                            summarize_conversations(conversations),
                        )
                    record_progress(conversation.status)
            finally:
                write_json(
                    output.with_suffix(".summary.json"), summarize_conversations(conversations)
                )
                write_json(
                    output.with_suffix(".operations.json"), summarize_operations(conversations)
                )
    typer.echo(json.dumps({"conversations": len(conversations), "output": str(output)}))


@app.command("rate-existing")
def rate_existing(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    conversations: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/pilot/conversations.jsonl"
    ),
    artifact_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "artifacts/prepared"
    ),
    run_id: Annotated[str, typer.Option("--run-id")] = "rerate",
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path(
        "artifacts/pilot/rated-conversations.jsonl"
    ),
) -> None:
    """Re-evaluate saved questions and answers without changing public text."""
    config = load_config(config_path)
    existing = read_jsonl(conversations, ConversationArtifact)
    with RunStore(
        config.storage.run_root,
        run_id,
        require_local_wal=config.storage.require_local_wal,
    ) as store:
        store.initialize_run(run_id, config.config_hash, config.profile)
        selector, generator_a, generator_b = _clients(config_path, run_id, store)
        coordinator = SynthesisCoordinator(
            config,
            run_id,
            store,
            selector,
            generator_a,
            generator_b,
        )
        rated = [coordinator.rate_existing(item, artifact_root) for item in existing]
    write_jsonl(output, rated)
    write_json(output.with_suffix(".summary.json"), summarize_conversations(rated))
    write_json(output.with_suffix(".operations.json"), summarize_operations(rated))
    typer.echo(json.dumps({"conversations": len(rated), "output": str(output)}))


@app.command("freeze-pool")
def freeze_pool_command(
    conversations: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path("artifacts/frozen-pool.json"),
) -> None:
    """Freeze fully accepted conversations into solver-visible metadata."""
    pool = make_frozen_pool(read_jsonl(conversations, ConversationArtifact))
    write_json(output, pool)
    typer.echo(json.dumps({"candidates": len(pool.candidates), "pool_hash": pool.pool_hash}))


@app.command()
def select(
    pool_path: Annotated[Path, typer.Option("--pool", exists=True, dir_okay=False)],
    policy_path: Annotated[Path, typer.Option("--policy", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path("artifacts/selection.json"),
) -> None:
    """Select a constrained subset with deterministic CP-SAT settings."""
    pool = read_json(pool_path, FrozenPool)
    policy = read_json(policy_path, SelectionPolicy)
    manifest = select_candidates(pool.candidates, policy)
    if manifest.pool_hash != pool.pool_hash:
        raise typer.BadParameter("Frozen pool identity changed")
    write_json(output, manifest)
    if manifest.solver_status == "INFEASIBLE":
        raise ShortfallError("SELECTION_INFEASIBLE", "Frozen pool cannot meet the policy")
    if manifest.solver_status == "UNKNOWN":
        raise SolverUnknownError("SELECTION_UNKNOWN", "Solver did not establish a result")
    typer.echo(manifest.model_dump_json())


@app.command()
def audit(
    pool_path: Annotated[Path, typer.Option("--pool", exists=True, dir_okay=False)],
    policy_path: Annotated[Path, typer.Option("--policy", exists=True, dir_okay=False)],
    selection_path: Annotated[Path, typer.Option("--selection", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option(dir_okay=False)] = Path("artifacts/audit.json"),
) -> None:
    """Independently recompute every implemented selection condition."""
    pool = read_json(pool_path, FrozenPool)
    policy = read_json(policy_path, SelectionPolicy)
    manifest = read_json(selection_path, SelectionManifest)
    report = audit_selection(pool.candidates, manifest, policy)
    bound = bind_audit(manifest, report)
    write_json(output, report)
    write_json(selection_path, bound)
    typer.echo(report.model_dump_json())


@app.command()
def export(
    config_path: ConfigOption = Path("configs/standard.yaml"),
    conversations: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "artifacts/conversations.jsonl"
    ),
    selection_path: Annotated[
        Path, typer.Option("--selection", exists=True, dir_okay=False)
    ] = Path("artifacts/selection.json"),
    destination: Annotated[Path, typer.Option(file_okay=False)] = Path("artifacts/export"),
) -> None:
    """Write training content separately from ratings and provenance."""
    config = load_config(config_path)
    outputs = export_bundle(
        read_jsonl(conversations, ConversationArtifact),
        read_json(selection_path, SelectionManifest),
        destination,
        profile=config.profile,
        student_processor_lock={
            "repo_id": config.student_view.processor_repo_id,
            "revision": config.student_view.processor_revision,
            "min_pixels": config.student_view.min_pixels,
            "max_pixels": config.student_view.max_pixels,
        },
    )
    typer.echo(json.dumps({key: str(path) for key, path in outputs.items()}))


@app.command()
def replay(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    run_id: Annotated[str, typer.Option("--run-id")] = "pilot",
) -> None:
    """Verify SQLite integrity and every stored request, response, and turn artifact."""
    config = load_config(config_path)
    with RunStore(
        config.storage.run_root,
        run_id,
        require_local_wal=config.storage.require_local_wal,
    ) as store:
        typer.echo(json.dumps(store.verify(), sort_keys=True))


@app.command()
def backup(
    config_path: ConfigOption = Path("configs/pilot.yaml"),
    run_id: Annotated[str, typer.Option("--run-id")] = "pilot",
    destination: Annotated[Path, typer.Option(file_okay=False)] = Path("backups"),
) -> None:
    """Create a consistent database and immutable-artifact backup."""
    config = load_config(config_path)
    with RunStore(
        config.storage.run_root,
        run_id,
        require_local_wal=config.storage.require_local_wal,
    ) as store:
        database = store.backup(destination)
    typer.echo(json.dumps({"database": str(database), "destination": str(destination)}))


@app.command()
def restore(
    database: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    artifacts: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    destination: Annotated[Path, typer.Option(file_okay=False)],
) -> None:
    """Restore a backup into a new empty local run directory."""
    RunStore.restore(database, artifacts, destination)
    typer.echo(json.dumps({"restored": str(destination)}))


def main() -> None:
    """Run the CLI and render application failures as stable JSON diagnostics."""
    try:
        app()
    except PixelogueError as error:
        typer.echo(json.dumps({"reason": error.reason, "message": str(error)}), err=True)
        raise SystemExit(int(error.exit_code)) from None


if __name__ == "__main__":
    main()
