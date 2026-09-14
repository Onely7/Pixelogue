"""Command-line boundary for Pixelogue operations."""

from __future__ import annotations

import hashlib
import json
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
    PixelogueError,
    ShortfallError,
    SolverUnknownError,
)
from pixelogue.export import export_bundle
from pixelogue.fixtures import FixtureRecord, make_fixtures
from pixelogue.io import read_json, read_jsonl, write_json, write_jsonl
from pixelogue.open_images import OpenImagesDownloader, OpenImagesPinnedRecord
from pixelogue.operations import (
    FrozenPool,
    compile_configuration,
    make_frozen_pool,
    prepare_sources,
    summarize_conversations,
)
from pixelogue.pipeline import SynthesisCoordinator, SynthesisJob
from pixelogue.planner import exact_schedule
from pixelogue.profiling import profile_database
from pixelogue.selection import (
    SelectionPolicy,
    audit_selection,
    bind_audit,
    select_candidates,
)
from pixelogue.serving import VllmClient
from pixelogue.sscd import SscdEmbedder
from pixelogue.store import RunStore

app = typer.Typer(
    no_args_is_help=True,
    help="Build, evaluate, select, and export image-grounded dialogues.",
)

ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


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
            max=32,
            help="Maximum images processed concurrently; defaults to runtime configuration.",
        ),
    ] = None,
) -> None:
    """Generate and independently rate bounded multi-turn conversations."""
    config = load_config(config_path)
    image_records = read_jsonl(images, ImageArtifact)
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
        for conversation in coordinator.synthesize_batch(
            jobs,
            artifact_root,
            max_workers=worker_count,
        ):
            conversations.append(conversation)
            write_jsonl(output, conversations)
            write_json(output.with_suffix(".summary.json"), summarize_conversations(conversations))
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
