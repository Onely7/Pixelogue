from pathlib import Path

import pytest

from pixelogue.errors import ExternalInputError
from pixelogue.fixtures import STRATA, make_fixtures
from pixelogue.store import RunStore


def test_store_detects_configuration_change_and_artifact_corruption(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    with RunStore(root, "run", require_local_wal=False) as store:
        store.initialize_run("run", "a" * 64, "pilot")
        artifact_hash = store.write_json_artifact("test", {"value": 1})
        assert store.verify()["artifacts"] == 1
        with pytest.raises(ExternalInputError) as caught:
            store.initialize_run("run", "b" * 64, "pilot")
        assert caught.value.reason == "RUN_CONFIG_MISMATCH"
        row = store.connection.execute(
            "SELECT relative_path FROM artifact WHERE artifact_hash = ?", (artifact_hash,)
        ).fetchone()
        (store.artifact_dir / row["relative_path"]).write_bytes(b"changed")
        with pytest.raises(ExternalInputError) as caught:
            store.verify()
        assert caught.value.reason == "ARTIFACT_HASH_MISMATCH"


def test_backup_and_restore_use_consistent_layout(tmp_path: Path) -> None:
    with RunStore(tmp_path / "runs", "run", require_local_wal=False) as store:
        store.initialize_run("run", "a" * 64, "pilot")
        store.write_json_artifact("test", {"value": 1})
        database = store.backup(tmp_path / "backup")
    RunStore.restore(
        database,
        tmp_path / "backup" / "run-artifacts",
        tmp_path / "restored",
    )
    with RunStore(tmp_path, "restored", require_local_wal=False) as restored:
        assert restored.verify()["artifacts"] == 1


def test_fixture_generator_covers_all_strata_and_splits(tmp_path: Path) -> None:
    records = make_fixtures(tmp_path, pairs_per_stratum=1)
    assert len(records) == len(STRATA) * 2 * 2
    assert {record.stratum for record in records} == set(STRATA)
    assert {record.split for record in records} == {"development", "confirmation"}
    assert all((tmp_path / record.image_path).is_file() for record in records)


def test_request_budget_reservations_survive_failure_and_finalize_success(tmp_path: Path) -> None:
    with RunStore(tmp_path, "budget", require_local_wal=False) as store:
        store.reserve_model_request(100, request_limit=2, output_token_limit=150)
        with pytest.raises(ExternalInputError) as caught:
            store.reserve_model_request(60, request_limit=2, output_token_limit=150)
        assert caught.value.reason == "OUTPUT_TOKEN_BUDGET_EXHAUSTED"
        store.release_model_reservation(100)
        row = store.connection.execute(
            "SELECT request_count, reserved_output_tokens FROM budget WHERE singleton=1"
        ).fetchone()
        assert row["request_count"] == 1
        assert row["reserved_output_tokens"] == 0
        store.reserve_model_request(100, request_limit=3, output_token_limit=150)
        store.finalize_model_request(100, 20)
        store.reserve_model_request(60, request_limit=3, output_token_limit=150)
        with pytest.raises(ExternalInputError) as caught:
            store.reserve_model_request(1, request_limit=3, output_token_limit=150)
        assert caught.value.reason == "REQUEST_BUDGET_EXHAUSTED"


def test_only_one_coordinator_can_open_a_run(tmp_path: Path) -> None:
    with RunStore(tmp_path, "single", require_local_wal=False):
        with pytest.raises(ExternalInputError) as caught:
            RunStore(tmp_path, "single", require_local_wal=False)
        assert caught.value.reason == "RUN_ALREADY_ACTIVE"
