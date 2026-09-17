from __future__ import annotations

import io
import json
from pathlib import Path
from threading import Event
from threading import enumerate as threads

import pytest
from typer.testing import CliRunner

from pixelogue.cli import app
from pixelogue.config import load_config
from pixelogue.contracts import ConversationArtifact
from pixelogue.io import write_jsonl
from pixelogue.progress import synthesis_progress


def test_waiting_updates_and_thread_cleanup_on_failure():
    waiting = Event()

    class Stream(io.StringIO):
        def write(self, value):
            result = super().write(value)
            if "waiting" in value:
                waiting.set()
            return result

    stream = Stream()
    with pytest.raises(RuntimeError, match="failed"):
        with synthesis_progress(2, 4, stream, interval=0.01) as record:
            record("REJECTED")
            assert waiting.wait(2)
            raise RuntimeError("failed")
    output = stream.getvalue()
    assert "waiting: 1/2" in output
    assert "interrupted: 1/2" in output
    assert "REJECTED=1" in output
    assert "finished" not in output
    assert not any(thread.name == "pixelogue-progress" for thread in threads())


@pytest.mark.parametrize("quiet", [False, True])
def test_synthesize_cli_progress_leaves_stdout_json_and_uses_scheduled_total(
    tmp_path, monkeypatch, image_artifact, quiet
):
    image, root = image_artifact
    manifest = tmp_path / "images.jsonl"
    write_jsonl(manifest, [image, image, image])
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"run_root": tmp_path / "runs", "require_local_wal": False}
            ),
            "data": config.data.model_copy(update={"target_dialogues": 2}),
        }
    )
    monkeypatch.setattr("pixelogue.cli.load_config", lambda path: config)
    monkeypatch.setattr("pixelogue.cli._clients", lambda *args: (None, None, None))

    def synthesize_batch(self, jobs, artifact_root, *, max_workers):
        for index, job in enumerate(jobs):
            if index:
                saved = [json.loads(line) for line in output.read_text().splitlines()]
                assert len(saved) == index
                assert saved[0]["conversation_id"] == "conversation-0"
            yield ConversationArtifact(
                conversation_id=f"conversation-{index}",
                image=job.image,
                target_language="en",
                generation_model="test",
                turns=(),
                status="REJECTED" if index == 0 else "ERROR",
            )

    monkeypatch.setattr("pixelogue.cli.SynthesisCoordinator.synthesize_batch", synthesize_batch)
    output = tmp_path / "conversations.jsonl"
    args = [
        "synthesize",
        "--images",
        str(manifest),
        "--artifact-root",
        str(root),
        "--output",
        str(output),
        "--run-id",
        "progress-test",
        "--workers",
        "2",
    ]
    if quiet:
        args.append("--quiet")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["conversations"] == 2
    assert len(output.read_text().splitlines()) == 2
    if quiet:
        assert result.stderr == ""
    else:
        assert "started: 0/2" in result.stderr
        assert "finished: 2/2" in result.stderr
        assert "REJECTED=1, ABSTAINED=0, ERROR=1" in result.stderr
