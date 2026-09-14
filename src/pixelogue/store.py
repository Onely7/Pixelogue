"""Crash-aware local SQLite ledger and content-addressed artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pixelogue.errors import ExternalInputError
from pixelogue.serialization import canonical_json

NETWORK_FILESYSTEMS = {"nfs", "nfs4", "cifs", "smbfs", "fuse.sshfs"}


def filesystem_type(path: Path) -> str | None:
    """Return the Linux mount type containing `path`, when discoverable."""
    resolved = path.resolve()
    best_mount = Path("/")
    best_type: str | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        after_fields = after.split()
        if len(fields) < 5 or not after_fields:
            continue
        mount = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        if len(mount.parts) >= len(best_mount.parts):
            best_mount = mount
            best_type = after_fields[0]
    return best_type


class RunStore:
    """Single-writer run state and immutable artifact store."""

    def __init__(self, run_root: Path, run_id: str, *, require_local_wal: bool = True) -> None:
        """Open or create a run ledger.

        Args:
            run_root: Local base directory for active runs.
            run_id: Stable user-supplied run identifier.
            require_local_wal: Reject known network filesystems when true.

        Raises:
            ExternalInputError: If the active database would use a network filesystem.
        """
        self.run_dir = run_root.resolve() / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        current_type = filesystem_type(self.run_dir)
        if require_local_wal and current_type in NETWORK_FILESYSTEMS:
            raise ExternalInputError(
                "SQLITE_WAL_REQUIRES_LOCAL_FILESYSTEM",
                f"Active run directory uses {current_type}: {self.run_dir}",
            )
        self._lock_stream = (self.run_dir / "coordinator.lock").open("a+b")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_stream.close()
            raise ExternalInputError(
                "RUN_ALREADY_ACTIVE",
                f"Another coordinator holds {self.run_dir}",
            ) from error
        self.artifact_dir = self.run_dir / "artifacts"
        self.artifact_dir.mkdir(exist_ok=True)
        self.database_path = self.run_dir / "run.sqlite3"
        self._mutex = threading.RLock()
        self.connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        mode = self.connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            raise ExternalInputError(
                "SQLITE_WAL_UNAVAILABLE", f"SQLite selected journal mode {mode}"
            )
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        """Checkpoint and close the run database."""
        with self._mutex:
            self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.connection.close()
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_UN)
            self._lock_stream.close()

    def __enter__(self) -> RunStore:
        """Return the open store for a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the store when leaving a context manager."""
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS run (
                run_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                profile TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS artifact (
                artifact_hash TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                relative_path TEXT NOT NULL UNIQUE,
                complete INTEGER NOT NULL CHECK (complete IN (0, 1)),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS model_call (
                call_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                model_repo TEXT NOT NULL,
                model_revision TEXT,
                model_lock_hash TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                request_artifact_hash TEXT NOT NULL REFERENCES artifact(artifact_hash),
                response_artifact_hash TEXT REFERENCES artifact(artifact_hash),
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER NOT NULL DEFAULT 0 CHECK (duration_ms >= 0),
                status TEXT NOT NULL,
                UNIQUE(model_lock_hash, stage, request_hash)
            );
            CREATE TABLE IF NOT EXISTS turn_commit (
                conversation_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                attempt INTEGER NOT NULL,
                artifact_hash TEXT NOT NULL REFERENCES artifact(artifact_hash),
                PRIMARY KEY(conversation_id, branch_id, turn_index)
            );
            CREATE TABLE IF NOT EXISTS event (
                event_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                previous_state TEXT,
                next_state TEXT NOT NULL,
                artifact_hash TEXT REFERENCES artifact(artifact_hash),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS budget (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                request_count INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                reserved_output_tokens INTEGER NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO budget(singleton) VALUES (1);
            """
        )
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(model_call)")}
        if "duration_ms" not in columns:
            self.connection.execute(
                "ALTER TABLE model_call ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0"
            )

    def initialize_run(self, run_id: str, config_hash: str, profile: str) -> None:
        """Create a run or verify that a resumed run has the same configuration."""
        with self._mutex:
            existing = self.connection.execute(
                "SELECT config_hash, profile FROM run WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                if existing["config_hash"] != config_hash or existing["profile"] != profile:
                    raise ExternalInputError(
                        "RUN_CONFIG_MISMATCH",
                        "Existing run was created with another configuration",
                    )
                return
            with self.connection:
                self.connection.execute(
                    "INSERT INTO run(run_id, config_hash, profile, status) VALUES (?, ?, ?, 'CREATED')",
                    (run_id, config_hash, profile),
                )

    def write_json_artifact(self, kind: str, value: object) -> str:
        """Atomically store canonical JSON and return its SHA-256 identity."""
        return self.write_artifact(kind, canonical_json(value))

    def write_artifact(self, kind: str, payload: bytes) -> str:
        """Atomically store bytes before registering the complete reference."""
        with self._mutex:
            artifact_hash = hashlib.sha256(payload).hexdigest()
            relative_path = Path(kind) / artifact_hash[:2] / artifact_hash
            destination = self.artifact_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                temporary = destination.with_name(f".{artifact_hash}.{os.getpid()}.tmp")
                with temporary.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
            with self.connection:
                self.connection.execute(
                    """INSERT OR IGNORE INTO artifact(artifact_hash, kind, relative_path, complete)
                       VALUES (?, ?, ?, 1)""",
                    (artifact_hash, kind, relative_path.as_posix()),
                )
            return artifact_hash

    def read_artifact(self, artifact_hash: str) -> bytes:
        """Read an artifact after verifying both ledger and bytes."""
        with self._mutex:
            row = self.connection.execute(
                "SELECT relative_path, complete FROM artifact WHERE artifact_hash = ?",
                (artifact_hash,),
            ).fetchone()
            if row is None or row["complete"] != 1:
                raise ExternalInputError(
                    "ARTIFACT_NOT_COMPLETE", f"Unknown artifact {artifact_hash}"
                )
            payload = (self.artifact_dir / row["relative_path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != artifact_hash:
                raise ExternalInputError(
                    "ARTIFACT_HASH_MISMATCH", f"Corrupt artifact {artifact_hash}"
                )
            return payload

    def verify(self) -> dict[str, int]:
        """Verify database integrity and every complete artifact hash."""
        with self._mutex:
            integrity = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ExternalInputError("SQLITE_INTEGRITY_FAILED", str(integrity))
            rows = self.connection.execute(
                "SELECT artifact_hash FROM artifact WHERE complete = 1 ORDER BY artifact_hash"
            ).fetchall()
            for row in rows:
                self.read_artifact(row["artifact_hash"])
            missing_responses = self.connection.execute(
                """SELECT COUNT(*) FROM model_call
                   WHERE status = 'COMPLETE' AND response_artifact_hash IS NULL"""
            ).fetchone()[0]
            if missing_responses:
                raise ExternalInputError(
                    "MODEL_CALL_RESPONSE_MISSING",
                    f"{missing_responses} complete calls lack response artifacts",
                )
            return {
                "artifacts": len(rows),
                "model_calls": self.connection.execute(
                    "SELECT COUNT(*) FROM model_call"
                ).fetchone()[0],
                "turn_commits": self.connection.execute(
                    "SELECT COUNT(*) FROM turn_commit"
                ).fetchone()[0],
            }

    def committed_artifact_hashes(self, conversation_id: str) -> tuple[str, ...]:
        """Return a contiguous committed turn prefix for one conversation."""
        with self._mutex:
            rows = self.connection.execute(
                """SELECT turn_index, artifact_hash FROM turn_commit
                   WHERE conversation_id = ? AND branch_id = 'main'
                   ORDER BY turn_index""",
                (conversation_id,),
            ).fetchall()
            indices = [row["turn_index"] for row in rows]
            if indices != list(range(1, len(rows) + 1)):
                raise ExternalInputError(
                    "TURN_COMMIT_GAP",
                    f"Committed turns are not contiguous for {conversation_id}",
                )
            return tuple(row["artifact_hash"] for row in rows)

    def reserve_model_request(
        self,
        reserved_output_tokens: int,
        *,
        request_limit: int,
        output_token_limit: int,
    ) -> None:
        """Atomically reserve one request and its worst-case output budget."""
        with self._mutex, self.connection:
            row = self.connection.execute(
                "SELECT request_count, output_tokens, reserved_output_tokens FROM budget WHERE singleton=1"
            ).fetchone()
            if row["request_count"] + 1 > request_limit:
                raise ExternalInputError("REQUEST_BUDGET_EXHAUSTED", str(request_limit))
            projected = (
                row["output_tokens"] + row["reserved_output_tokens"] + reserved_output_tokens
            )
            if projected > output_token_limit:
                raise ExternalInputError("OUTPUT_TOKEN_BUDGET_EXHAUSTED", str(output_token_limit))
            self.connection.execute(
                """UPDATE budget
                   SET request_count = request_count + 1,
                       reserved_output_tokens = reserved_output_tokens + ?
                   WHERE singleton = 1""",
                (reserved_output_tokens,),
            )

    def finalize_model_request(
        self, reserved_output_tokens: int, actual_output_tokens: int
    ) -> None:
        """Replace a successful request reservation with measured output tokens."""
        with self._mutex, self.connection:
            self.connection.execute(
                """UPDATE budget
                   SET reserved_output_tokens = reserved_output_tokens - ?,
                       output_tokens = output_tokens + ?
                   WHERE singleton = 1""",
                (reserved_output_tokens, actual_output_tokens),
            )

    def release_model_reservation(self, reserved_output_tokens: int) -> None:
        """Release output capacity after a failed request while retaining its request count."""
        with self._mutex, self.connection:
            row = self.connection.execute(
                "SELECT reserved_output_tokens FROM budget WHERE singleton=1"
            ).fetchone()
            if row["reserved_output_tokens"] < reserved_output_tokens:
                raise ExternalInputError(
                    "BUDGET_RESERVATION_MISMATCH",
                    "Cannot release more output tokens than are reserved",
                )
            self.connection.execute(
                """UPDATE budget
                   SET reserved_output_tokens = reserved_output_tokens - ?
                   WHERE singleton = 1""",
                (reserved_output_tokens,),
            )

    def completed_model_call(
        self,
        model_lock_hash: str,
        stage: str,
        request_hash: str,
    ) -> tuple[bytes, int, int] | None:
        """Return a verified saved response and usage for an identical complete request."""
        with self._mutex:
            row = self.connection.execute(
                """SELECT response_artifact_hash, input_tokens, output_tokens
                   FROM model_call
                   WHERE model_lock_hash = ? AND stage = ? AND request_hash = ?
                     AND status = 'COMPLETE'""",
                (model_lock_hash, stage, request_hash),
            ).fetchone()
            if row is None or row["response_artifact_hash"] is None:
                return None
            return (
                self.read_artifact(row["response_artifact_hash"]),
                row["input_tokens"],
                row["output_tokens"],
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Expose one short coordinator transaction."""
        with self._mutex, self.connection:
            yield self.connection

    def backup(self, destination: Path) -> Path:
        """Create a consistent SQLite backup and copy immutable artifacts."""
        with self._mutex:
            destination.mkdir(parents=True, exist_ok=True)
            database_backup = destination / f"{self.run_dir.name}.sqlite3"
            with sqlite3.connect(database_backup) as target:
                self.connection.backup(target)
            artifact_backup = destination / f"{self.run_dir.name}-artifacts"
            if artifact_backup.exists():
                raise ExternalInputError(
                    "BACKUP_EXISTS", f"Backup already exists: {artifact_backup}"
                )
            shutil.copytree(self.artifact_dir, artifact_backup)
            return database_backup

    @staticmethod
    def restore(database: Path, artifacts: Path, destination: Path) -> None:
        """Restore a backup into a new, empty local run directory."""
        if destination.exists() and any(destination.iterdir()):
            raise ExternalInputError("RESTORE_DESTINATION_NOT_EMPTY", str(destination))
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(database, destination / "run.sqlite3")
        shutil.copytree(artifacts, destination / "artifacts", dirs_exist_ok=False)
