from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class SQLiteJobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT,
                    status TEXT,
                    repo TEXT,
                    created_at REAL,
                    updated_at REAL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            conn.commit()

    def upsert_job(self, job: dict[str, Any]) -> None:
        payload = json.dumps(job, ensure_ascii=False, default=str)
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        job_id, kind, status, repo, created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        kind=excluded.kind,
                        status=excluded.status,
                        repo=excluded.repo,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at,
                        payload_json=excluded.payload_json
                    """,
                    (
                        str(job.get("job_id") or ""),
                        str(job.get("kind") or ""),
                        str(job.get("status") or ""),
                        str(job.get("repo") or ""),
                        float(job.get("created_at") or 0.0),
                        float(job.get("updated_at") or 0.0),
                        payload,
                    ),
                )
                conn.commit()

    def load_jobs(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM jobs ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
        for (payload_json,) in rows:
            try:
                job = json.loads(payload_json)
            except Exception:
                continue
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                continue
            out[job_id] = job
        return out
