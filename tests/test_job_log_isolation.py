from __future__ import annotations

import sys
import threading
import time
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import main as web_main


def _emit_job_logs(
    stream: object,
    job_id: str,
    marker: str,
    count: int,
) -> None:
    tee = web_main._Tee(job_id)
    out_token = web_main._ACTIVE_JOB_STDOUT_TEE.set(tee)
    err_token = web_main._ACTIVE_JOB_STDERR_TEE.set(tee)
    try:
        for idx in range(count):
            stream.write(f"{job_id}:{marker}:{idx}\n")
            time.sleep(0.001)
    finally:
        web_main._ACTIVE_JOB_STDOUT_TEE.reset(out_token)
        web_main._ACTIVE_JOB_STDERR_TEE.reset(err_token)
        tee.close()


def test_concurrent_job_logs_remain_isolated():
    job_a = "job-a"
    job_b = "job-b"
    with web_main._JOBS_LOCK:
        web_main._JOBS.clear()
        web_main._JOBS[job_a] = {"job_id": job_a, "log": "", "status": "running"}
        web_main._JOBS[job_b] = {"job_id": job_b, "log": "", "status": "running"}

    stream = web_main._JobAwareStream(StringIO(), stream_kind="stdout")
    t1 = threading.Thread(target=_emit_job_logs, args=(stream, job_a, "AAA", 40))
    t2 = threading.Thread(target=_emit_job_logs, args=(stream, job_b, "BBB", 40))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    snap_a = web_main._job_snapshot(job_a) or {}
    snap_b = web_main._job_snapshot(job_b) or {}
    log_a = str(snap_a.get("log") or "")
    log_b = str(snap_b.get("log") or "")

    assert f"{job_a}:AAA:" in log_a
    assert f"{job_b}:BBB:" in log_b
    assert "BBB" not in log_a
    assert "AAA" not in log_b
