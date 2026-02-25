from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from workflow_common import collect_key_artifact_hashes


def detect_harness_error(repo_root: Path) -> bool:
    analysis_path = repo_root / "crash_analysis.md"
    if not analysis_path.is_file():
        return False
    try:
        text = analysis_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return bool(re.search(r"HARNESS ERROR", text, re.IGNORECASE))


def bytes_human(num_bytes: int) -> str:
    n = max(0, int(num_bytes))
    units = ["B", "KB", "MB", "GB"]
    idx = 0
    val = float(n)
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(val)}{units[idx]}"
    return f"{val:.1f}{units[idx]}"


def tree_file_stats(root: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    if not root.is_dir():
        return files, total_bytes
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        files += 1
        try:
            total_bytes += int(p.stat().st_size)
        except Exception:
            pass
    return files, total_bytes


def collect_fuzz_inventory(repo_root: Path) -> dict[str, Any]:
    fuzz_dir = repo_root / "fuzz"
    out_dir = fuzz_dir / "out"
    corpus_dir = fuzz_dir / "corpus"
    artifacts_dir = out_dir / "artifacts"

    binaries: list[str] = []
    options_files: list[str] = []
    if out_dir.is_dir():
        for p in sorted(out_dir.iterdir()):
            if not p.is_file():
                continue
            name = p.name
            if name.endswith(".options"):
                options_files.append(name)
            if os.access(p, os.X_OK) or p.suffix.lower() == ".exe":
                binaries.append(name)

    artifact_files: list[str] = []
    if artifacts_dir.is_dir():
        for p in sorted(artifacts_dir.rglob("*")):
            if p.is_file():
                artifact_files.append(str(p.relative_to(repo_root)))

    corpus_stats: dict[str, dict[str, Any]] = {}
    corpus_total_files = 0
    corpus_total_bytes = 0
    if corpus_dir.is_dir():
        for d in sorted(corpus_dir.iterdir()):
            if not d.is_dir():
                continue
            files, size_bytes = tree_file_stats(d)
            corpus_total_files += files
            corpus_total_bytes += size_bytes
            corpus_stats[d.name] = {
                "files": files,
                "bytes": size_bytes,
                "human": bytes_human(size_bytes),
            }

    return {
        "fuzz_dir": str(fuzz_dir),
        "fuzz_out_dir": str(out_dir),
        "fuzz_corpus_dir": str(corpus_dir),
        "fuzzer_binaries": binaries,
        "fuzzer_count": len(binaries),
        "options_files": options_files,
        "artifact_files": artifact_files,
        "artifact_count": len(artifact_files),
        "corpus_stats": corpus_stats,
        "corpus_total_files": corpus_total_files,
        "corpus_total_bytes": corpus_total_bytes,
        "corpus_total_human": bytes_human(corpus_total_bytes),
    }


def write_run_summary(out: dict[str, Any]) -> None:
    repo_root_raw = out.get("repo_root")
    if not repo_root_raw:
        return
    repo_root = Path(str(repo_root_raw))
    if not repo_root.exists():
        return

    crash_found = bool(out.get("crash_found"))
    last_error = str(out.get("last_error") or "").strip()
    failed = bool(out.get("failed"))
    status = "error" if (failed or last_error) else ("crash_found" if crash_found else "ok")
    harness_error = detect_harness_error(repo_root)
    run_details = out.get("run_details") or []
    fuzz_inventory = collect_fuzz_inventory(repo_root)
    key_artifact_hashes = collect_key_artifact_hashes(repo_root)

    bundle_dirs = [
        d.name
        for d in repo_root.iterdir()
        if d.is_dir() and d.name.startswith(("challenge_bundle", "false_positive", "unreproducible"))
    ]

    data = {
        "repo_url": out.get("repo_url"),
        "repo_root": str(repo_root),
        "status": status,
        "message": out.get("message"),
        "last_step": out.get("last_step"),
        "step_count": out.get("step_count"),
        "build_attempts": out.get("build_attempts"),
        "build_rc": out.get("build_rc"),
        "build_error_kind": out.get("build_error_kind") or "",
        "build_error_code": out.get("build_error_code") or "",
        "run_rc": out.get("run_rc"),
        "last_error": last_error,
        "crash_found": crash_found,
        "crash_evidence": out.get("crash_evidence") or "none",
        "run_error_kind": out.get("run_error_kind") or "",
        "run_details": run_details,
        "last_fuzzer": out.get("last_fuzzer"),
        "last_crash_artifact": out.get("last_crash_artifact"),
        "harness_error": harness_error,
        "fix_patch_path": out.get("fix_patch_path") or "",
        "fix_patch_files": out.get("fix_patch_files") or [],
        "fix_patch_bytes": out.get("fix_patch_bytes") or 0,
        "crash_info_path": str(repo_root / "crash_info.md"),
        "crash_analysis_path": str(repo_root / "crash_analysis.md"),
        "reproducer_path": str(repo_root / "reproduce.py"),
        "bundles": bundle_dirs,
        "fuzz_inventory": fuzz_inventory,
        "key_artifact_hashes": key_artifact_hashes,
        "plan_policy": {
            "fix_on_crash": bool(out.get("plan_fix_on_crash", True)),
            "max_fix_rounds": int(out.get("plan_max_fix_rounds") or 1),
        },
        "timestamp": time.time(),
    }

    summary_json = repo_root / "run_summary.json"
    summary_md = repo_root / "run_summary.md"
    try:
        summary_json.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass

    md_lines = [
        "# Run Summary",
        "",
        f"- Status: {status}",
        f"- Repo: {data['repo_url']}",
        f"- Repo root: {data['repo_root']}",
        f"- Last step: {data['last_step']}",
        f"- Build attempts: {data['build_attempts']}",
        f"- Build rc: {data['build_rc']}",
        f"- Build error kind/code: {data['build_error_kind'] or 'none'}/{data['build_error_code'] or 'none'}",
        f"- Run rc: {data['run_rc']}",
        f"- Crash evidence: {data['crash_evidence']}",
        f"- Crash found: {crash_found}",
        f"- Harness error: {harness_error}",
        f"- Fuzzer binaries: {fuzz_inventory['fuzzer_count']}",
        f"- Corpus files: {fuzz_inventory['corpus_total_files']}",
        f"- Corpus size: {fuzz_inventory['corpus_total_human']}",
        f"- Plan crash policy: {'fix' if data['plan_policy']['fix_on_crash'] else 'report-only'}",
        f"- Plan max fix rounds: {data['plan_policy']['max_fix_rounds']}",
        f"- Key artifact hashes: {len(key_artifact_hashes)}",
    ]
    if key_artifact_hashes:
        md_lines.extend(["", "## Key Artifact Hashes"])
        for path, digest in sorted(key_artifact_hashes.items()):
            md_lines.append(f"- {path}: `{digest}`")
    if run_details:
        md_lines.extend(["", "## Fuzzer Effectiveness"])
        for item in run_details:
            md_lines.append(
                "- {fuzzer}: rc={rc}, cov={cov}, ft={ft}, corpus={corp_files}/{corp_size}, rss={rss}MB".format(
                    fuzzer=item.get("fuzzer"),
                    rc=item.get("rc"),
                    cov=item.get("final_cov"),
                    ft=item.get("final_ft"),
                    corp_files=item.get("final_corpus_files"),
                    corp_size=bytes_human(int(item.get("final_corpus_size_bytes") or 0)),
                    rss=item.get("final_rss_mb"),
                )
            )
    if last_error:
        md_lines.extend(["", "## Last Error", "```text", last_error, "```"])
    if crash_found:
        md_lines.extend(
            [
                "",
                "## Crash",
                f"- Fuzzer: {data['last_fuzzer']}",
                f"- Artifact: {data['last_crash_artifact']}",
                f"- crash_info.md: {data['crash_info_path']}",
                f"- crash_analysis.md: {data['crash_analysis_path']}",
            ]
        )
    if data["fix_patch_path"]:
        md_lines.extend(
            [
                "",
                "## Fix Patch",
                f"- Patch: {data['fix_patch_path']}",
                f"- Files changed: {len(data['fix_patch_files'])}",
            ]
        )
        if data["fix_patch_files"]:
            md_lines.extend([f"- {p}" for p in data["fix_patch_files"]])
    if bundle_dirs:
        md_lines.extend(["", "## Bundles"] + [f"- {b}" for b in bundle_dirs])

    try:
        summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    except Exception:
        pass

    out_dir = Path(str(fuzz_inventory.get("fuzz_out_dir") or ""))
    if out_dir.is_dir():
        eff_json = out_dir / "fuzz_effectiveness.json"
        eff_md = out_dir / "fuzz_effectiveness.md"
        eff = {
            "status": status,
            "repo_url": data.get("repo_url"),
            "run_rc": data.get("run_rc"),
            "crash_found": crash_found,
            "crash_evidence": data.get("crash_evidence"),
            "run_details": run_details,
            "fuzz_inventory": fuzz_inventory,
            "timestamp": data.get("timestamp"),
        }
        try:
            eff_json.write_text(json.dumps(eff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        eff_lines = [
            "# Fuzz Effectiveness",
            "",
            f"- Status: {status}",
            f"- Crash found: {crash_found}",
            f"- Run rc: {data.get('run_rc')}",
            f"- Fuzzer binaries: {fuzz_inventory['fuzzer_count']}",
            f"- Corpus files: {fuzz_inventory['corpus_total_files']}",
            f"- Corpus size: {fuzz_inventory['corpus_total_human']}",
        ]
        if run_details:
            eff_lines.extend(["", "## Per Fuzzer"])
            for item in run_details:
                eff_lines.append(
                    "- {fuzzer}: rc={rc}, cov={cov}, ft={ft}, corpus={corp_files}/{corp_size}, exec/s={eps}, rss={rss}MB".format(
                        fuzzer=item.get("fuzzer"),
                        rc=item.get("rc"),
                        cov=item.get("final_cov"),
                        ft=item.get("final_ft"),
                        corp_files=item.get("final_corpus_files"),
                        corp_size=bytes_human(int(item.get("final_corpus_size_bytes") or 0)),
                        eps=item.get("final_execs_per_sec"),
                        rss=item.get("final_rss_mb"),
                    )
                )
        try:
            eff_md.write_text("\n".join(eff_lines) + "\n", encoding="utf-8")
        except Exception:
            pass
