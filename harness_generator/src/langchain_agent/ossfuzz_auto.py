from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import sys

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# We reuse CodexHelper for robust non-interactive OpenCode CLI runs.
# NOTE: langchain_agent is not a package (no __init__.py), so we rely on sys.path
# setup done by the web entrypoint to import from harness_generator/src.
from codex_helper import CodexHelper  # type: ignore


class OssFuzzAutoError(RuntimeError):
    pass


def _oss_fuzz_repo_url() -> str:
    # Allow overriding for mirrors or internal forks.
    return os.environ.get("SHERPA_OSS_FUZZ_REPO_URL", "https://github.com/google/oss-fuzz.git").strip()


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _sanitize_project_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "sherpa"


def _project_name_from_repo_url(repo_url: str) -> str:
    base = repo_url.rstrip("/")
    base = base.split("/")[-1]
    if base.endswith(".git"):
        base = base[: -len(".git")]
    base = _sanitize_project_name(base)
    h = hashlib.sha1(repo_url.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return _sanitize_project_name(f"sherpa-{base}-{h}")


def _detect_language(repo_root: Path) -> str:
    # Strong Java signals
    for marker in (
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
    ):
        if (repo_root / marker).is_file():
            return "java"
    try:
        if any(repo_root.rglob("*.java")):
            return "java"
    except Exception:
        pass

    # C/C++ signals
    for marker in ("CMakeLists.txt", "configure.ac", "configure.in", "meson.build"):
        if (repo_root / marker).is_file():
            return "cpp"
    try:
        if any(repo_root.rglob("*.c")) or any(repo_root.rglob("*.cc")) or any(repo_root.rglob("*.cpp")):
            return "cpp"
    except Exception:
        pass

    return "unknown"


def _stream_cmd(cmd: Sequence[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int | None = None) -> int:
    print(f"[*] ➜  {' '.join(map(str, cmd))}")
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    start = time.time()
    for line in proc.stdout:
        print(line, end="")
        if timeout is not None and (time.time() - start) > timeout:
            try:
                proc.terminate()
            except Exception:
                pass
            break
    try:
        return proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return proc.wait()


def _clone_repo(repo_url: str, dst: Path) -> None:
    # Prefer host git if present; otherwise use dockerized git.
    if _which("git"):
        rc = _stream_cmd(["git", "clone", "--depth", "1", repo_url, str(dst)])
        if rc != 0:
            raise OssFuzzAutoError(f"git clone failed (rc={rc})")
        return

    if not _which("docker"):
        raise OssFuzzAutoError("Neither 'git' nor 'docker' is available to clone the repository.")

    rc = _stream_cmd(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(dst.parent.resolve())}:/work",
            "-w",
            "/work",
            "alpine/git",
            "clone",
            "--depth",
            "1",
            repo_url,
            dst.name,
        ]
    )
    if rc != 0:
        raise OssFuzzAutoError(f"docker git clone failed (rc={rc})")


def _ensure_oss_fuzz_checkout(oss_fuzz_dir: Path) -> None:
    helper = oss_fuzz_dir / "infra" / "helper.py"
    if helper.is_file():
        # Optional: keep checkout updated (off by default).
        if os.environ.get("SHERPA_OSS_FUZZ_UPDATE", "").strip() in {"1", "true", "yes"} and _which("git"):
            try:
                _stream_cmd(["git", "-C", str(oss_fuzz_dir), "pull", "--ff-only"], timeout=180)
            except Exception:
                pass
        return

    oss_fuzz_dir = oss_fuzz_dir.resolve()
    repo_url = _oss_fuzz_repo_url()

    # Bootstrap if missing/empty.
    if not oss_fuzz_dir.exists():
        oss_fuzz_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] OSS-Fuzz checkout not found. Cloning into: {oss_fuzz_dir}")
        _clone_repo(repo_url, oss_fuzz_dir)
    else:
        try:
            is_empty = not any(oss_fuzz_dir.iterdir())
        except Exception:
            is_empty = False

        if is_empty:
            print(f"[*] OSS-Fuzz directory is empty. Cloning into: {oss_fuzz_dir}")
            try:
                shutil.rmtree(oss_fuzz_dir)
            except Exception:
                pass
            _clone_repo(repo_url, oss_fuzz_dir)
        else:
            raise OssFuzzAutoError(
                f"Invalid oss-fuzz checkout: missing {helper}. Directory exists and is not empty. "
                f"Please set oss_fuzz_dir to a valid oss-fuzz root (or clear it so SHERPA can clone)."
            )

    helper = oss_fuzz_dir / "infra" / "helper.py"
    if not helper.is_file():
        raise OssFuzzAutoError(
            f"Failed to bootstrap oss-fuzz checkout: still missing {helper}. "
            f"Tried cloning from {repo_url}."
        )


def _scan_built_fuzzers(oss_fuzz_dir: Path, project: str) -> list[Path]:
    out_dir = oss_fuzz_dir / "build" / "out" / project
    if not out_dir.is_dir():
        return []

    fuzzers: list[Path] = []
    tool_names = {
        "llvm-symbolizer",
        "llvm-symbolizer.exe",
        "clang",
        "clang++",
        "gcc",
        "g++",
        "python",
        "python3",
    }
    for p in out_dir.iterdir():
        if p.is_dir():
            continue
        # Skip common non-binary artifacts
        if p.name.endswith((".zip", ".dict", ".options", ".txt", ".md", ".json", ".yaml", ".yml")):
            continue
        # Heuristic: fuzzer binaries typically have no extension in OSS-Fuzz
        if p.suffix:
            continue
        if p.name in tool_names:
            continue
        try:
            if p.stat().st_size <= 0:
                continue
        except Exception:
            continue
        fuzzers.append(p)

    if not fuzzers:
        return []

    # Prefer names that look like fuzzers.
    preferred = [p for p in fuzzers if re.search(r"fuzz", p.name, re.IGNORECASE)]
    return sorted(preferred or fuzzers)


def _scan_crash_artifacts(oss_fuzz_dir: Path, project: str) -> list[Path]:
    out_dir = oss_fuzz_dir / "build" / "out" / project
    if not out_dir.is_dir():
        return []
    artifacts: list[Path] = []
    for p in out_dir.rglob("*"):
        if p.is_dir():
            continue
        n = p.name
        if n.startswith("crash-") or n.startswith("oom-") or n.startswith("timeout-"):
            artifacts.append(p)
    return sorted(artifacts)


def _extract_json_object(text: str) -> dict[str, object] | None:
    if not text:
        return None
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        val = json.loads(blob)
    except Exception:
        return None
    return val if isinstance(val, dict) else None


def _normalize_openai_base_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u
    suffix = "/chat/completions"
    if u.endswith(suffix):
        return u[: -len(suffix)]
    return u.rstrip("/")


def _read_key_from_env_file(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'\"")
        if k in {"OPENAI_API_KEY"} and v:
            return v
    return ""


def _read_env_from_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    for line in text.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("'\"")
        if not k or not v:
            continue
        out[k] = v
    return out


def _ensure_openai_env(ai_key_path: Path) -> None:
    env = _read_env_from_env_file(ai_key_path)
    if env.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
    if env.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = env["OPENAI_BASE_URL"]


def _repo_context_snippet(repo_root: Path) -> str:
    lines: list[str] = []
    try:
        top = sorted(p.name for p in repo_root.iterdir())
        lines.append("Top-level files/dirs:")
        lines.extend(top[:60])
    except Exception:
        pass
    for candidate in ("CMakeLists.txt", "configure.ac", "meson.build", "Makefile", "README.md"):
        p = repo_root / candidate
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
                lines.append(f"\n--- {candidate} (head) ---")
                lines.append("\n".join(content.splitlines()[:120]))
            except Exception:
                continue
    return "\n".join(lines).strip()


def _llm_generate_ossfuzz_project(
    *,
    repo_url: str,
    project: str,
    lang: str,
    target_dir: Path,
    ossproj_dir: Path,
    ai_key_path: Path,
) -> None:
    if OpenAI is None:
        raise OssFuzzAutoError("openai python package is not available for LLM fallback")

    _ensure_openai_env(ai_key_path)
    api_key = os.environ.get("OPENAI_API_KEY") or _read_key_from_env_file(ai_key_path)
    if not api_key:
        raise OssFuzzAutoError("Missing OPENAI_API_KEY for LLM fallback")

    base_url = _normalize_openai_base_url(os.environ.get("OPENAI_BASE_URL", ""))
    model = os.environ.get("OPENAI_MODEL", "glm-4.7").strip() or "glm-4.7"

    ctx = _repo_context_snippet(target_dir)
    system = (
        "You are generating an OSS-Fuzz project folder.\n"
        "Return ONLY JSON with this schema:\n"
        "{\n"
        "  \"files\": {\n"
        "    \"Dockerfile\": \"...\",\n"
        "    \"build.sh\": \"...\",\n"
        "    \"project.yaml\": \"...\",\n"
        "    \"<optional harness file>\": \"...\"\n"
        "  }\n"
        "}\n"
        "Rules:\n"
        "- Dockerfile must clone the upstream repo into $SRC/<project>\n"
        "- build.sh must build at least one fuzzer into $OUT with no extension\n"
        "- project.yaml must set the project name, language, and homepage\n"
        "- Keep it minimal and buildable\n"
        "- Do not include Markdown fences\n"
    )
    user = (
        f"Project name: {project}\n"
        f"Repo URL: {repo_url}\n"
        f"Language: {lang}\n"
        "Local repo path: ./target (for context only)\n"
        "\nRepository context:\n"
        f"{ctx}\n"
    )

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
    )
    text = getattr(resp.choices[0].message, "content", None) or ""
    obj = _extract_json_object(text) or {}
    files = obj.get("files") if isinstance(obj, dict) else None
    if not isinstance(files, dict) or not files:
        raise OssFuzzAutoError("LLM fallback did not return any files")

    ossproj_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        if not isinstance(rel, str) or not isinstance(content, str):
            continue
        rel = rel.lstrip("/").replace("\\", "/")
        out_path = ossproj_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class OssFuzzAutoInput:
    repo_url: str
    oss_fuzz_dir: Path
    ai_key_path: Path
    time_budget: int


def run_ossfuzz_auto(inp: OssFuzzAutoInput) -> str:
    _ensure_oss_fuzz_checkout(inp.oss_fuzz_dir)
    _ensure_openai_env(inp.ai_key_path)

    repo_url = inp.repo_url.strip()
    if not repo_url:
        raise OssFuzzAutoError("repo_url is required")

    project = _project_name_from_repo_url(repo_url)
    projects_dir = inp.oss_fuzz_dir / "projects"
    project_dir = projects_dir / project
    if project_dir.exists():
        # Extremely unlikely due to hash suffix; but keep safe.
        raise OssFuzzAutoError(f"Project directory already exists: {project_dir}")

    with tempfile.TemporaryDirectory(prefix="sherpa-ossfuzz-") as td:
        work = Path(td)
        target_dir = work / "target"
        ossproj_dir = work / "oss-fuzz-project"
        target_dir.mkdir(parents=True, exist_ok=True)
        ossproj_dir.mkdir(parents=True, exist_ok=True)

        print(f"[*] Cloning target repo: {repo_url}")
        _clone_repo(repo_url, target_dir)

        lang = _detect_language(target_dir)
        if lang == "unknown":
            raise OssFuzzAutoError(
                "Unable to auto-detect a supported language for OSS-Fuzz project generation. Supported: C/C++ and Java."
            )

        # Ask OpenCode to create a runnable OSS-Fuzz project directory.
        # We keep OpenCode working on a small temp git repo containing ONLY:
        # - ./target/ (cloned source for analysis)
        # - ./oss-fuzz-project/ (the output project files)
        prompt = textwrap.dedent(
            f"""
            You are generating an OSS-Fuzz project definition for a third-party repository.

            Target repository is available locally under: ./target/

            Your job: create a COMPLETE oss-fuzz project folder under ./oss-fuzz-project/ with these required files:
            - Dockerfile
            - build.sh
            - project.yaml
            Plus any harness source files needed.

            Constraints:
            - The generated OSS-Fuzz project name is: {project}
            - The upstream repo URL is: {repo_url}
            - Language: {lang} (either 'cpp' for C/C++ libFuzzer or 'java' for Jazzer)

            OSS-Fuzz environment notes:
            - build.sh is executed inside the project's Docker image with env vars like $SRC, $OUT, $CC, $CXX, $CFLAGS, $CXXFLAGS, $LIB_FUZZING_ENGINE.
            - build.sh must produce at least one fuzzer binary in $OUT with no extension.
            - Dockerfile must fetch the upstream source into $SRC/{project} (e.g. via git clone) and COPY build.sh to $SRC/build.sh.

            Requirements:
            - Keep it minimal but buildable.
            - Prefer CMake/Autotools/Make/Meson detection for C/C++.
            - For Java, use Jazzer conventions (build.sh should compile/build and place a Jazzer driver in $OUT).
            - Create ONE initial high-value harness (attacker-reachable parser or public API) based on the code in ./target/.

            When finished, write the path to the most important generated file (likely oss-fuzz-project/build.sh) into ./done.
            """
        ).strip()

        codex = CodexHelper(repo_path=work, ai_key_path=str(inp.ai_key_path), copy_repo=False)
        codex_err: Exception | None = None
        try:
            codex.run_codex_command(prompt)
        except Exception as e:
            codex_err = e

        # Basic validation before copying into oss-fuzz checkout
        required = [ossproj_dir / "Dockerfile", ossproj_dir / "build.sh", ossproj_dir / "project.yaml"]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            # Fallback to a direct LLM call (OpenAI-compatible) to generate files.
            try:
                _llm_generate_ossfuzz_project(
                    repo_url=repo_url,
                    project=project,
                    lang=lang,
                    target_dir=target_dir,
                    ossproj_dir=ossproj_dir,
                    ai_key_path=inp.ai_key_path,
                )
            except Exception as e:
                if codex_err:
                    raise OssFuzzAutoError(
                        f"OpenCode failed and LLM fallback also failed: codex_err={codex_err} fallback_err={e}"
                    )
                raise OssFuzzAutoError(f"OpenCode did not generate required oss-fuzz files: {missing}; fallback_err={e}")

            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise OssFuzzAutoError(f"LLM fallback did not generate required oss-fuzz files: {missing}")

        print(f"[*] Installing generated project into oss-fuzz: {project_dir}")
        project_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ossproj_dir, project_dir, dirs_exist_ok=False)

    # Build image + fuzzers using oss-fuzz helper
    helper = inp.oss_fuzz_dir / "infra" / "helper.py"

    print(f"[*] OSS-Fuzz: build_image {project}")
    rc = _stream_cmd([sys.executable, str(helper), "build_image", "--no-pull", project], cwd=inp.oss_fuzz_dir)
    if rc != 0:
        raise OssFuzzAutoError(f"oss-fuzz build_image failed (rc={rc})")

    # Build fuzzers (retry loop with OpenCode fixes)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"[*] OSS-Fuzz: build_fuzzers {project} (attempt {attempt}/{max_attempts})")
        rc = _stream_cmd([sys.executable, str(helper), "build_fuzzers", project], cwd=inp.oss_fuzz_dir)
        fuzzers = _scan_built_fuzzers(inp.oss_fuzz_dir, project)
        if rc == 0 and fuzzers:
            break

        if attempt >= max_attempts:
            raise OssFuzzAutoError(
                f"oss-fuzz build_fuzzers failed after {max_attempts} attempts (rc={rc}), fuzzers_found={len(fuzzers)}"
            )

        # Ask OpenCode to minimally fix the oss-fuzz project files.
        print("[*] build failed; asking OpenCode to fix OSS-Fuzz project files…")
        proj_repo = CodexHelper(repo_path=(inp.oss_fuzz_dir / "projects" / project), ai_key_path=str(inp.ai_key_path), copy_repo=False)
        out_dir = inp.oss_fuzz_dir / "build" / "out" / project
        log_hint = "\n".join(
            [
                f"helper_rc={rc}",
                f"out_dir={out_dir}",
                "If build_fuzzers output indicated missing deps, add them in Dockerfile.",
                "If compilation/linking failed, adjust build.sh/harness accordingly.",
            ]
        )
        fix_prompt = textwrap.dedent(
            f"""
            Fix this OSS-Fuzz project so that `infra/helper.py build_fuzzers {project}` succeeds.

            Constraints:
            - Only edit files in this project directory (Dockerfile, build.sh, project.yaml, harness sources).
            - Keep changes minimal and focused on making build_fuzzers succeed.
            - Ensure at least one fuzzer binary is produced in $OUT.

            Context:
            {log_hint}

            When finished, write the path to the key file you modified into ./done.
            """
        ).strip()
        proj_repo.run_codex_command(fix_prompt)

    fuzzers = _scan_built_fuzzers(inp.oss_fuzz_dir, project)
    if not fuzzers:
        raise OssFuzzAutoError("No fuzzer binaries found after successful build_fuzzers.")

    fuzzer = fuzzers[0].name
    baseline_artifacts = set(map(str, _scan_crash_artifacts(inp.oss_fuzz_dir, project)))

    print(f"[*] OSS-Fuzz: run_fuzzer {project} {fuzzer} (time_budget={inp.time_budget}s)")
    # libFuzzer supports -max_total_time.
    rc = _stream_cmd(
        [
            sys.executable,
            str(helper),
            "run_fuzzer",
            project,
            fuzzer,
            "--",
            f"-max_total_time={int(inp.time_budget)}",
        ],
        cwd=inp.oss_fuzz_dir,
        timeout=int(inp.time_budget) + 600,
    )
    print(f"[*] run_fuzzer rc={rc}")

    new_artifacts = [p for p in _scan_crash_artifacts(inp.oss_fuzz_dir, project) if str(p) not in baseline_artifacts]
    if new_artifacts:
        first = new_artifacts[0]
        return f"OSS-Fuzz run completed: crash artifact found: {first} (project={project}, fuzzer={fuzzer})"

    return f"OSS-Fuzz run completed: no crash found (project={project}, fuzzer={fuzzer})"
