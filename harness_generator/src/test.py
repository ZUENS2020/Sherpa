from __future__ import annotations

from fuzz_unharnessed_repo import NonOssFuzzHarnessGenerator, RepoSpec
from pathlib import Path
import subprocess
import tempfile
from git import Repo, exc as git_exc
import os


def _clone_repo(spec: RepoSpec) -> Path:
    root = spec.workdir or Path(tempfile.mkdtemp(prefix="sherpa-fuzz-"))
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        # If provided, allow using an existing working folder (e.g., dev)
        print(f"[*] Using existing working directory: {root}")
        os.chdir(root)
        return root
    print(f"[*] Cloning {spec.url} → {root}")
    repo = Repo.clone_from(spec.url, root)
    if spec.ref:
        try:
            repo.git.checkout(spec.ref)
        except git_exc.GitCommandError:
            repo.git.fetch("origin", spec.ref)
            repo.git.checkout("FETCH_HEAD")
    print(f"[*] Checked out commit {repo.head.commit.hexsha}")
    os.chdir(root)
    return root

def _resolve_paths(input_dir: str, output_path: str) -> tuple[Path, Path, Path, Path]:
    """解析路径并返回 (static_analysis_dir, analyzer_bin, output_path_abs, input_dir_abs)。"""
    env_dir = (os.environ.get("STATIC_ANALYSIS_DIR") or "").strip()
    if env_dir:
        static_analysis_dir = Path(env_dir).expanduser().resolve()
    else:
        static_analysis_dir = (Path(__file__).resolve().parents[2] / "Static-Analysis").resolve()
    analyzer_bin = static_analysis_dir / "analyzer"
    output_abs = (static_analysis_dir / output_path).resolve() if not os.path.isabs(output_path) else Path(output_path)
    input_dir_abs = (static_analysis_dir / input_dir).resolve() if not os.path.isabs(input_dir) else Path(input_dir)
    return static_analysis_dir, analyzer_bin, output_abs, input_dir_abs


def run_analyzer(input_dir: str, output_path: str) -> str:
    """调用本地 analyzer 生成调用图 DOT 文件，返回简单结果字符串。"""
    static_analysis_dir, analyzer_bin, output_abs, input_abs = _resolve_paths(input_dir, output_path)

    if not analyzer_bin.exists():
        raise FileNotFoundError(f"未找到 analyzer 可执行文件: {analyzer_bin}")
    if not os.access(analyzer_bin, os.X_OK):
        raise PermissionError(f"analyzer 不可执行: {analyzer_bin}")
    if not input_abs.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_abs}")

    cmd = [str(analyzer_bin), "-input", str(input_abs), "-output", str(output_abs)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(static_analysis_dir),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"analyzer 执行失败，退出码 {e.returncode}:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        ) from e

    return (
        "Analyzer 执行成功。\n"
        f"命令: {' '.join(cmd)}\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"输出文件: {output_abs}"
    )

if __name__ == "__main__":
    repospec = RepoSpec(
        url="https://github.com/syoyo/tinyexr.git"
    )
    generator = NonOssFuzzHarnessGenerator(
    repo_spec= repospec,
    ai_key_path=Path("./.env"),)
    generator.generate()
