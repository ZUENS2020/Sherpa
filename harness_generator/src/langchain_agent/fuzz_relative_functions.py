from __future__ import annotations
import os
from dotenv import load_dotenv
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取父文件夹路径 dirname:获取目录名(路径去掉文件名)，嵌套两次获取父文件夹
sys.path.append(parent_dir)  # 添加父文件夹路径
from pathlib import Path

from workflow_graph import FuzzWorkflowInput, run_fuzz_workflow

load_dotenv()


def _find_ai_key_path() -> Path:
    """Find a suitable .env/key file for Codex/OpenAI automation.

    Priority:
      1) Environment variable AI_KEY_PATH
      2) Repo root .env
      3) harness_generator/.env
      4) src/.env
    """
    env_path = os.environ.get("AI_KEY_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = []
    # .../harness_generator/src/langchain_agent/fuzz_relative_functions.py
    # parents[4] should be repo root
    for idx in (4, 3, 2):
        try:
            candidates.append(here.parents[idx] / ".env")
        except Exception:
            continue
    for p in candidates:
        if p.is_file():
            return p
    # default (may not exist yet)
    return (here.parents[4] / ".env")


def fuzz_logic(
    repo_url: str,
    max_len: int = 1024,
    time_budget: int = 900,
    email: str | None = None,
    docker_image: str | None = None,
    ai_key_path: Path | None = None,
    oss_fuzz_dir: str | None = None,
    model: str | None = None,
) -> str:
    # Set model in environment so OpenCode can pick it up
    if model and model.strip() and not os.environ.get("OPENCODE_MODEL"):
        os.environ["OPENCODE_MODEL"] = model.strip()
    if oss_fuzz_dir and oss_fuzz_dir.strip():
        os.environ["SHERPA_DEFAULT_OSS_FUZZ_DIR"] = oss_fuzz_dir.strip()
    print(f"[DEBUG] Entering run_fuzz_workflow with repo_url={repo_url}")
    try:
        result = run_fuzz_workflow(
            FuzzWorkflowInput(
                repo_url=repo_url,
                email=email,
                time_budget=int(time_budget or 900),
                max_len=int(max_len or 1024),
                docker_image=docker_image,
                ai_key_path=(ai_key_path or _find_ai_key_path()),
                model=model,
            )
        )
        print(f"[DEBUG] run_fuzz_workflow returned successfully")
        return result
    except Exception as e:
        print(f"[DEBUG] run_fuzz_workflow failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    pass
