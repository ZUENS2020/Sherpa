from __future__ import annotations
import os
from dotenv import load_dotenv
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 获取父文件夹路径 dirname:获取目录名(路径去掉文件名)，嵌套两次获取父文件夹
sys.path.append(parent_dir)  # 添加父文件夹路径
from pathlib import Path

from ossfuzz_auto import OssFuzzAutoInput, run_ossfuzz_auto

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
) -> str:
    # NOTE: email/docker_image/max_len are currently not used in OSS-Fuzz mode.
    oss_dir = (oss_fuzz_dir or "").strip()
    if not oss_dir:
        raise ValueError("oss_fuzz_dir is required. Configure it in Web: OSS-Fuzz 本地路径（oss_fuzz_dir）")

    return run_ossfuzz_auto(
        OssFuzzAutoInput(
            repo_url=repo_url,
            oss_fuzz_dir=Path(oss_dir).expanduser().resolve(),
            ai_key_path=(ai_key_path or _find_ai_key_path()),
            time_budget=int(time_budget or 900),
        )
    )

if __name__ == "__main__":
    pass