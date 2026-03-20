from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "harness_generator" / "src" / "langchain_agent" / "opencode_skills"


def _load(stage: str) -> str:
    return (SKILL_ROOT / stage / "SKILL.md").read_text(encoding="utf-8")


def test_synthesize_contract_matches_workflow_hard_checks() -> None:
    synth = _load("synthesize")
    assert "at least one harness source file under `fuzz/`" in synth
    assert "before completing scaffold docs/json" in synth
    assert "`fuzz/repo_understanding.json` must include non-empty" in synth
    assert "`chosen_target_api`" in synth
    assert "`evidence` (non-empty array)" in synth
    assert "must be a target API identifier" in synth
    assert "forbidden examples: `fuzz/xxx_fuzz.cc`" in synth
    assert "`build_system` must not be `unknown`" in synth
    assert "`evidence` must be a non-empty string array" in synth
    assert "never use shell substitutions like `$(nproc)`" in synth
    assert '["-j", str(os.cpu_count() or 1)]' in synth


def test_synthesize_completion_contract_repairs_harness_and_understanding() -> None:
    complete = _load("synthesize_complete_scaffold")
    assert "if harness source is missing, create at least one harness source file" in complete
    assert "before only-doc/json fixes" in complete
    assert "ensure non-empty `build_system`, `chosen_target_api`, `chosen_target_reason`, `fuzzer_entry_strategy`" in complete
    assert "ensure `evidence` is a non-empty array" in complete
    assert "semantically invalid" in complete
    assert "not a harness file path" in complete
    assert 'build_system.lower() != "unknown"' in complete
    assert "if `fuzz/build.py` exists and uses invalid parallel style" in complete


def test_schema_and_fix_stage_contracts_cover_known_failure_modes() -> None:
    plan = _load("plan")
    plan_fix = _load("plan_fix_targets_schema")
    fix_build = _load("fix_build")
    fix_crash_h = _load("fix_crash_harness_error")
    fix_crash_u = _load("fix_crash_upstream_bug")

    assert "forbidden: `name = LLVMFuzzerTestOneInput`" in plan
    assert "`api` must describe a target API identifier" in plan
    assert "Read and fix <path>[:line]" in plan
    assert "forbidden: `name = LLVMFuzzerTestOneInput`" in plan_fix
    assert "semantic reminder: do not rewrite `api` to harness file paths" in plan_fix
    assert "Read and fix <path>[:line]" in plan_fix
    assert "canonical vcpkg examples" in fix_build
    assert "never `z`, `bz2`, `lzma`" in fix_build
    assert "do not bypass workflow acceptance" in fix_build
    assert "Read and fix <path>[:line]" in fix_build
    assert "must produce textual code changes; pure no-op is invalid." in fix_crash_h
    assert "do not bypass acceptance by tampering" in fix_crash_h
    assert "Read and fix <path>[:line]" in fix_crash_h
    assert "must produce textual code changes; pure no-op is invalid." in fix_crash_u
    assert "do not bypass acceptance by tampering" in fix_crash_u
    assert "Read and fix <path>[:line]" in fix_crash_u


def test_seed_generation_skill_enforces_real_archive_first_policy() -> None:
    seed = _load("seed_generation")
    assert "real archive samples first" in seed
    assert "contrib/oss-fuzz/corpus.zip" in seed
    assert "Avoid hand-crafted magic-only files" in seed
    assert "malformed/truncated seeds <= 30%" in seed
    assert "at least one semantically valid archive sample exists" in seed
