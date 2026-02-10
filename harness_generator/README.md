# OSS-Fuzz Harness Generation Toolkit

The **Harness Generation Toolkit** automates the workflow of adding new fuzz
harnesses and running them end-to-end. It supports two modes:

- **OSS-Fuzz mode** (traditional OSS-Fuzz projects)
- **Non-OSS-Fuzz mode** (local workflow for arbitrary Git repos)

---

## Contents

```
harness-generator/
├── batch_generate.py          # batch driver (multiple targets)
├── src/                       # Python package with core logic
│   ├── codex_helper.py        # OpenCode CLI wrapper (sentinel + retry logic)
│   ├── harness_generator.py   # OSS-Fuzz orchestrator
│   └── fuzz_unharnessed_repo.py  # Non-OSS-Fuzz local workflow
└── scripts/                   # triage & reporting utilities
    ├── sort_jobs.py
    ├── summarize.py
    ├── generate_reports.py
    └── gather_reports.py
```

---

## 1. Non-OSS-Fuzz Workflow (Local)

The default Web API uses this path. It:
1. Plan target APIs (`fuzz/PLAN.md`, `fuzz/targets.json`)
2. Synthesize harness + `fuzz/build.py`
3. Build (auto-fix if failed)
4. Run fuzzers
5. Analyze crash, generate reports and optional fix patch

### Output Files (repo root)
- `crash_info.md`
- `crash_analysis.md`
- `reproduce.py`
- `fix.patch` / `fix_summary.md` (if fix step is triggered)
- `run_summary.md` / `run_summary.json`
- `challenge_bundle*/`, `false_positive*/`, or `unreproducible*/`

### Output Root
You can force all outputs into a single host directory via:
```
SHERPA_OUTPUT_DIR=/path/to/output
```
Each run creates its own subfolder `<repo>-<8位id>/` under that root.

### Run a single repo (local workflow)
```
python -m src.fuzz_unharnessed_repo --repo https://github.com/user/repo.git \
  --time-budget 900 --max-len 1024 --docker-image auto
```

---

## 2. OSS-Fuzz Workflow

`harness_generator.py` targets a **local OSS-Fuzz checkout** and automates:
1. Baseline build
2. Harness creation
3. Rebuild with retries
4. Seed generation
5. Fuzzer run + crash analysis

### Run a single OSS-Fuzz project
```
python -m src.harness_generator <project> <path/to/oss-fuzz/checkout> <key.env> \
  --sanitizer address --codex-cli opencode --max-retries 3
```

---

## 3. Batch Generation

`batch_generate.py` reads YAML describing multiple targets and runs the workflow
multiple rounds per repo.

```
python batch_generate.py --targets ./yamls/c-projects.yaml --threads 32 --rounds 8
```

Outputs are stored under `./jobs/<project>_<uuid>/` by default.

---

## 4. Triage & Reporting Utilities

| Script | Purpose |
|--------|---------|
| `sort_jobs.py` | Move each job directory into `./sorted/<bucket>` |
| `generate_reports.py` | Create disclosure-style `bug_report.md` |
| `gather_reports.py` | Collect artifacts into `./sorted/reports/` |
| `summarize.py` | Markdown summary of jobs |

---

## 5. Requirements

- Docker + OSS-Fuzz build dependencies (for OSS-Fuzz mode)
- OpenCode CLI in `$PATH` (or specify `--codex-cli`)
- OpenAI-compatible API key via `OPENAI_API_KEY` or `.env` file
- Python ≥ 3.9

---

## 6. Notes

- Crash analysis marks **HARNESS ERROR** for false positives
- Fix patches are recorded as `fix.patch` whenever the fix step runs
- `run_summary.md/json` provides a compact per-run summary for automation

### Domestic network mirrors
- Node.js: `https://npmmirror.com/mirrors/node`
- npm registry: `https://registry.npmmirror.com`
- pip index: `https://pypi.tuna.tsinghua.edu.cn/simple`
- APT mirrors (TUNA): `http://mirrors.tuna.tsinghua.edu.cn`
- Git mirror (default): `https://gitclone.com/github.com/`

### OpenCode safety
By default, `SHERPA_OPENCODE_NO_EXEC=1` blocks build/run commands during OpenCode runs. Only read-only inspection commands are permitted.

### Split OpenCode container
The Web UI runs without OpenCode installed. OpenCode is executed inside a separate image:
- `SHERPA_OPENCODE_DOCKER_IMAGE=sherpa-opencode:latest`
