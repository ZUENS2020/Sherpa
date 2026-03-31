---
name: seed_generation
description: Generate high-signal seed corpus with real samples first and controlled synthetic expansion.
compatibility: opencode
metadata:
  stage: seed-generation
  owner: sherpa
---

## What this skill does
Creates warm-up corpus files and seed diagnostics for a target fuzzer, favoring valid and diverse inputs.

## When to use this skill
Use this skill during pre-run seed generation and seed repair cycles.

## Required inputs
- `fuzz/observed_target.json` (when present)
- `fuzz/selected_targets.json`
- `fuzz/target_analysis.json`
- harness source under `fuzz/`
- current corpus directory for active fuzzer

## Required outputs
- seed files under `fuzz/corpus/<fuzzer_name>/`
- `seed_exploration_<fuzzer>.json`
- `seed_check_<fuzzer>.json`

## Workflow
1. Explore target format by reading the target function's source code.
2. **Determine correct seed format** by analyzing the function's actual input requirements (see seed format classification below).
3. Import real samples first, then add controlled synthetic variants.
4. Keep family coverage balanced and noise bounded.
5. Write required seed diagnostics JSON files.

## Seed format classification

**You are the primary classifier for seed format.** Read the function code and determine what input format it actually expects.

### Format types and examples

- **decoder-binary**: Raw format decoders / compression primitives.
  - Expects: precise binary stream (raw DEFLATE, raw LZ, PNG chunks, JPEG segments).
  - Examples: inflateBack9, LZ4_decompress_safe, png_read_header
  - Common mistake: inflateBack9 is NOT archive format - it expects raw DEFLATE64 data, not gzip/zip.

- **archive-container**: Container-format wrappers that open/extract multi-file archives.
  - Expects: container-encapsulated format (gzip file, zip file, tar file, rar file).
  - Examples: gunzip, unzip, tar_extract
  - Key difference from decoder: archive functions handle container headers/metadata.

- **parser-structure**: Structured text/data format parsers.
  - Expects: text or structured data (JSON, XML, YAML).
  - Examples: json_parse, yaml_scan, xml_tokenize

- **parser-token**: Lexer/scanner for text tokenization.
  - Expects: raw text
  - Examples: lex, tokenize, read_line

- **parser-format**: Format string parsing.
  - Expects: format strings with specifiers
  - Examples: printf, scanf, format_parse

- **parser-numeric**: Numeric argument parsing.
  - Expects: numeric strings
  - Examples: atoi, strtod, parse_int

- **generic**: Use when no specific format is identified or for I/O wrapper functions.

### How to determine the correct format

1. Read the target function's source code from the repository
2. Look at:
   - Function signature (parameters tell you what input type is expected)
   - Function body (how does it process the input?)
   - Comments or documentation
3. Check if the function:
   - Directly processes raw bytes → decoder-binary
   - Calls container extraction (unzip, tar, gzip header parsing) → archive-container
   - Parses structured text → parser type
   - Is just a wrapper around standard I/O (fread, read, etc.) → generic

### Important: inflate vs gzip distinction

- `inflateBack9` → **decoder-binary** (raw DEFLATE64 decompression)
- `inflateInit2` → generic (just initialization, no actual decompression)
- `gzip_open` / `gunzip` → **archive-container** (handles gzip container format)
- `unzipOpen` / `unzip` → **archive-container** (handles zip container format)

## Constraints
- Global filtering defaults to `soft` mode:
  - preserve semantically distinct seeds
  - still reject oversized files and exact-content duplicates
  - avoid malformed-only growth when required families are missing
- For `archive-container`:
  - use real archive samples first (`contrib/oss-fuzz/corpus.zip`, `contrib/oss-fuzz/**`, `test/**`, `tests/**`)
  - avoid hand-crafted magic-only files
  - keep malformed/truncated seeds <= 30%
  - ensure at least one semantically valid archive sample exists
- `seed_exploration_*.json` and `seed_check_*.json` must follow coordinator-required schema.
- When diagnostics include concrete paths, use `Read and fix <path>[:line]`.

## Command policy
- Allowed: read-only commands only.
- Forbidden: build/execute commands.

## Acceptance checklist
- Required family buckets are covered or explicitly documented with reason.
- Corpus is not dominated by malformed archive samples.
- Valid real samples are present before synthetic edge cases.
- Seed format matches the function's actual input requirements (not just the preliminary `seed_profile` from `target_analysis.json`).

## Done contract
- Write one created/updated seed file path into `./done`.