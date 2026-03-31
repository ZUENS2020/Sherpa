---
name: analysis
description: Generate pre-plan analysis context artifacts for target selection and repair-aware planning.
compatibility: opencode
metadata:
  stage: analysis
  owner: sherpa
---

## What this skill does
Builds analysis context files consumed by downstream `plan` and `synthesize` stages.

## When to use this skill
Use this skill in the dedicated `analysis` stage before `plan`.

## Required inputs
- repository source tree (read-only)
- MCP tools from task-scoped PromeFuzz companion (HTTP MCP), when available
- preferred MCP tools in this round:
  - preprocessor: `run_ast_preprocessor`, `extract_api_functions`, `build_library_callgraph`
  - semantic (if enabled): `init_knowledge_base`, `retrieve_documents`, `comprehend_*`
- optional companion outputs under `/shared/output/_k8s_jobs/<job-id>/promefuzz/` as fallback
- previous repair context from coordinator hint (if provided)

## Required inputs (additional)
- `fuzz/target_analysis.json` (preliminary target analysis from regex/tree-sitter heuristics)

## Required outputs
- `fuzz/analysis_context.json`
- `fuzz/antlr_plan_context.json` (if grammar/static context is available)
- `fuzz/target_analysis.json` (verified/corrected — must contain `analysis_source: "opencode-verified"` entries)

## Workflow
1. Query MCP evidence first when MCP is available.
   Use preprocessor tools first, then semantic tools for evidence-backed summaries.
2. Read existing analysis artifacts (if any) and companion file outputs as fallback.
3. Refresh static analysis summaries for grammar/target context.
4. Verify and correct `target_type` for each entry in `fuzz/target_analysis.json` (`recommended_targets` and `candidate_functions`):
   - Read the function's source code, signature, file context, and MCP evidence.
   - Cross-check the preliminary `target_type` (from regex/keyword rules) against actual semantics.
   - Apply the target-type classification rules below.
   - When correcting `target_type`, also update `seed_profile` to match.
   - Set `"analysis_source": "opencode-verified"` on every verified/corrected entry.
   - Write the updated `fuzz/target_analysis.json` back to disk.
5. Merge findings into `fuzz/analysis_context.json` with concise evidence.
6. Ensure downstream plan can consume paths and summaries directly.

## Target-type classification

Preliminary `target_type` values in `fuzz/target_analysis.json` come from keyword heuristics and may be wrong.
You MUST verify each entry and correct misclassifications using semantic evidence.

### Type definitions and disambiguation

- **decoder**: Raw format decoders / compression primitives.
  - Expects: precise binary stream (raw DEFLATE, raw LZ, PNG chunks, JPEG segments).
  - Keywords: inflate, deflate, lz, zstd, lzma, brotli, decode, decompress, unpack.
  - Common misclassification: functions like `inflateBack9`, `LZ4_decompress_safe` are decoders, NOT archive.
  - Seed profile: `decoder-binary`.

- **archive**: Container-format wrappers that open/extract multi-file archives.
  - Expects: container-encapsulated format (gzip file, zip file, tar file, rar file).
  - Keywords: gzip, gunzip, zip, unzip, tar, untar, rar, 7z, archive.
  - Key distinction from decoder: archive functions handle container headers/metadata/file entries;
    decoders handle raw compressed byte streams.
  - Seed profile: `archive-container`.

- **parser**: Structured text/data format parsers.
  - Expects: text or structured data (JSON, XML, YAML, format strings, tokens).
  - Keywords: parse, scan, lex, token, json, xml, yaml, format, printf, specifier.
  - Sub-profiles: `parser-structure` (default), `parser-token` (lexer/scanner), `parser-format` (format-string), `parser-numeric` (numeric arg parsing).

- **image**: Image format processors (png, jpeg, gif, bmp, webp). Seed: `decoder-binary`.
- **document**: Document format processors (pdf, html, markdown). Seed: `document-text`.
- **network**: Network protocol handlers (http, tls, dns, packet). Seed: `network-message`.
- **database**: Database/query processors (sql, sqlite). Seed: `generic`.
- **serializer**: Serialization/encoding emitters (emit, dump, serialize). Seed: `serializer-structured`.
- **interpreter**: Script/bytecode evaluators (eval, vm, compile, bytecode). Seed: `generic`.
- **generic**: Use only when no semantic evidence supports a specific type.

### Allowed values

- `target_type`: parser, decoder, archive, image, document, network, database, serializer, interpreter, generic.
- `seed_profile`: parser-structure, parser-token, parser-format, parser-numeric, decoder-binary, archive-container, serializer-structured, document-text, network-message, generic.

## Command policy
- Allowed: read-only commands (`find`, `rg`, `grep`, `cat`, `head`, `tail`, read-only `sed`, `ls`).
- Forbidden: build/execute/mutation commands.

## Degraded mode
- If MCP is unavailable or returns invalid output, continue using local/static evidence.
- If semantic MCP tools are unavailable, continue with preprocessor evidence and mark degraded reason.
- Record degraded reason in `fuzz/analysis_context.json` instead of silently skipping MCP evidence.

## Done contract
- Write `fuzz/analysis_context.json` into `./done`.
