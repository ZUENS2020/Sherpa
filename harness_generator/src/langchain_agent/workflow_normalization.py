from __future__ import annotations

from typing import Any

import workflow_common as _wf_common


_BINARY_FORMAT_HINTS = {
    "png",
    "libpng",
    "ihdr",
    "plte",
    "idat",
    "iend",
    "iccp",
    "splt",
    "jpeg",
    "jpg",
    "gif",
    "webp",
    "tiff",
    "bmp",
    "zip",
    "gzip",
    "zlib",
    "deflate",
    "inflate",
    "archive",
    "chunk",
    "crc",
    "checksum",
}


def _looks_like_binary_format_parser(*parts: str) -> bool:
    text = " ".join(p for p in parts if p).lower()
    if not text:
        return False
    return any(hint in text for hint in _BINARY_FORMAT_HINTS)


def infer_seed_profile(name: str, context: str, *, target_type: str) -> str:
    normalized_target_type = str(target_type or "").strip().lower()
    text = f"{name}\n{context}".lower()
    if normalized_target_type == "parser":
        if _looks_like_binary_format_parser(name, context):
            return "decoder-binary"
        if any(tok in text for tok in ("arg_id", "argument id", "positional", "named argument", "named arg", "number", "numeric")):
            return "parser-numeric"
        if any(tok in text for tok in ("format", "replacement field", "specifier", "brace", "printf", "fmt")):
            return "parser-format"
        if any(tok in text for tok in ("token", "lexer", "lex", "scan", "scanner", "read_", "readline", "read line")):
            return "parser-token"
        return "parser-structure"
    mapping = {
        "decoder": "decoder-binary",
        "image": "decoder-binary",
        "archive": "archive-container",
        "serializer": "serializer-structured",
        "document": "document-text",
        "network": "network-message",
    }
    return mapping.get(normalized_target_type, "generic")


def normalize_seed_profile(seed_profile: str, *, target_type: str, name: str, context: str) -> str:
    normalized = str(seed_profile or "").strip().lower()
    normalized_target_type = str(target_type or "").strip().lower()
    if normalized not in _wf_common.ALLOWED_SEED_PROFILES:
        normalized = ""
    if not normalized or normalized == "pending":
        return infer_seed_profile(name, context, target_type=normalized_target_type)
    if normalized_target_type == "parser" and normalized.startswith("parser-") and _looks_like_binary_format_parser(name, context):
        return "decoder-binary"
    if normalized_target_type != "parser" and normalized.startswith("parser-"):
        return infer_seed_profile(name, context, target_type=normalized_target_type)
    return normalized


def normalize_target_identity(*, target_name: str, target_api: str) -> dict[str, str]:
    name = str(target_name or "").strip()
    api = str(target_api or "").strip()
    if not name:
        name = api
    if not api:
        api = name
    return {"target_name": name, "target_api": api}


def _normalize_identity_token(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _target_type_from_run_details(
    run_details: Any,
    *,
    target_name: str,
    target_api: str,
) -> str:
    if not isinstance(run_details, list):
        return ""
    wanted = {
        _normalize_identity_token(target_name),
        _normalize_identity_token(target_api),
    }
    wanted.discard("")
    details = [item for item in run_details if isinstance(item, dict)]
    for detail in details:
        candidates = {
            _normalize_identity_token(str(detail.get("target_name") or "")),
            _normalize_identity_token(str(detail.get("target_api") or "")),
            _normalize_identity_token(str(detail.get("fuzzer") or "")),
        }
        if wanted and not (wanted & candidates):
            continue
        target_type = str(detail.get("target_type") or "").strip().lower()
        if target_type:
            return target_type
    for detail in details:
        target_type = str(detail.get("target_type") or "").strip().lower()
        if target_type:
            return target_type
    return ""


def normalize_target_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row or {})
    target_name = str(out.get("target_name") or out.get("target") or out.get("name") or "").strip()
    target_api = str(out.get("api") or target_name).strip()
    target_type = str(out.get("target_type") or "generic").strip().lower()
    identity = normalize_target_identity(target_name=target_name, target_api=target_api)
    out["target_name"] = identity["target_name"]
    out["name"] = identity["target_name"]
    out["target"] = identity["target_name"]
    out["api"] = identity["target_api"]
    out["target_type"] = target_type
    out["seed_profile"] = normalize_seed_profile(
        str(out.get("seed_profile") or ""),
        target_type=target_type,
        name=identity["target_name"],
        context=identity["target_api"],
    )
    return out


def normalize_workflow_context(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc or {})
    if "coverage_target_name" in out or "coverage_target_api" in out:
        identity = normalize_target_identity(
            target_name=str(out.get("coverage_target_name") or ""),
            target_api=str(out.get("coverage_target_api") or ""),
        )
        out["coverage_target_name"] = identity["target_name"]
        out["coverage_target_api"] = identity["target_api"]
    target_type = str(out.get("coverage_target_type") or "").strip().lower()
    if not target_type:
        target_type = _target_type_from_run_details(
            out.get("run_details"),
            target_name=str(out.get("coverage_target_name") or ""),
            target_api=str(out.get("coverage_target_api") or ""),
        )
        if target_type:
            out["coverage_target_type"] = target_type
    if "coverage_seed_profile" in out and target_type:
        out["coverage_seed_profile"] = normalize_seed_profile(
            str(out.get("coverage_seed_profile") or ""),
            target_type=target_type,
            name=str(out.get("coverage_target_name") or ""),
            context=str(out.get("coverage_target_api") or ""),
        )
    return out
