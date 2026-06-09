"""Vulnerability-directed scoring, attack-hint, and candidate-priority helpers.

Pure functions extracted from workflow_graph.py:
  - `_vuln_*` configuration knobs (env-driven thresholds/weights)
  - security-signal scoring (`_compute_security_signal_scores`,
    `_derive_security_priority`, `_extract_security_scores`, ...)
  - attack-hint synthesis (`_candidate_attack_hint`, `_normalize_attack_hint`,
    `_attack_boundary_values`, signal→category/sanitizer mappings)
  - `_candidate_priority` ranking

This module is a leaf: it depends only on the standard library. It must not
import workflow_graph (no cycles). workflow_graph re-exports every name here so
existing call sites and tests that reference `workflow_graph._<name>` keep
working unchanged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


def _vuln_hunting_enabled() -> bool:
    raw = (os.environ.get("SHERPA_VULN_HUNTING_ENABLED") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _vuln_score_mode() -> str:
    raw = (os.environ.get("SHERPA_VULN_SCORE_MODE") or "risk_first_v1").strip().lower()
    if raw in {"risk_first_v1"}:
        return raw
    return "risk_first_v1"


def _vuln_internal_api_min_score() -> float:
    raw = (os.environ.get("SHERPA_VULN_INTERNAL_API_MIN_SCORE") or "0.75").strip()
    try:
        return max(0.0, min(float(raw), 1.0))
    except Exception:
        return 0.75


def _vuln_non_public_reachability_cap() -> float:
    """Reachability ceiling applied to non-public candidates so public targets
    rank ahead. Overridable via SHERPA_VULN_NON_PUBLIC_REACHABILITY_CAP."""
    raw = (os.environ.get("SHERPA_VULN_NON_PUBLIC_REACHABILITY_CAP") or "0.30").strip()
    try:
        return max(0.0, min(float(raw), 1.0))
    except Exception:
        return 0.30


def _vuln_min_evidence_confidence() -> float:
    raw = (os.environ.get("SHERPA_VULN_MIN_EVIDENCE_CONFIDENCE") or "0.45").strip()
    try:
        return max(0.0, min(float(raw), 1.0))
    except Exception:
        return 0.45


def _vuln_topk() -> int:
    raw = (os.environ.get("SHERPA_VULN_TOPK") or "24").strip()
    try:
        return max(1, min(int(raw), 80))
    except Exception:
        return 24


def _vuln_max_iterations_per_candidate() -> int:
    raw = (os.environ.get("SHERPA_VULN_MAX_ITERATIONS_PER_CANDIDATE") or "5").strip()
    try:
        return max(1, min(int(raw), 50))
    except Exception:
        return 5


_VULN_REPLAN_PRIORITY_THRESHOLD = 0.65


def _vuln_replan_priority_threshold() -> float:
    raw = os.environ.get("SHERPA_VULN_REPLAN_PRIORITY_THRESHOLD") or ""
    if raw:
        try:
            return max(0.0, min(float(raw.strip()), 1.0))
        except ValueError:
            pass
    return _VULN_REPLAN_PRIORITY_THRESHOLD


def _vuln_score_weights() -> dict[str, float]:
    # Pure vuln-driven scoring. All non-vuln factors removed.
    return {
        "vuln_likelihood": 0.50,
        "exploitability": 0.30,
        "reachability_confidence": 0.20,
    }


def _security_signal_ids() -> tuple[str, ...]:
    return (
        "mem_oob_candidate",
        "integer_overflow_candidate",
        "format_string_candidate",
        "path_traversal_candidate",
        "command_injection_candidate",
        "authz_bypass_candidate",
        "null_deref_candidate",
        "uaf_candidate",
    )


def _security_signal_patterns() -> dict[str, str]:
    return {
        "mem_oob_candidate": r"(memcpy|memmove|strcpy|strncpy|strcat|strncat|\[[^\]]+\]|pointer|offset|index|bounds?)",
        "integer_overflow_candidate": (
            r"(overflow|underflow|wrap(?:around)?|truncat|narrow(?:ing)?|"
            r"(?:width|height|stride|rowbytes|rowsize|pitch|offset|size|len|length|capacity)"
            r"\s*(?:[\*\+\-]|<<)|"
            r"(?:[\*\+\-]|<<)\s*(?:width|height|stride|rowbytes|rowsize|pitch|offset|size|len|length|capacity)|"
            r"(?:size_t|ssize_t|uint(?:8|16|32|64)?_t|int(?:8|16|32|64)?_t)\s*\)|"
            r"(?:alloc|malloc|calloc|realloc)[^;\n]{0,80}(?:width|height|stride|rowbytes|rowsize|size|len|length|capacity))"
        ),
        "format_string_candidate": r"(printf|fprintf|sprintf|snprintf|vsnprintf|vprintf|format|string_format|fmt::)",
        "path_traversal_candidate": r"(path|filepath|filename|fopen|open\(|readfile|writefile|\.\./)",
        "command_injection_candidate": r"(system\(|popen\(|exec\(|spawn\(|shell|command)",
        "authz_bypass_candidate": r"(auth|authorize|permission|acl|role|token|session|bypass|skip[_-]?check)",
        "null_deref_candidate": r"(null|nullptr|none|optional|dereference|->)",
        "uaf_candidate": r"(free\(|delete|release|destroy|dispose|lifetime|dangling)",
    }


def _empty_security_scores() -> dict[str, float]:
    return {signal: 0.0 for signal in _security_signal_ids()}


def _compute_security_signal_scores(
    *,
    name: str,
    signature: str,
    file_hint: str,
    risk_signals: list[str] | None = None,
    risk_signal_source_breakdown: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    text = f"{name}\n{signature}\n{file_hint}".lower()
    scores = _empty_security_scores()
    signals = {str(x).strip().lower() for x in list(risk_signals or []) if str(x).strip()}
    source_breakdown: dict[str, set[str]] = {}
    for source_name, values in dict(risk_signal_source_breakdown or {}).items():
        if not isinstance(values, list):
            continue
        source_breakdown[str(source_name).strip().lower()] = {
            str(x).strip().lower() for x in values if str(x).strip()
        }
    for signal_id, pattern in _security_signal_patterns().items():
        if re.search(pattern, text, re.IGNORECASE):
            scores[signal_id] = max(scores[signal_id], 0.62)
        if signal_id in source_breakdown.get("regex", set()) or signal_id in source_breakdown.get("semantic", set()):
            scores[signal_id] = max(scores[signal_id], 0.78)
        elif signal_id in source_breakdown.get("weak_file", set()) or signal_id in source_breakdown.get("file", set()):
            scores[signal_id] = max(scores[signal_id], 0.44)
        elif signal_id in signals:
            scores[signal_id] = max(scores[signal_id], 0.68)
    if "bounds" in signals:
        scores["mem_oob_candidate"] = max(scores["mem_oob_candidate"], 0.68)
        scores["integer_overflow_candidate"] = max(scores["integer_overflow_candidate"], 0.56)
    if "parser-like" in signals or "state-machine" in signals:
        scores["null_deref_candidate"] = max(scores["null_deref_candidate"], 0.5)
    return {k: round(max(0.0, min(float(v), 1.0)), 4) for k, v in scores.items()}


def _derive_security_priority(
    *,
    target_type: str,
    runtime_viability: str,
    security_scores: dict[str, float] | None = None,
) -> tuple[float, float, float, str]:
    scores = dict(_empty_security_scores())
    for key, value in dict(security_scores or {}).items():
        if key in scores:
            try:
                scores[key] = max(0.0, min(float(value), 1.0))
            except Exception:
                scores[key] = 0.0
    non_zero = [float(v) for v in scores.values() if float(v) > 0.0]
    non_zero.sort(reverse=True)
    top = non_zero[0] if non_zero else 0.0
    top3_avg = (sum(non_zero[:3]) / min(3, len(non_zero))) if non_zero else 0.0
    target_type_l = str(target_type or "").strip().lower()
    runtime_viability_l = str(runtime_viability or "").strip().lower()

    vuln_likelihood = 0.65 * top + 0.35 * top3_avg
    if target_type_l in {"parser", "decoder", "archive", "document"}:
        vuln_likelihood += 0.06

    exploitability = (
        0.50 * max(scores["mem_oob_candidate"], scores["uaf_candidate"])
        + 0.22 * scores["integer_overflow_candidate"]
        + 0.14 * scores["command_injection_candidate"]
        + 0.08 * scores["path_traversal_candidate"]
        + 0.06 * scores["authz_bypass_candidate"]
    )

    reachability = {"high": 0.82, "medium": 0.62, "low": 0.40}.get(runtime_viability_l, 0.5)
    if target_type_l in {"parser", "decoder", "archive"}:
        reachability += 0.08
    if scores["format_string_candidate"] > 0.0 or scores["null_deref_candidate"] > 0.0:
        reachability += 0.03

    vuln_likelihood = max(0.0, min(vuln_likelihood, 1.0))
    exploitability = max(0.0, min(exploitability, 1.0))
    reachability = max(0.0, min(reachability, 1.0))

    ordered = sorted(scores.items(), key=lambda kv: float(kv[1]), reverse=True)
    reason_parts = [f"{sig}:{score:.2f}" for sig, score in ordered if float(score) > 0.0][:3]
    reason = ", ".join(reason_parts) if reason_parts else "no_strong_security_signal"
    return (
        round(vuln_likelihood, 4),
        round(exploitability, 4),
        round(reachability, 4),
        reason,
    )


def _extract_security_scores(item: dict[str, Any]) -> dict[str, float]:
    raw = item.get("security_signal_scores")
    if not isinstance(raw, dict):
        return _empty_security_scores()
    out = _empty_security_scores()
    for key in _security_signal_ids():
        try:
            out[key] = max(0.0, min(float(raw.get(key) or 0.0), 1.0))
        except Exception:
            out[key] = 0.0
    return out


def _top_security_signals(
    security_scores: dict[str, float] | None,
    *,
    threshold: float | None = None,
) -> list[str]:
    th = _vuln_min_evidence_confidence() if threshold is None else max(0.0, min(float(threshold), 1.0))
    pairs = sorted(
        ((str(k), float(v)) for k, v in dict(security_scores or {}).items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return [sig for sig, score in pairs if score >= th]


def _signal_slug(signal_id: str) -> str:
    raw = str(signal_id or "").strip().lower()
    if not raw:
        return "generic"
    raw = re.sub(r"_candidate$", "", raw)
    raw = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return raw or "generic"


def _signal_vuln_category(signal_id: str) -> str:
    mapping = {
        "mem_oob_candidate": "heap-buffer-overflow",
        "integer_overflow_candidate": "integer-overflow",
        "format_string_candidate": "format-string",
        "path_traversal_candidate": "path-traversal",
        "command_injection_candidate": "command-injection",
        "authz_bypass_candidate": "authorization-bypass",
        "null_deref_candidate": "null-dereference",
        "uaf_candidate": "use-after-free",
    }
    return mapping.get(str(signal_id or "").strip().lower(), "memory-corruption")


def _signal_sanitizer_hint(signal_id: str) -> str:
    mapping = {
        "mem_oob_candidate": "address",
        "integer_overflow_candidate": "undefined",
        "format_string_candidate": "address",
        "path_traversal_candidate": "address",
        "command_injection_candidate": "address",
        "authz_bypass_candidate": "address",
        "null_deref_candidate": "address",
        "uaf_candidate": "address",
    }
    return mapping.get(str(signal_id or "").strip().lower(), "address")


def _attack_boundary_values(signal_id: str, *, target_type: str, api: str) -> list[str]:
    signal = str(signal_id or "").strip().lower()
    target_type_l = str(target_type or "").strip().lower()
    api_l = str(api or "").strip().lower()

    if signal == "mem_oob_candidate":
        return ["len=0", "len=1", "len=4096", "offset=len-1", "offset=0xFFFFFFFF"]
    if signal == "integer_overflow_candidate":
        return ["count=0x7FFFFFFF", "count=0x80000000", "size=0xFFFFFFFF", "count*stride overflow"]
    if signal == "format_string_candidate":
        return ["fmt=%n", "fmt=%999999s", "fmt=%p%p%p", "fmt={{{{{"]
    if signal == "path_traversal_candidate":
        return ["path=../etc/passwd", "path=..\\\\windows\\\\win.ini", "path=/tmp/../../secret"]
    if signal == "command_injection_candidate":
        return ["arg=;id", "arg=$(id)", "arg=`id`", "arg=&&touch /tmp/pwned"]
    if signal == "authz_bypass_candidate":
        return ["role=admin", "token=''", "user_id=-1", "permission=wildcard"]
    if signal == "null_deref_candidate":
        return ["ptr=null", "count=0", "header_only=true", "optional_field_missing"]
    if signal == "uaf_candidate":
        return ["free_then_use", "double_close", "lifetime=end_before_use", "refcount=0"]

    if target_type_l in {"parser", "decoder", "archive", "document"}:
        return ["size=0", "size=1", "size=4096", "declared_len=0xFFFFFFFF"]
    if "parse" in api_l or "decode" in api_l:
        return ["input=''", "input='\\x00'", "input='A'*4096", "declared_len=actual_len+1"]
    return ["size=0", "size=1", "size=4096", "count=0xFFFFFFFF"]


def _candidate_attack_hint(
    *,
    api: str,
    target_type: str,
    signal_id: str,
    source_path: str,
    security_reason: str,
) -> dict[str, Any]:
    signal = str(signal_id or "").strip().lower()
    api_text = str(api or "").strip() or "target_api"
    source_text = str(source_path or "").strip()
    vuln_category = _signal_vuln_category(signal)
    key_code_path = [api_text]
    if source_text:
        source_stem = Path(source_text).stem.strip()
        if source_stem and source_stem not in {api_text, api_text.split("::")[-1]}:
            key_code_path.append(source_stem)
    trigger_condition = {
        "mem_oob_candidate": f"{api_text} uses attacker-influenced length or offset beyond allocated bounds",
        "integer_overflow_candidate": f"{api_text} derives buffer or loop sizes from arithmetic that can overflow",
        "format_string_candidate": f"{api_text} forwards attacker-controlled format content into formatter sinks",
        "path_traversal_candidate": f"{api_text} consumes file paths without constraining traversal sequences",
        "command_injection_candidate": f"{api_text} reaches shell or process execution with partially controlled input",
        "authz_bypass_candidate": f"{api_text} evaluates authorization decisions with bypass-prone state combinations",
        "null_deref_candidate": f"{api_text} dereferences optional or stateful pointers on malformed input",
        "uaf_candidate": f"{api_text} may access released state after cleanup or ownership transfer",
    }.get(signal, f"{api_text} exposes a high-risk path that warrants adversarial input exploration")
    if security_reason:
        trigger_condition = f"{trigger_condition}; evidence={security_reason}"
    return {
        "trigger_condition": trigger_condition,
        "key_code_path": key_code_path,
        "boundary_values": _attack_boundary_values(signal, target_type=target_type, api=api_text),
        "vuln_category": vuln_category,
        "sanitizer_hint": _signal_sanitizer_hint(signal),
    }


def _normalize_attack_hint(
    value: Any,
    *,
    api: str,
    target_type: str,
    signal_id: str,
    source_path: str,
    security_reason: str,
) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        hint = _candidate_attack_hint(
            api=api,
            target_type=target_type,
            signal_id=signal_id,
            source_path=source_path,
            security_reason=security_reason,
        )
        hint["trigger_condition"] = value.strip()
        return hint
    return {}


def _candidate_priority(
    *,
    vuln_likelihood: float,
    exploitability: float,
    reachability_confidence: float,
    evidence_count: int,
    signal_score: float,
) -> float:
    raw = (
        0.50 * max(0.0, min(float(vuln_likelihood), 1.0))
        + 0.24 * max(0.0, min(float(exploitability), 1.0))
        + 0.18 * max(0.0, min(float(reachability_confidence), 1.0))
        + 0.05 * max(0.0, min(float(signal_score), 1.0))
        + 0.03 * min(max(int(evidence_count), 0), 5) / 5.0
    )
    return round(max(0.0, min(raw, 1.0)), 4)
