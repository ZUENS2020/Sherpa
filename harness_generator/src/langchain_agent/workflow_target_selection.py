from __future__ import annotations

from typing import Any, Callable


def sort_ranked_items(
    ranked_items: list[dict[str, Any]],
    *,
    security_priority_mode: bool,
    is_internal_api_symbol_fn: Callable[[str], bool],
    runtime_viability_rank_fn: Callable[[str], int],
    prefer_deeper: bool = False,
) -> list[dict[str, Any]]:
    rows = list(ranked_items)
    if security_priority_mode:
        # Vuln-driven sort on *effective* risk: likelihood minus the demotion
        # penalties (non-core/contrib/helper surface + recent-yield/exhaustion),
        # then exploitability → reachability. Without subtracting the penalty an
        # auxiliary/platform helper (contrib, arm-neon, *_wasm, exhausted target)
        # with a marginally higher raw likelihood outranks a core decoder API,
        # which is exactly the demotion this penalty is meant to apply. Non-vuln
        # factors (coverage_gap, complexity, api_relevance, depth) stay excluded.
        def _effective_risk(row: dict) -> float:
            penalty = float(
                row.get("target_score_penalty")
                or row.get("target_surface_penalty")
                or 0.0
            )
            return float(row.get("vuln_likelihood") or 0.0) - penalty

        rows.sort(
            key=lambda row: (
                -_effective_risk(row),
                -float(row.get("vuln_likelihood") or 0.0),
                -float(row.get("exploitability") or 0.0),
                -float(row.get("reachability_confidence") or 0.0),
                -len(list(row.get("evidence_ids") or [])),
                -len(list(row.get("security_signals") or [])),
                str(row.get("target_name") or ""),
            )
        )
    else:
        rows.sort(
            key=lambda row: (
                -float(row.get("target_score") or 0.0),
                -float(row.get("priority") or 0.0),
                -float(row.get("vuln_likelihood") or 0.0),
                -float(row.get("exploitability") or 0.0),
                -float(row.get("reachability_confidence") or 0.0),
                -int(row.get("depth_score") or 0),
                -runtime_viability_rank_fn(str(row.get("runtime_viability") or "")),
                str(row.get("target_name") or ""),
            )
        )
    return rows


def assign_execution_priority(
    ranked_items: list[dict[str, Any]],
    *,
    max_targets: int,
    security_priority_mode: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked_items):
        updated = dict(row)
        updated["rank"] = int(idx + 1)
        updated["execution_priority"] = int(idx + 1) if idx < max_targets else 0
        if security_priority_mode:
            # In vuln-driven mode, any top-N target is must_run.
            updated["must_run"] = bool(idx < max_targets)
        else:
            target_type = str(updated.get("target_type") or "").strip().lower()
            updated["must_run"] = bool(
                idx < max_targets and (idx == 0 or target_type in {"archive", "parser", "decoder"})
            )
        out.append(updated)
    return out
