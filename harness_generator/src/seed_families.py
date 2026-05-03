from __future__ import annotations


DECODER_BINARY_SEED_FAMILIES = [
    "magic_headers",
    "chunk_layout",
    "length_boundary_values",
    "checksum_crc_variants",
    "truncated_sections",
    "metadata_chunks",
    "compressed_payload_variants",
]

PNG_SEED_FAMILIES = [
    "png_signature",
    "png_chunk_order",
    "png_crc_variants",
    "png_ihdr_dimensions",
    "png_idat_payloads",
    "png_ancillary_chunks",
]


def is_fmt_format_target(*parts: str) -> bool:
    text = " ".join(p for p in parts if p).lower()
    return bool(
        "fmt" in text
        and any(tok in text for tok in ("format", "format_to", "vformat", "println", "print", "replacement field", "specifier"))
    )


def seed_families_for_target(seed_profile: str, *parts: str) -> tuple[list[str], list[str]]:
    """Return advisory seed families for seed generation and feedback.

    Suggested families are guidance, not control-plane requirements. Empty
    suggested lists are valid and must not be treated as missing values.
    """
    profile = str(seed_profile or "").strip().lower()
    text = " ".join(p for p in parts if p).lower()
    suggested: list[str] = []
    optional: list[str] = []

    if profile == "parser-format" and is_fmt_format_target(text):
        suggested.extend(
            [
                "replacement_fields",
                "escaped_braces",
                "positional_arguments",
                "format_specifiers",
                "width_precision",
                "fill_align",
                "type_conversions",
                "malformed_replacement_fields",
            ]
        )
        return suggested, optional
    if profile == "parser-structure":
        suggested.extend(["document_markers", "block_scalars", "anchors_aliases", "tags_directives"])
        optional.extend(["flow_structures", "unterminated_fragments", "malformed_separators"])
    elif profile == "parser-token":
        suggested.extend(["delimiter_fragments", "unterminated_fragments", "malformed_separators"])
        optional.extend(["document_markers", "tags_directives", "flow_structures"])
    elif profile == "parser-format":
        suggested.extend(["delimiter_fragments", "unterminated_fragments", "malformed_separators"])
    elif profile == "parser-numeric":
        suggested.extend(["delimiter_fragments", "malformed_separators"])
    elif profile == "decoder-binary":
        optional.extend(DECODER_BINARY_SEED_FAMILIES)
        if any(tok in text for tok in ("png", "libpng", "ihdr", "idat", "iccp", "splt")):
            optional.extend(PNG_SEED_FAMILIES)

    if profile.startswith("parser-") and any(tok in text for tok in ("yaml", "yml")):
        for family in [
            "flow_structures",
            "block_scalars",
            "anchors_aliases",
            "tags_directives",
            "document_markers",
            "delimiter_fragments",
            "unterminated_fragments",
            "malformed_separators",
        ]:
            if family not in suggested:
                suggested.append(family)
    return suggested, [x for x in optional if x not in suggested]
