from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "harness_generator" / "src" / "langchain_agent"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import contract_analysis as ca  # noqa: E402

NUL_DOC = """
/* Parse a JSON document.
 * @param buf  the input; must be NUL-terminated.
 * @return parsed value, or NULL on error. Caller must free with json_free().
 */
json_value *json_parse(const char *buf, size_t len) {
    return do_parse(buf, len);
}
"""

NONNULL_DOC = """
// settings must not be NULL and must be initialized via json_settings_init
// before calling this function.
int json_parse_ex(json_settings *settings, const char *buf, size_t len) {
    return 0;
}
"""

NO_DOC = """
int helper(int x) { return x + 1; }
"""


def test_extract_nul_terminated_and_ownership():
    c = ca.extract_documented_preconditions(NUL_DOC, "json_parse")
    assert "nul_terminated" in c["preconditions"]
    assert "ownership" in c["preconditions"]


def test_extract_nonnull_and_must_initialize():
    c = ca.extract_documented_preconditions(NONNULL_DOC, "json_parse_ex")
    assert "nonnull" in c["preconditions"]
    assert "must_initialize" in c["preconditions"]


def test_no_doc_returns_empty():
    c = ca.extract_documented_preconditions(NO_DOC, "helper")
    assert c["preconditions"] == []


def test_missing_function_returns_empty():
    c = ca.extract_documented_preconditions(NUL_DOC, "nonexistent")
    assert c["preconditions"] == []


def test_crash_frame_functions_skips_internals():
    log = (
        "==1==ERROR: AddressSanitizer: heap-buffer-overflow\n"
        "    #0 0x1 in __asan_memcpy\n"
        "    #1 0x2 in scan_string json.c:1638\n"
        "    #2 0x3 in json_parse json.c:200\n"
        "    #3 0x4 in LLVMFuzzerTestOneInput h.c:10\n"
        "    #4 0x5 in main\n"
    )
    fns = ca.crash_frame_functions(log)
    assert fns == ["scan_string", "json_parse"]


def test_build_contract_context_end_to_end(tmp_path):
    (tmp_path / "json.c").write_text(NUL_DOC, encoding="utf-8")
    crash = (
        "==1==ERROR: AddressSanitizer: heap-buffer-overflow READ\n"
        "    #0 0x1 in json_parse json.c:5\n"
        "    #1 0x2 in LLVMFuzzerTestOneInput h.c:1\n"
    )
    block = ca.build_contract_triage_context(tmp_path, crash)
    assert "api_contract" in block
    assert "json_parse" in block
    assert "nul_terminated" in block
    # the guidance steering out-of-contract -> harness_bug is present
    assert "harness_bug" in block and "upstream_bug" in block


def test_build_contract_context_empty_when_no_preconditions(tmp_path):
    (tmp_path / "x.c").write_text(NO_DOC, encoding="utf-8")
    crash = "    #0 0x1 in helper x.c:1\n    #1 0x2 in LLVMFuzzerTestOneInput\n"
    assert ca.build_contract_triage_context(tmp_path, crash) == ""
