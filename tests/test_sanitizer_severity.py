from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "harness_generator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fuzz_unharnessed_repo as f  # noqa: E402

# The exact UBSan report produced by json-parser's pointer-as-accumulator idiom
# (json.c:437) — benign undefined behavior that must NOT be treated as a crash.
BENIGN_UB_LOG = (
    "/src/json.c:437:34: runtime error: applying non-zero offset 2 to null pointer\n"
    "SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior /src/json.c:437:34\n"
)

HEAP_OVERFLOW_LOG = (
    "==123==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xdead\n"
    "    #0 0x1 in parse foo.c:10\n"
    "SUMMARY: AddressSanitizer: heap-buffer-overflow foo.c:10 in parse\n"
)

SEGV_LOG = (
    "==474161==ERROR: AddressSanitizer: SEGV on unknown address 0xfffffffffffffff8\n"
    "SUMMARY: AddressSanitizer: SEGV json.c:1039 in json_value_free_ex\n"
)

NULL_DEREF_UB_LOG = (
    "x.c:5:7: runtime error: member access within null pointer of type 'struct S'\n"
    "SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior x.c:5:7\n"
)


def test_benign_pointer_overflow_is_benign_ub():
    assert f.classify_sanitizer_severity(BENIGN_UB_LOG) == "benign_ub"


def test_heap_overflow_is_memory_safety():
    assert f.classify_sanitizer_severity(HEAP_OVERFLOW_LOG) == "memory_safety"


def test_segv_is_memory_safety():
    assert f.classify_sanitizer_severity(SEGV_LOG) == "memory_safety"


def test_null_deref_ub_is_not_benign():
    # *Dereferencing* null is a real bug — only *offset-to-null* is benign.
    assert f.classify_sanitizer_severity(NULL_DEREF_UB_LOG) == "memory_safety"


def test_clean_log_is_none():
    assert f.classify_sanitizer_severity("#1024 pulse cov: 35 ft: 35\n") == "none"


def test_benign_ub_with_real_asan_is_memory_safety():
    # A benign UB co-occurring with a real ASan error must still be a crash.
    combined = BENIGN_UB_LOG + HEAP_OVERFLOW_LOG
    assert f.classify_sanitizer_severity(combined) == "memory_safety"


def test_unknown_sanitizer_report_is_other_not_benign():
    other = (
        "==1==ERROR: ThreadSanitizer: data race\n"
        "SUMMARY: ThreadSanitizer: data race t.c:3\n"
    )
    assert f.classify_sanitizer_severity(other) == "other"


def test_signature_includes_severity():
    sig = f.extract_crash_stack_signature(BENIGN_UB_LOG)
    assert sig["severity"] == "benign_ub"
    sig2 = f.extract_crash_stack_signature(HEAP_OVERFLOW_LOG)
    assert sig2["severity"] == "memory_safety"


def test_ubsan_run_options_non_fatal_by_default():
    # Default fuzzing runs must keep UBSan non-fatal so benign UB cannot starve
    # the corpus.
    assert "halt_on_error=0" in f._UBSAN_RUN_OPTIONS
    assert f.SANITIZER_CONFIGS["address"]["ubsan_options"] == f._UBSAN_RUN_OPTIONS


def test_benign_ub_nonfatal_toggle_default_on():
    assert f._benign_ub_nonfatal() is True
