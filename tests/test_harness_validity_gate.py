from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "harness_generator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import fuzz_unharnessed_repo as f  # noqa: E402

# A harness-origin crash: top user frame is in a *_fuzz.c file (jsmn dump()).
HARNESS_CRASH = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow READ of size 4\n"
    "    #0 0x1 in __asan_memcpy\n"
    "    #1 0x2 in dump /src/fuzz/dump_fuzz.c:42:9\n"
    "    #2 0x3 in LLVMFuzzerTestOneInput /src/fuzz/dump_fuzz.c:61:3\n"
    "SUMMARY: AddressSanitizer: heap-buffer-overflow\n"
)

# A library-origin crash: top user frame is in the library (json.h). The path
# even contains '/fuzz/../json.h' — must NOT be mistaken for a harness file.
LIBRARY_CRASH = (
    "==1==ERROR: AddressSanitizer: heap-buffer-overflow READ of size 1\n"
    "    #0 0x1 in json_parse_number /src/json.h-abc/fuzz/../json.h:1925:18\n"
    "    #1 0x2 in json_parse_ex /src/json.h-abc/fuzz/../json.h:2212:3\n"
    "    #2 0x3 in LLVMFuzzerTestOneInput /src/json.h-abc/fuzz/json_parse_ex_fuzz.cc:10:31\n"
    "SUMMARY: AddressSanitizer: heap-buffer-overflow\n"
)

# A harness-origin leak: the allocation's first user frame is the harness file.
HARNESS_LEAK = (
    "==1==ERROR: LeakSanitizer: detected memory leaks\n"
    "Direct leak of 61 byte(s) in 1 object(s) allocated from:\n"
    "    #0 0x1 in malloc\n"
    "    #1 0x2 in fuzz_file_reader /src/fuzz/tinyobj_parse_obj_fuzz.c:20:10\n"
    "    #2 0x3 in tinyobj_parse_obj /src/tinyobj_loader_c.h:500\n"
    "SUMMARY: AddressSanitizer: 61 byte(s) leaked\n"
)

CLEAN = "#1024 pulse cov: 35 ft: 35 corp: 6/549b exec/s: 744\n"


def test_first_user_frame_skips_sanitizer_internals():
    assert f.crash_first_user_frame_file(HARNESS_CRASH) == "dump_fuzz.c"
    assert f.crash_first_user_frame_file(LIBRARY_CRASH) == "json.h"


def test_harness_origin_true_for_fuzz_file_crash():
    assert f.crash_is_harness_origin(HARNESS_CRASH) is True


def test_library_origin_not_flagged_even_with_fuzz_in_path():
    # The crash executes in json.h (library) — must be preserved as a real bug,
    # even though the include path contains '/fuzz/../json.h'.
    assert f.crash_is_harness_origin(LIBRARY_CRASH) is False


def test_harness_leak_attributed_to_harness():
    assert f.crash_is_harness_origin(HARNESS_LEAK) is True


def test_clean_log_not_harness_origin():
    assert f.crash_is_harness_origin(CLEAN) is False
    assert f.crash_first_user_frame_file(CLEAN) == ""


def test_known_harness_basename_match():
    log = (
        "==1==ERROR: AddressSanitizer: SEGV\n"
        "    #0 0x1 in handle /src/fuzz/weird_name.c:5\n"
    )
    # not a *_fuzz.* file, but explicitly known as the harness source
    assert f.crash_is_harness_origin(log, ("weird_name.c",)) is True
    assert f.crash_is_harness_origin(log) is False


def test_gate_toggle_default_on():
    assert f._harness_validity_gate_enabled() is True
