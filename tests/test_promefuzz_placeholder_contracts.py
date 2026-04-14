from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROMEFUZZ_DIR = ROOT / "promefuzz-mcp"
if str(PROMEFUZZ_DIR) not in sys.path:
    sys.path.insert(0, str(PROMEFUZZ_DIR))

from promefuzz_mcp.comprehender.func_relevance import FunctionRelevanceComprehender
from promefuzz_mcp.comprehender.func_usage import FunctionUsageComprehender
from promefuzz_mcp.comprehender.purpose import PurposeComprehender
from promefuzz_mcp.preprocessor.complexity import ComplexityCalculator
from promefuzz_mcp.preprocessor.incidental import IncidentalExtractor
from promefuzz_mcp.preprocessor.relevance import CallRelevance, ClassRelevance, TypeRelevance


def test_comprehender_placeholders_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        PurposeComprehender({}).comprehend()
    with pytest.raises(NotImplementedError):
        FunctionUsageComprehender({}).comprehend("foo")
    with pytest.raises(NotImplementedError):
        FunctionUsageComprehender({}).comprehend_all(["foo"])
    with pytest.raises(NotImplementedError):
        FunctionRelevanceComprehender({}).comprehend("purpose", {})


def test_preprocessor_placeholders_raise_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        ComplexityCalculator({}).calculate()
    with pytest.raises(NotImplementedError):
        IncidentalExtractor({}).extract()
    with pytest.raises(NotImplementedError):
        TypeRelevance({}).calculate()
    with pytest.raises(NotImplementedError):
        ClassRelevance({}).calculate()
    with pytest.raises(NotImplementedError):
        CallRelevance({}).calculate()
