"""
Pytest configuration and fixtures for PromeFuzz MCP tests.
"""

import sys
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "preprocessor": {
            "run_dummydriver_test": False,
            "dump_relevance_as_csv": False,
            "dump_call_graph": False,
        },
        "comprehender": {
            "embedding_llm": "test_embedding",
            "comprehension_llm": "",
            "retrieve_top_k": 3,
            "function_batch_size": 24,
        },
        "llm": {
            "default_llm": "test_llm",
            "validate_llm": False,
            "enable_log": True,
            "test_llm": {
                "llm_type": "openai",
                "base_url": "https://api.test.com/v1",
                "api_key": "test_key",
                "model": "test-model",
                "temperature": 0.5,
                "max_tokens": 1000,
                "timeout": 60,
                "retry_times": 3,
            },
            "embedding_llm": {
                "llm_type": "ollama",
                "host": "localhost",
                "port": 11434,
                "model": "test-embed",
                "max_tokens": -1,
                "timeout": 30,
                "retry_times": 3,
            },
        },
        "bin": {
            "preprocessor": "processor/build/bin/preprocessor",
            "cgprocessor": "processor/build/bin/cgprocessor",
        },
    }


@pytest.fixture
def sample_meta():
    """Sample metadata for testing."""
    return {
        "classes": {
            "sample.cpp:10:1": {
                "name": "TestClass",
                "isAbstract": False,
            }
        },
        "functions": {
            "sample.cpp:20:1": {
                "name": "test_function",
                "declLoc": "sample.h:5:1",
                "return": {"baseType": "int"},
                "param": [
                    {"baseType": "int", "name": "arg1"},
                    {"baseType": "char*", "name": "arg2"},
                ],
                "heldbyClass": None,
            },
            "sample.cpp:30:1": {
                "name": "TestClass::method",
                "declLoc": "sample.h:15:1",
                "return": {"baseType": "void"},
                "param": [],
                "heldbyClass": "TestClass",
            },
        },
        "composites": {},
        "enums": {},
        "typedefs": {},
    }


@pytest.fixture
def sample_api_functions():
    """Sample API functions for testing."""
    return [
        {
            "header": "/path/to/sample.h",
            "name": "test_function",
            "loc": "sample.cpp:20:1",
            "decl_loc": "sample.h:5:1",
        },
        {
            "header": "/path/to/sample.h",
            "name": "another_function",
            "loc": "sample.cpp:25:1",
            "decl_loc": "sample.h:10:1",
        },
    ]


@pytest.fixture
def sample_callgraph_edges():
    """Sample call graph edges for testing."""
    return [
        ("caller1", "loc1", "callee1", "loc2"),
        ("caller2", "loc3", "callee2", "loc4"),
    ]


@pytest.fixture
def sample_knowledge_base_info():
    """Sample knowledge base info for testing."""
    return {
        "id": "test_kb",
        "document_count": 5,
        "chunk_count": 100,
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the LLM."
                }
            }
        ]
    }


@pytest.fixture
def project_root_path():
    """Get project root path."""
    return Path(__file__).parent.parent


@pytest.fixture
def processor_dir(project_root_path):
    """Get processor directory path."""
    return project_root_path / "processor"


@pytest.fixture
def sample_source_files(temp_dir):
    """Create sample source files for testing."""
    # Create sample C++ source file
    source_file = temp_dir / "sample.cpp"
    source_file.write_text("""
#include "sample.h"

int test_function(int arg1, char* arg2) {
    return arg1;
}

void another_function() {
    // implementation
}
""")

    # Create sample header file
    header_file = temp_dir / "sample.h"
    header_file.write_text("""
#ifndef SAMPLE_H
#define SAMPLE_H

int test_function(int arg1, char* arg2);
void another_function();

#endif
""")

    return {
        "source": source_file,
        "header": header_file,
        "dir": temp_dir,
    }


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Reset config singleton after each test."""
    from promefuzz_mcp.config import Config
    yield
    Config._instance = None
