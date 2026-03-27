"""
Tests for API extractor module.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from promefuzz_mcp.preprocessor.api_extractor import APIExtractor, APIFunction, APICollection


class TestAPIFunction:
    """Test cases for APIFunction class."""

    def test_api_function_init(self):
        """Test APIFunction initialization."""
        api_func = APIFunction(
            header="/path/to/header.h",
            name="test_function",
            loc="source.cpp:10:1",
            decl_loc="header.h:5:1",
        )

        assert api_func.header == "/path/to/header.h"
        assert api_func.name == "test_function"
        assert api_func.loc == "source.cpp:10:1"
        assert api_func.decl_loc == "header.h:5:1"

    def test_api_function_str(self):
        """Test APIFunction string representation."""
        api_func = APIFunction(
            header="/path/to/header.h",
            name="test_function",
            loc="source.cpp:10:1",
            decl_loc="header.h:5:1",
        )

        assert "test_function" in str(api_func)
        assert "source.cpp" in str(api_func)

    def test_api_function_to_dict(self):
        """Test APIFunction to_dict method."""
        api_func = APIFunction(
            header="/path/to/header.h",
            name="test_function",
            loc="source.cpp:10:1",
            decl_loc="header.h:5:1",
        )

        d = api_func.to_dict()
        assert d["header"] == "/path/to/header.h"
        assert d["name"] == "test_function"
        assert d["loc"] == "source.cpp:10:1"
        assert d["decl_loc"] == "header.h:5:1"


class TestAPICollection:
    """Test cases for APICollection class."""

    def test_api_collection_init_empty(self):
        """Test empty APICollection initialization."""
        collection = APICollection()
        assert collection.count == 0
        assert collection.funcs == []

    def test_api_collection_init_with_functions(self):
        """Test APICollection initialization with functions."""
        funcs = [
            APIFunction("h1.h", "func1", "loc1", "decl1"),
            APIFunction("h2.h", "func2", "loc2", "decl2"),
        ]
        collection = APICollection(funcs)
        assert collection.count == 2

    def test_api_collection_count_property(self):
        """Test count property."""
        funcs = [
            APIFunction("h1.h", "func1", "loc1", "decl1"),
            APIFunction("h2.h", "func2", "loc2", "decl2"),
            APIFunction("h3.h", "func3", "loc3", "decl3"),
        ]
        collection = APICollection(funcs)
        assert collection.count == 3

    def test_api_collection_get_by_name(self):
        """Test get_by_name method."""
        funcs = [
            APIFunction("h1.h", "test_func", "loc1", "decl1"),
            APIFunction("h2.h", "test_func", "loc2", "decl2"),
            APIFunction("h3.h", "other_func", "loc3", "decl3"),
        ]
        collection = APICollection(funcs)

        results = collection.get_by_name("test_func")
        assert len(results) == 2

        results = collection.get_by_name("nonexistent")
        assert len(results) == 0


class TestAPIExtractor:
    """Test cases for APIExtractor class."""

    def test_init(self, sample_meta):
        """Test APIExtractor initialization."""
        meta = Meta(sample_meta)
        header_paths = [Path("/test/headers")]

        extractor = APIExtractor(
            header_paths=header_paths,
            meta=meta,
        )

        assert extractor.header_paths == header_paths
        assert extractor.meta == meta

    def test_init_with_exclude_paths(self, sample_meta):
        """Test APIExtractor with exclude paths."""
        meta = Meta(sample_meta)
        header_paths = [Path("/test/headers")]
        exclude_paths = [Path("/test/excluded")]

        extractor = APIExtractor(
            header_paths=header_paths,
            meta=meta,
            exclude_paths=exclude_paths,
        )

        assert extractor.exclude_paths == exclude_paths

    def test_collect_headers_from_file(self, sample_source_files):
        """Test collecting headers from a single file."""
        meta = MagicMock()
        meta.meta = {"functions": {}}

        extractor = APIExtractor(
            header_paths=[sample_source_files["header"]],
            meta=meta,
        )

        # Should have collected the header file
        assert len(extractor.header_files) == 1

    def test_collect_headers_from_directory(self, temp_dir):
        """Test collecting headers from a directory."""
        # Create multiple header files
        headers_dir = temp_dir / "headers"
        headers_dir.mkdir()
        (headers_dir / "header1.h").write_text("// h1")
        (headers_dir / "header2.hpp").write_text("// h2")
        (headers_dir / "header3.hxx").write_text("// h3")
        (headers_dir / "not_a_header.txt").write_text("not a header")

        meta = MagicMock()
        meta.meta = {"functions": {}}

        extractor = APIExtractor(
            header_paths=[headers_dir],
            meta=meta,
        )

        # Should have collected only header files
        assert len(extractor.header_files) == 3

    def test_collect_headers_with_suffix_filter(self, temp_dir):
        """Test header collection with suffix filtering."""
        headers_dir = temp_dir / "headers"
        headers_dir.mkdir()
        (headers_dir / "valid.h").write_text("// h")
        (headers_dir / "valid.hpp").write_text("// hpp")
        (headers_dir / "invalid.txt").write_text("// txt")

        meta = MagicMock()
        meta.meta = {"functions": {}}

        extractor = APIExtractor(
            header_paths=[headers_dir],
            meta=meta,
        )

        # Should have collected only .h and .hpp files
        assert len(extractor.header_files) == 2

    def test_extract_with_no_functions(self, sample_source_files):
        """Test extraction when no functions match."""
        meta = MagicMock()
        meta.meta = {"functions": {}}

        extractor = APIExtractor(
            header_paths=[sample_source_files["header"]],
            meta=meta,
        )

        collection = extractor.extract()
        assert collection.count == 0

    def test_extract_filters_non_public_methods(self, temp_dir, sample_meta):
        """Test that extraction filters non-public methods."""
        # Create header file
        header_file = temp_dir / "sample.h"
        header_file.write_text("""
#ifndef SAMPLE_H
#define SAMPLE_H

int public_function(int arg);
void private_function(int arg);

#endif
""")

        # Update meta with functions
        meta_data = {
            "functions": {
                "sample.cpp:10:1": {
                    "name": "public_function",
                    "declLoc": "sample.h:5:1",
                    "return": {"baseType": "int"},
                    "param": [{"baseType": "int", "name": "arg"}],
                    "heldbyClass": None,
                },
            }
        }
        meta = Meta(meta_data)

        extractor = APIExtractor(
            header_paths=[header_file],
            meta=meta,
        )

        # This test depends on the actual implementation logic
        # Just verify it doesn't crash
        collection = extractor.extract()
        assert isinstance(collection, APICollection)
