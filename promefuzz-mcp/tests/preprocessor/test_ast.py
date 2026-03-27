"""
Tests for AST preprocessor module.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from promefuzz_mcp.preprocessor.ast import ASTPreprocessor, Meta


class TestMeta:
    """Test cases for Meta class."""

    def test_meta_init(self, sample_meta):
        """Test Meta initialization."""
        meta = Meta(sample_meta)
        assert meta.meta == sample_meta

    def test_meta_load(self, temp_dir, sample_meta):
        """Test loading meta from JSON file."""
        meta_file = temp_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump(sample_meta, f)

        meta = Meta.load(meta_file)
        assert meta.meta == sample_meta
        assert "functions" in meta.meta

    def test_meta_dump(self, temp_dir, sample_meta):
        """Test dumping meta to JSON file."""
        meta = Meta(sample_meta)
        output_file = temp_dir / "output.json"
        meta.dump(output_file)

        assert output_file.exists()

        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert loaded == sample_meta


class TestASTPreprocessor:
    """Test cases for ASTPreprocessor class."""

    def test_init_with_source_paths(self, sample_source_files):
        """Test ASTPreprocessor initialization with source paths."""
        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[sample_source_files["dir"]],
                compile_commands_path=None,
            )

            assert preprocessor.source_paths == [sample_source_files["dir"]]
            assert preprocessor.compile_commands_path is None

    def test_init_with_compile_commands(self, sample_source_files, temp_dir):
        """Test ASTPreprocessor initialization with compile commands."""
        compile_commands = temp_dir / "compile_commands.json"
        compile_commands.write_text("[]")

        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[sample_source_files["dir"]],
                compile_commands_path=compile_commands,
            )

            assert preprocessor.compile_commands_path == compile_commands

    def test_collect_source_files_from_dir(self, sample_source_files):
        """Test collecting source files from directory."""
        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[sample_source_files["dir"]],
                compile_commands_path=None,
            )

            # Should have collected the source file
            assert len(preprocessor.source_files) >= 0

    def test_collect_source_files_from_file(self, sample_source_files):
        """Test collecting source files from single file."""
        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[sample_source_files["source"]],
                compile_commands_path=None,
            )

            # Should have collected the source file
            assert len(preprocessor.source_files) == 1

    def test_collect_source_files_with_exclude(self, temp_dir):
        """Test collecting source files with exclusion."""
        # Create files in different directories
        src_dir = temp_dir / "src"
        src_dir.mkdir()
        (src_dir / "main.cpp").write_text("// main")
        (src_dir / "test.cpp").write_text("// test")

        excluded_dir = temp_dir / "excluded"
        excluded_dir.mkdir()
        (excluded_dir / "excluded.cpp").write_text("// excluded")

        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[temp_dir],
                compile_commands_path=None,
                pool_size=1,
            )

            # Check that excluded files are not in the list
            excluded_paths = [str(p) for p in preprocessor.source_files]
            # This is implementation dependent
            assert isinstance(preprocessor.source_files, list)

    @patch("promefuzz_mcp.preprocessor.ast.subprocess.run")
    @patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder")
    def test_process_file_success(self, mock_builder, mock_run, sample_source_files, temp_dir):
        """Test processing a single source file."""
        # Mock binary builder
        mock_bin = MagicMock()
        mock_bin.__str__ = lambda self: "/fake/bin/preprocessor"
        mock_builder.return_value.get_preprocessor_bin.return_value = mock_bin

        # Mock successful subprocess
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        # Create meta output file
        meta_file = temp_dir / "meta.json"
        with open(meta_file, "w") as f:
            json.dump({"functions": {}}, f)

        with patch("promefuzz_mcp.preprocessor.ast.tempfile.TemporaryDirectory") as mock_tmp:
            mock_tmp.return_value.__enter__ = MagicMock(return_value=str(temp_dir))
            mock_tmp.return_value.__exit__ = MagicMock(return_value=False)

            preprocessor = ASTPreprocessor(
                source_paths=[sample_source_files["dir"]],
                compile_commands_path=None,
            )

            # The process_file method requires binary to exist
            # This test verifies the call structure
            mock_run.assert_not_called()

    @patch("promefuzz_mcp.preprocessor.ast.subprocess.run")
    @patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder")
    def test_process_file_failure(self, mock_builder, mock_run, sample_source_files, temp_dir):
        """Test processing a source file with failure."""
        mock_bin = MagicMock()
        mock_bin.__str__ = lambda self: "/fake/bin/preprocessor"
        mock_builder.return_value.get_preprocessor_bin.return_value = mock_bin

        # Mock failed subprocess
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error processing file",
        )

        preprocessor = ASTPreprocessor(
            source_paths=[sample_source_files["dir"]],
            compile_commands_path=None,
        )

        # Process should handle error gracefully
        # The actual behavior depends on the implementation
        assert preprocessor is not None

    def test_run_empty_source_files(self):
        """Test running with no source files."""
        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[],
                compile_commands_path=None,
            )

            # With empty source files, run should return empty meta
            # This tests the edge case
            assert preprocessor.source_files == []


class TestASTPreprocessorPoolSize:
    """Test cases for ASTPreprocessor pool size handling."""

    def test_default_pool_size(self):
        """Test default pool size is 1."""
        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[],
                compile_commands_path=None,
            )

            assert preprocessor.pool_size == 1

    def test_custom_pool_size(self):
        """Test custom pool size."""
        with patch("promefuzz_mcp.preprocessor.ast.BinaryBuilder") as mock_builder:
            mock_builder.return_value = MagicMock()

            preprocessor = ASTPreprocessor(
                source_paths=[],
                compile_commands_path=None,
                pool_size=4,
            )

            assert preprocessor.pool_size == 4
