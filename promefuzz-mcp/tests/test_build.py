"""
Tests for build module.
"""

import pytest
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import tempfile

from promefuzz_mcp.build import BinaryBuilder, check_binaries, build_binaries


class TestBinaryBuilder:
    """Test cases for BinaryBuilder class."""

    def test_init(self, temp_dir, mock_config):
        """Test BinaryBuilder initialization."""
        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_preprocessor_bin_path=MagicMock(return_value=temp_dir / "preprocessor"),
                get_cgprocessor_bin_path=MagicMock(return_value=temp_dir / "cgprocessor"),
            )
            builder = BinaryBuilder(config_path=temp_dir / "config.toml")
            assert builder.project_root.exists()

    def test_check_binaries_not_exist(self, temp_dir, mock_config):
        """Test check_binaries when binaries don't exist."""
        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_preprocessor_bin_path=MagicMock(return_value=temp_dir / "preprocessor"),
                get_cgprocessor_bin_path=MagicMock(return_value=temp_dir / "cgprocessor"),
            )
            builder = BinaryBuilder(config_path=temp_dir / "config.toml")
            assert builder.check_binaries() == False

    def test_check_binaries_exist(self, temp_dir, mock_config):
        """Test check_binaries when binaries exist."""
        # Create dummy binary files
        bin_dir = temp_dir / "bin"
        bin_dir.mkdir()
        (bin_dir / "preprocessor").touch()
        (bin_dir / "cgprocessor").touch()

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_preprocessor_bin_path=MagicMock(return_value=bin_dir / "preprocessor"),
                get_cgprocessor_bin_path=MagicMock(return_value=bin_dir / "cgprocessor"),
            )
            builder = BinaryBuilder(config_path=temp_dir / "config.toml")
            assert builder.check_binaries() == True

    def test_check_binaries_partial_exist(self, temp_dir, mock_config):
        """Test check_binaries when only one binary exists."""
        bin_dir = temp_dir / "bin"
        bin_dir.mkdir()
        (bin_dir / "preprocessor").touch()
        # cgprocessor doesn't exist

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_preprocessor_bin_path=MagicMock(return_value=bin_dir / "preprocessor"),
                get_cgprocessor_bin_path=MagicMock(return_value=bin_dir / "cgprocessor"),
            )
            builder = BinaryBuilder(config_path=temp_dir / "config.toml")
            assert builder.check_binaries() == False

    @patch("promefuzz_mcp.build.shutil.which")
    def test_check_build_tools_all_present(self, mock_which):
        """Test _check_build_tools when all tools are present."""
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            builder = BinaryBuilder()
            assert builder._check_build_tools() == True

    @patch("promefuzz_mcp.build.shutil.which")
    def test_check_build_tools_missing(self, mock_which):
        """Test _check_build_tools when some tools are missing."""
        def which_mock(tool):
            if tool == "cmake":
                return "/usr/bin/cmake"
            return None

        mock_which.side_effect = which_mock

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            builder = BinaryBuilder()
            assert builder._check_build_tools() == False

    @patch("promefuzz_mcp.build.subprocess.run")
    @patch("promefuzz_mcp.build.shutil.which")
    def test_build_cxx_processor_success(self, mock_which, mock_run, temp_dir):
        """Test successful C++ processor build."""
        # Mock all build tools exist
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"

        # Mock successful cmake and make
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        cxx_dir = temp_dir / "cxx"
        cxx_dir.mkdir()
        (cxx_dir / "CMakeLists.txt").write_text("project(test)")

        build_dir = temp_dir / "build"
        build_dir.mkdir()

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_preprocessor_bin_path=MagicMock(return_value=build_dir / "preprocessor"),
                get_cgprocessor_bin_path=MagicMock(return_value=build_dir / "cgprocessor"),
            )
            builder = BinaryBuilder()
            builder.processor_dir = temp_dir
            builder.build_dir = build_dir

            result = builder._build_cxx_processor()
            assert result == True
            assert mock_run.call_count >= 2  # cmake and make

    @patch("promefuzz_mcp.build.subprocess.run")
    @patch("promefuzz_mcp.build.shutil.which")
    def test_build_cxx_processor_cmake_failure(self, mock_which, mock_run, temp_dir):
        """Test C++ processor build with cmake failure."""
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"
        mock_run.return_value = MagicMock(returncode=1, stderr="CMake error")

        cxx_dir = temp_dir / "cxx"
        cxx_dir.mkdir()
        (cxx_dir / "CMakeLists.txt").write_text("project(test)")

        build_dir = temp_dir / "build"
        build_dir.mkdir()

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            builder = BinaryBuilder()
            builder.processor_dir = temp_dir
            builder.build_dir = build_dir

            result = builder._build_cxx_processor()
            assert result == False

    @patch("promefuzz_mcp.build.subprocess.run")
    @patch("promefefuzz_mcp.build.shutil.which")
    def test_build_cxx_processor_make_failure(self, mock_which, mock_run, temp_dir):
        """Test C++ processor build with make failure."""
        mock_which.side_effect = lambda x: f"/usr/bin/{x}"

        # First call (cmake) succeeds, second call (make) fails
        mock_run.side_effect = [
            MagicMock(returncode=0, stderr=""),
            MagicMock(returncode=1, stderr="Make error"),
        ]

        cxx_dir = temp_dir / "cxx"
        cxx_dir.mkdir()
        (cxx_dir / "CMakeLists.txt").write_text("project(test)")

        build_dir = temp_dir / "build"
        build_dir.mkdir()

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock()
            builder = BinaryBuilder()
            builder.processor_dir = temp_dir
            builder.build_dir = build_dir

            result = builder._build_cxx_processor()
            assert result == False

    def test_get_preprocessor_bin(self, temp_dir):
        """Test getting preprocessor binary path."""
        bin_dir = temp_dir / "bin"
        bin_dir.mkdir()
        (bin_dir / "preprocessor").touch()

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_preprocessor_bin_path=MagicMock(return_value=bin_dir / "preprocessor"),
            )
            builder = BinaryBuilder()
            path = builder.get_preprocessor_bin()
            assert path == bin_dir / "preprocessor"

    def test_get_preprocessor_bin_not_found(self, temp_dir):
        """Test getting preprocessor binary that doesn't exist."""
        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_preprocessor_bin_path=MagicMock(return_value=temp_dir / "preprocessor"),
            )
            builder = BinaryBuilder()

            with pytest.raises(FileNotFoundError):
                builder.get_preprocessor_bin()

    def test_get_cgprocessor_bin(self, temp_dir):
        """Test getting cgprocessor binary path."""
        bin_dir = temp_dir / "bin"
        bin_dir.mkdir()
        (bin_dir / "cgprocessor").touch()

        with patch("promefuzz_mcp.build.get_config") as mock_get_config:
            mock_get_config.return_value = MagicMock(
                get_cgprocessor_bin_path=MagicMock(return_value=bin_dir / "cgprocessor"),
            )
            builder = BinaryBuilder()
            path = builder.get_cgprocessor_bin()
            assert path == bin_dir / "cgprocessor"


class TestCheckBinaries:
    """Test cases for check_binaries convenience function."""

    @patch("promefuzz_mcp.build.BinaryBuilder")
    def test_check_binaries_delegates_to_builder(self, mock_builder_class):
        """Test check_binaries delegates to BinaryBuilder."""
        mock_builder = MagicMock()
        mock_builder.check_binaries.return_value = True
        mock_builder_class.return_value = mock_builder

        result = check_binaries()

        mock_builder.check_binaries.assert_called_once()
        assert result == True


class TestBuildBinaries:
    """Test cases for build_binaries convenience function."""

    @patch("promefuzz_mcp.build.BinaryBuilder")
    def test_build_binaries_delegates_to_builder(self, mock_builder_class):
        """Test build_binaries delegates to BinaryBuilder."""
        mock_builder = MagicMock()
        mock_builder.build.return_value = True
        mock_builder_class.return_value = mock_builder

        result = build_binaries(force=True)

        mock_builder.build.assert_called_once_with(force=True)
        assert result == True
