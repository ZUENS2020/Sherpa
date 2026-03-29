"""
Binary tools build module for PromeFuzz MCP.
"""

import os
import subprocess
import shutil
from pathlib import Path
from loguru import logger
from typing import Optional

from .config import get_config


class BinaryBuilder:
    """Binary tools builder for PromeFuzz MCP."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = get_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.processor_dir = self.project_root / "processor"
        self.build_dir = self.processor_dir / "build"

    def check_binaries(self) -> bool:
        """Check if binaries exist."""
        preprocessor = self.config.get_preprocessor_bin_path()
        cgprocessor = self.config.get_cgprocessor_bin_path()

        preprocessor_exists = preprocessor.exists() if preprocessor else False
        cgprocessor_exists = cgprocessor.exists() if cgprocessor else False

        if preprocessor_exists and cgprocessor_exists:
            logger.info(f"Preprocessor binary: {preprocessor}")
            logger.info(f"CGProcessor binary: {cgprocessor}")
            return True

        if not preprocessor_exists:
            logger.warning(f"Preprocessor binary not found: {preprocessor}")
        if not cgprocessor_exists:
            logger.warning(f"CGProcessor binary not found: {cgprocessor}")

        return False

    def build(self, force: bool = False) -> bool:
        """
        Build processor binaries.

        Args:
            force: Force rebuild even if binaries exist

        Returns:
            True if build successful, False otherwise
        """
        if self.check_binaries() and not force:
            logger.info("Binaries already exist, skipping build")
            return True

        logger.info("Building processor binaries...")

        # Check build tools
        if not self._check_build_tools():
            logger.error("Build tools not found. Please install CMake, make, and Clang.")
            return False

        # Create build directory
        self.build_dir.mkdir(parents=True, exist_ok=True)

        # Build CXX processor
        if not self._build_cxx_processor():
            logger.error("Failed to build CXX processor")
            return False

        logger.success("Processor binaries built successfully")
        return True

    def _check_build_tools(self) -> bool:
        """Check if required build tools are available."""
        required = ["cmake", "make", "clang", "clang++"]

        for tool in required:
            if shutil.which(tool) is None:
                logger.error(f"Required build tool not found: {tool}")
                return False

        logger.info("All required build tools found")
        return True

    def _build_cxx_processor(self) -> bool:
        """Build C++ processor tools."""
        cxx_dir = self.processor_dir / "cxx"
        if not cxx_dir.exists():
            logger.error(f"CXX source directory not found: {cxx_dir}")
            return False

        import shutil
        clang_install_dir = str(os.environ.get("CLANG_INSTALL_DIR") or "").strip()
        llvm_config_path = shutil.which("llvm-config")
        if llvm_config_path:
            result = subprocess.run(
                [llvm_config_path, "--prefix"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                clang_install_dir = result.stdout.strip()
        if not clang_install_dir:
            for candidate in (
                "/usr/lib/llvm-18",
                "/usr/lib/llvm-17",
                "/usr/lib/llvm-16",
                "/usr/lib/llvm-15",
                "/usr/lib/llvm-14",
                "/usr/lib/llvm",
            ):
                if Path(candidate).is_dir():
                    clang_install_dir = candidate
                    break

        cmake_args = []
        if clang_install_dir:
            cmake_args.extend(
                [
                    f"-DCMAKE_PREFIX_PATH={clang_install_dir}",
                    f"-DLLVM_DIR={clang_install_dir}/lib/cmake/llvm",
                ]
            )

        cmake_cmd = [
            "cmake",
            "-S", str(cxx_dir),
            "-B", str(self.build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ] + cmake_args

        logger.info(f"Running: {' '.join(cmake_cmd)}")

        try:
            result = subprocess.run(
                cmake_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                logger.error(f"CMake failed: {result.stderr}")
                return False

            logger.info("CMake configuration successful")

        except Exception as e:
            logger.error(f"Failed to run cmake: {e}")
            return False

        # Run make
        make_cmd = ["cmake", "--build", str(self.build_dir), "-j"]

        logger.info(f"Running: {' '.join(make_cmd)}")

        try:
            result = subprocess.run(
                make_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode != 0:
                logger.error(f"Make failed: {result.stderr}")
                return False

            logger.info("Build successful")

        except Exception as e:
            logger.error(f"Failed to run make: {e}")
            return False

        return True

    def get_preprocessor_bin(self) -> Path:
        """Get preprocessor binary path."""
        path = self.config.get_preprocessor_bin_path()
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor binary not found: {path}")
        return path

    def get_cgprocessor_bin(self) -> Path:
        """Get cgprocessor binary path."""
        path = self.config.get_cgprocessor_bin_path()
        if not path.exists():
            raise FileNotFoundError(f"CGProcessor binary not found: {path}")
        return path


def build_binaries(config_path: Optional[Path] = None, force: bool = False) -> bool:
    """
    Convenience function to build binaries.

    Args:
        config_path: Path to config file
        force: Force rebuild

    Returns:
        True if successful
    """
    builder = BinaryBuilder(config_path)
    return builder.build(force=force)


def check_binaries(config_path: Optional[Path] = None) -> bool:
    """
    Convenience function to check if binaries exist.

    Args:
        config_path: Path to config file

    Returns:
        True if binaries exist
    """
    builder = BinaryBuilder(config_path)
    return builder.check_binaries()
