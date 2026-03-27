"""
PromeFuzz MCP Server.

This module provides MCP tools for code analysis and comprehension.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
import click

# Initialize logging
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)


def create_mcp_app():
    """Create FastMCP application."""
    try:
        from fastmcp import FastMCP

        mcp = FastMCP("PromeFuzz")

        # Import tools after FastMCP is available
        from .server_tools import register_tools

        register_tools(mcp)

        return mcp

    except ImportError as e:
        logger.error(f"Failed to import FastMCP: {e}")
        logger.info("Installing dependencies: pip install fastmcp")
        raise


@click.group()
def cli():
    """PromeFuzz MCP Tools CLI."""
    pass


@cli.command()
@click.option("--config", type=click.Path(), help="Config file path")
@click.option("--host", default="localhost", help="Server host")
@click.option("--port", default=8000, help="Server port")
@click.option("--skip-build", is_flag=True, help="Skip binary build check")
def start(config: Optional[str], host: str, port: int, skip_build: bool):
    """Start the MCP server."""
    # Set config path if provided
    if config:
        from . import config as config_module
        config_module.get_config(Path(config))

    # Check and build binaries
    if not skip_build:
        from .build import check_binaries, build_binaries

        if not check_binaries():
            logger.info("Building processor binaries...")
            if not build_binaries():
                logger.error("Failed to build processor binaries")
                sys.exit(1)
    else:
        logger.warning("Skipping binary build check")

    # Create and run MCP server
    mcp = create_mcp_app()

    logger.info(f"Starting MCP server on {host}:{port}")
    mcp.run(transport="stdio")


@cli.command()
@click.option("--force", is_flag=True, help="Force rebuild")
def build(force: bool):
    """Build processor binaries."""
    from .build import build_binaries

    if build_binaries(force=force):
        logger.success("Build completed successfully")
    else:
        logger.error("Build failed")
        sys.exit(1)


@cli.command()
def check():
    """Check if processor binaries exist."""
    from .build import check_binaries

    if check_binaries():
        logger.success("Processor binaries found")
    else:
        logger.warning("Processor binaries not found")
        logger.info("Run 'promefuzz-mcp build' to build them")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
