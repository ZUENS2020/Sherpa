#!/bin/bash
cd /home/nightsglow/workspace/PromfuzzOOO/Tools/promefuzz-mcp
export PYTHONPATH=/home/nightsglow/workspace/PromfuzzOOO/Tools/promefuzz-mcp
exec python3 -c "from promefuzz_mcp.server import main; main()" start --skip-build
