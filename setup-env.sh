#!/bin/bash
# setup-env.sh for sherpa project
# Usage: source ./setup-env.sh

VENV_DIR=".venv"
REQ_FILE="harness_generator/requirements.txt"
PYTHON_BIN="python3"

# Detect Apple Silicon and recommend Homebrew Python if needed
if [[ $(uname -m) == "arm64" ]]; then
    echo "Detected Apple Silicon (arm64)."
    if ! command -v $PYTHON_BIN &> /dev/null; then
        echo "$PYTHON_BIN not found. Please install Python 3 via Homebrew: brew install python3"
        exit 1
    fi
fi

# Install OpenCode CLI if missing
if ! command -v opencode &> /dev/null; then
    echo "opencode not found. Installing via npm..."
    if command -v npm &> /dev/null; then
        npm i -g opencode-ai
    else
        echo "npm not found. Please install Node.js (which includes npm) and then run: npm i -g opencode-ai"
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_BIN -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r $REQ_FILE

echo "Environment setup complete."
