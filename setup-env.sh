#!/bin/bash
# setup-env.sh for sherpa project
# Usage: source ./setup-env.sh

VENV_DIR=".venv"
REQ_FILE="harness_generator/requirements.txt"
PYTHON_BIN="python3"

# Domestic mirrors (override via env if needed)
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmmirror.com}"
APT_MIRROR="${APT_MIRROR:-http://mirrors.tuna.tsinghua.edu.cn}"
USE_APT_MIRROR="${SHERPA_APT_USE_MIRROR:-1}"

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
        npm config set registry "$NPM_REGISTRY"
        npm i -g opencode-ai
    else
        echo "npm not found. Please install Node.js (which includes npm) and then run: npm i -g opencode-ai"
        exit 1
    fi
fi

# Use domestic apt mirrors on Debian/Ubuntu (best-effort)
if command -v apt-get &> /dev/null && [[ "$USE_APT_MIRROR" != "0" ]]; then
    echo "Switching apt sources to domestic mirror: $APT_MIRROR"
    sudo sed -i -E "s#https?://(archive|security).ubuntu.com/ubuntu#${APT_MIRROR}/ubuntu#g" /etc/apt/sources.list || true
    sudo sed -i -E "s#https?://deb.debian.org/debian#${APT_MIRROR}/debian#g" /etc/apt/sources.list || true
    sudo sed -i -E "s#https?://security.debian.org/debian-security#${APT_MIRROR}/debian-security#g" /etc/apt/sources.list || true
    sudo apt-get update -y || true
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
