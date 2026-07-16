#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${WHITEBOX_VENV:-$HOME/venvs/whitebox}"
HEADER_ROOT="$HOME/.local/python310-dev"
HEADER_DEB="/tmp/libpython3.10-dev.deb"
HEADER_URL="https://security.ubuntu.com/ubuntu/pool/main/p/python3.10/libpython3.10-dev_3.10.12-1~22.04.16_amd64.deb"

if ! command -v python3.10 >/dev/null 2>&1; then
    echo "python3.10 is required but was not found." >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    python3.10 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements-lock.txt"

if [[ ! -f "$HEADER_ROOT/usr/include/python3.10/Python.h" ]]; then
    if ! command -v curl >/dev/null 2>&1 || ! command -v dpkg-deb >/dev/null 2>&1; then
        echo "curl and dpkg-deb are required for no-sudo Python headers." >&2
        exit 1
    fi
    mkdir -p "$HEADER_ROOT"
    curl -fLo "$HEADER_DEB" "$HEADER_URL"
    dpkg-deb -x "$HEADER_DEB" "$HEADER_ROOT"
fi

echo
echo "Environment ready."
echo "Activate with: source $PROJECT_DIR/activate_whitebox.sh"
