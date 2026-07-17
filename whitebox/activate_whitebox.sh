#!/usr/bin/env bash

_WHITEBOX_VENV="${WHITEBOX_VENV:-$HOME/venvs/whitebox}"
if [[ ! -f "$_WHITEBOX_VENV/bin/activate" ]]; then
    echo "whitebox virtual environment not found: $_WHITEBOX_VENV" >&2
    echo "Run: bash setup_whitebox_env.sh" >&2
    return 1 2>/dev/null || exit 1
fi

source "$_WHITEBOX_VENV/bin/activate"
export CPATH="$HOME/.local/python310-dev/usr/include/python3.10:$HOME/.local/python310-dev/usr/include${CPATH:+:$CPATH}"
unset _WHITEBOX_VENV
