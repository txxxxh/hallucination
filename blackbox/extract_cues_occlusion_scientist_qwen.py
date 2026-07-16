#!/usr/bin/env python3
"""Qwen entry point for the ScientistQA cue-occlusion experiment.

The intervention and evaluation implementation is shared with
``extract_cues_occlusion_scientist.py``.  This entry point supplies Qwen/DashScope
defaults while allowing every value to be overridden on the command line.

Example:

    export DASHSCOPE_API_KEY=...
    python extract_cues_occlusion_scientist_qwen.py --limit 100

For another OpenAI-compatible Qwen provider:

    python extract_cues_occlusion_scientist_qwen.py \
        --target-model qwen3.5-flash \
        --base-url https://provider.example/v1
"""
from __future__ import annotations

import os
import sys
from typing import List

import extract_cues_occlusion_scientist as scientist


DEFAULT_QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-flash")
DEFAULT_QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
)
DEFAULT_QWEN_OUTDIR = "outputs_scientist/only_deletion_uncertain_qwen3.5_flash"


def _with_thinking_disabled(kwargs: dict) -> dict:
    configured = dict(kwargs)
    extra_body = dict(configured.get("extra_body") or {})
    extra_body["enable_thinking"] = False
    configured["extra_body"] = extra_body
    return configured


def _disable_qwen_thinking() -> None:
    """Inject enable_thinking=false into target and negator API calls."""
    original_completion_create = scientist._completion_create
    original_pilot_chat_create = scientist.pilot._chat_create

    def qwen_completion_create(client, **kwargs):
        return original_completion_create(
            client, **_with_thinking_disabled(kwargs)
        )

    def qwen_pilot_chat_create(client, **kwargs):
        return original_pilot_chat_create(
            client, **_with_thinking_disabled(kwargs)
        )

    scientist._completion_create = qwen_completion_create
    scientist.pilot._chat_create = qwen_pilot_chat_create


def _has_option(argv: List[str], option: str) -> bool:
    return option in argv or any(value.startswith(option + "=") for value in argv)


def _configure_environment() -> None:
    """Map common Qwen credential variables to the shared client variable."""

    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set QWEN_API_KEY or DASHSCOPE_API_KEY (OPENAI_API_KEY also works)."
        )
    os.environ["OPENAI_API_KEY"] = api_key


def _inject_qwen_defaults(argv: List[str]) -> List[str]:
    configured = list(argv)
    defaults = (
        ("--target-model", DEFAULT_QWEN_MODEL),
        ("--negator-model", DEFAULT_QWEN_MODEL),
        ("--base-url", DEFAULT_QWEN_BASE_URL),
        ("--max-sampling-batch", "4"),
        ("--outdir", DEFAULT_QWEN_OUTDIR),
    )
    for option, value in defaults:
        if not _has_option(configured, option):
            configured.extend([option, value])
    if not _has_option(configured, "--negation-allow-uncertain"):
        configured.append("--negation-allow-uncertain")
    return configured


def main() -> int:
    sys.argv = _inject_qwen_defaults(sys.argv)
    if not any(value in {"-h", "--help"} for value in sys.argv[1:]):
        _configure_environment()
        _disable_qwen_thinking()
    return scientist.main()


if __name__ == "__main__":
    raise SystemExit(main())
