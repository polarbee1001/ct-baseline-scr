"""Shared helpers for the CLI scripts: path setup and config loading."""

from __future__ import annotations

import os
import sys

import yaml

# Repository root is the parent of the scripts/ directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


def load_config(path: str | None = None) -> dict:
    """Load config.yaml (env var KIDNEY_CONFIG, else the repo-root config)."""
    path = path or os.environ.get("KIDNEY_CONFIG") or os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> str:
    """Resolve a config-relative path against the repository root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)
