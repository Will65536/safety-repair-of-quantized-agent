"""Configuration loading and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_UNRESOLVED_ENV = re.compile(r"\$\{[^}]+\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if _UNRESOLVED_ENV.search(expanded):
            raise ValueError(f"Unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML mapping and expand ${VARIABLE} references."""
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return _expand_environment(value)


def load_model_config(path: Path) -> Dict[str, Any]:
    config = load_yaml(path)
    required = {"id", "family", "loader", "model_path", "tokenizer_path", "precision"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Model config {path} is missing: {sorted(missing)}")
    if config["loader"] not in {
        "transformers",
        "spinquant_saved_state",
        "quarot_saved_state",
        "flatquant_prequantized",
    }:
        raise ValueError(f"Unsupported model loader: {config['loader']!r}")
    return config


def load_benchmark_config(path: Path) -> Dict[str, Any]:
    config = load_yaml(path)
    required = {"id", "loader", "dataset", "protocol"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Benchmark config {path} is missing: {sorted(missing)}")
    if config["loader"] not in {
        "finguard",
        "wildguard_llama_likelihood",
        "lm_eval_harness",
        "wikitext2_perplexity",
        "behavioral_generation",
    }:
        raise ValueError(f"Unsupported benchmark loader: {config['loader']!r}")
    return config
