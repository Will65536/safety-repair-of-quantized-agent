"""Explicit loaders for supported model artifact formats."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def load_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    loader = config["loader"]
    if loader == "transformers":
        from safe_quant.model_loaders.transformers import load_transformers_model

        return load_transformers_model(config)
    if loader == "spinquant_saved_state":
        from safe_quant.model_loaders.spinquant import load_spinquant_model

        return load_spinquant_model(config)
    if loader == "flatquant_prequantized":
        from safe_quant.model_loaders.flatquant import load_flatquant_model

        return load_flatquant_model(config)
    if loader == "quarot_saved_state":
        from safe_quant.model_loaders.quarot import load_quarot_model

        return load_quarot_model(config)
    raise ValueError(f"Unsupported model loader: {loader!r}")
