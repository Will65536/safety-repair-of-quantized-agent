import json
from pathlib import Path

import pytest

from safe_quant.model_loaders.flatquant import _validate_artifact


def _config(tmp_path: Path) -> dict:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "flatquant",
                    "w_bits": 4,
                    "a_bits": 4,
                    "kv_bits": 4,
                    "w_is_symmetric": True,
                    "a_is_symmetric": True,
                    "kv_is_symmetric": False,
                }
            }
        ),
        encoding="utf-8",
    )
    return {
        "artifact_format": "flatquant_real_safetensors",
        "model_path": str(artifact),
        "local_files_only": True,
        "trust_remote_code": True,
        "precision": {"weights": 4, "activations": 4, "kv_cache": 4},
    }


def test_flatquant_accepts_prequantized_w4a4kv4(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert _validate_artifact(config) == Path(config["model_path"])


def test_flatquant_rejects_wrong_artifact_precision(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["precision"]["kv_cache"] = 16

    with pytest.raises(ValueError, match="W4A4KV4"):
        _validate_artifact(config)


def test_flatquant_rejects_nonlocal_loading(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["local_files_only"] = False

    with pytest.raises(ValueError, match="local immutable checkpoint"):
        _validate_artifact(config)
