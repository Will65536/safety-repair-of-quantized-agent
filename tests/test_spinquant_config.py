from pathlib import Path

import pytest

from types import SimpleNamespace

from safe_quant.model_loaders.spinquant import (
    _runtime_args,
    _skip_overwritten_preload_transforms,
)


def test_spinquant_runtime_is_load_only(tmp_path: Path) -> None:
    rotation = tmp_path / "R.bin"
    state = tmp_path / "qmodel.pt"
    rotation.touch()
    state.touch()
    config = {
        "artifact_format": "spinquant_qmodel_state",
        "precision": {"weights": 4, "activations": 4, "kv_cache": 4},
        "spinquant": {
            "rotation_path": str(rotation),
            "quantized_state_path": str(state),
        },
    }

    args = _runtime_args(config)

    assert args.load_qmodel_path == str(state)
    assert args.save_qmodel_path is None
    assert args.w_bits == args.a_bits == args.k_bits == args.v_bits == 4
    assert args.nsamples == 0
    assert args.w_rtn is False


def test_spinquant_rejects_wrong_precision(tmp_path: Path) -> None:
    config = {
        "artifact_format": "spinquant_qmodel_state",
        "precision": {"weights": 4, "activations": 4, "kv_cache": 16},
        "spinquant": {},
    }

    with pytest.raises(ValueError, match="W4A4KV4"):
        _runtime_args(config)


def test_saved_state_fast_load_restores_spinquant_functions() -> None:
    def fuse(model):
        return model

    def rotate(model, args):
        return model

    main = SimpleNamespace(
        fuse_norm_utils=SimpleNamespace(fuse_layer_norms=fuse),
        rotation_utils=SimpleNamespace(rotate_model=rotate),
    )
    with _skip_overwritten_preload_transforms(main):
        assert main.fuse_norm_utils.fuse_layer_norms(object()) is None
        assert main.rotation_utils.rotate_model(object(), object()) is None
    assert main.fuse_norm_utils.fuse_layer_norms is fuse
    assert main.rotation_utils.rotate_model is rotate
