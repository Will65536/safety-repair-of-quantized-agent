from pathlib import Path

import pytest

from safe_quant.config import load_benchmark_config, load_model_config


def test_model_config_expands_environment(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "models"
    monkeypatch.setenv("SAFE_QUANT_MODEL_ROOT", str(model_root))
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """\
id: example
family: llama3
loader: transformers
model_path: ${SAFE_QUANT_MODEL_ROOT}/base
tokenizer_path: ${SAFE_QUANT_MODEL_ROOT}/base
precision: {weights: bf16, activations: bf16, kv_cache: bf16}
""",
        encoding="utf-8",
    )

    config = load_model_config(config_path)

    assert config["model_path"] == str(model_root / "base")


def test_model_config_rejects_unresolved_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_MODEL_ROOT", raising=False)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """\
id: example
family: llama3
loader: transformers
model_path: ${MISSING_MODEL_ROOT}/base
tokenizer_path: base
precision: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unresolved environment variable"):
        load_model_config(config_path)


def test_flatquant_model_loader_is_supported(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """\
id: flatquant-example
family: llama3.1
loader: flatquant_prequantized
model_path: /models/flatquant
tokenizer_path: /models/flatquant
precision: {weights: 4, activations: 4, kv_cache: 4}
""",
        encoding="utf-8",
    )

    assert load_model_config(config_path)["loader"] == "flatquant_prequantized"


def test_wildguard_likelihood_benchmark_config_is_supported(tmp_path: Path) -> None:
    config_path = tmp_path / "wildguard.yaml"
    config_path.write_text(
        """\
id: wildguardtest_prompt
loader: wildguard_llama_likelihood
dataset: {path: example}
protocol: {version: reference}
""",
        encoding="utf-8",
    )

    assert load_benchmark_config(config_path)["loader"] == "wildguard_llama_likelihood"


@pytest.mark.parametrize("loader", ["lm_eval_harness", "wikitext2_perplexity"])
def test_general_benchmark_loaders_are_supported(
    tmp_path: Path, loader: str
) -> None:
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""\
id: example
loader: {loader}
dataset: {{path: example}}
protocol: {{version: frozen}}
""",
        encoding="utf-8",
    )

    assert load_benchmark_config(config_path)["loader"] == loader
