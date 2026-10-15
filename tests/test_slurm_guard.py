import pytest

from safe_quant.evaluate import _require_slurm_gpu


def test_model_evaluation_requires_slurm(monkeypatch) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    with pytest.raises(RuntimeError, match="must run inside a Slurm job"):
        _require_slurm_gpu()
