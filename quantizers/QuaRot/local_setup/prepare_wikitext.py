"""Cache official WikiText-2 inputs and record native calibration provenance.

CPU-only tokenization/data preparation, safe on a login node. No model loading.
The GPU run still uses data_utils.get_loaders itself; no replacement sampler.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = "/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/bf16"
DATA = ROOT / "local_setup/data"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HOME"] = str(ROOT / "local_setup/hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(ROOT / "local_setup/hf-cache/datasets")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "fake_quant"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-check", action="store_true")
    args = parser.parse_args()
    if args.offline_check:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.pop(name, None)
    import torch
    import datasets
    import data_utils
    import transformers
    DATA.mkdir(parents=True, exist_ok=True)
    train = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    calibration = data_utils.get_loaders("wikitext2", nsamples=128, seed=0, seqlen=2048, model=BASE)
    evaluation = data_utils.get_loaders("wikitext2", seed=0, seqlen=2048, model=BASE, eval_mode=True)
    assert len(calibration) == 128
    assert all(ids.shape == target.shape == (1, 2048) for ids, target in calibration)
    token_hashes = [hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest() for ids, _ in calibration]
    token_hash = hashlib.sha256(evaluation.input_ids.contiguous().numpy().tobytes()).hexdigest()
    manifest = {
        "dataset": "wikitext/wikitext-2-raw-v1", "train_split": "train", "eval_split": "test",
        "train_fingerprint": train._fingerprint, "test_fingerprint": test._fingerprint,
        "train_download_checksums": train.info.download_checksums,
        "test_download_checksums": test.info.download_checksums,
        "train_cache_files": train.cache_files, "test_cache_files": test.cache_files,
        "base_model": BASE, "nsamples": 128, "seqlen": 2048, "seed": 0,
        "calibration_token_sha256": token_hashes, "eval_token_sha256": token_hash,
        "eval_tokens": evaluation.input_ids.numel(),
        "eval_complete_sequences": evaluation.input_ids.numel() // 2048,
        "datasets": datasets.__version__, "transformers": transformers.__version__,
        "native_data_utils_sha256": hashlib.sha256((ROOT / "fake_quant/data_utils.py").read_bytes()).hexdigest(),
    }
    manifest_path = DATA / "wikitext2_manifest.json"
    if args.offline_check:
        assert json.loads(manifest_path.read_text()) == manifest, "Cached data/tokenization changed"
    else:
        if manifest_path.exists():
            assert json.loads(manifest_path.read_text()) == manifest, "Refusing to overwrite changed dataset provenance"
        else:
            torch.save({"samples": calibration, "eval_input_ids": evaluation.input_ids}, DATA / "wikitext2_tokens.pt")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"data_preparation": "passed", "offline_check": args.offline_check,
                      "calibration_samples": 128, "sequence_length": 2048,
                      "eval_tokens": manifest["eval_tokens"], "manifest": str(manifest_path)}), flush=True)


if __name__ == "__main__":
    main()
