"""Download a pinned, data/code-only snapshot; never import its custom runtime."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant"
REVISION = "65380056097d262d882e427359c9c8be2d63abf7"
MODEL_DIR = Path("/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/flatquant-w4a4kv4")
SETUP = Path(__file__).resolve().parent


def main():
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(MODEL_DIR),
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.cpp", "*.cu", "*.h", "*.cuh", "README.md"],
        ignore_patterns=["*.so", "*.pyc", "**/__pycache__/**"],
        max_workers=4,
    )
    files = {}
    for path in sorted(MODEL_DIR.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(MODEL_DIR).parts:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        files[str(path.relative_to(MODEL_DIR))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) <= set(files), "Missing weight shard"
    assert not list(MODEL_DIR.rglob("*.so")), "Do not use a bundled CUDA binary"
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "directory": str(MODEL_DIR), "files": files}
    (SETUP / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"download": "complete", "files": len(files), "bytes": sum(f["bytes"] for f in files.values()), "manifest": str(SETUP / "checkpoint_manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
