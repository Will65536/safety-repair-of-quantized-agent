# Third-party source and artifact notices

This repository includes source-only snapshots for reproducibility. Public
availability does not remove the original licenses or make all components share
one license. No new blanket license is assigned to the entire project by this
publication handoff.

| Component | Upstream | Preserved license |
| --- | --- | --- |
| SpinQuant | https://github.com/facebookresearch/SpinQuant | `quantizers/SpinQuant/LICENSE`, CC BY-NC 4.0 |
| QuaRot | https://github.com/spcl/QuaRot | `quantizers/QuaRot/LICENSE`, Apache 2.0 |
| FlatQuant | https://github.com/ruikangliu/FlatQuant | `quantizers/FlatQuant/LICENSE`, MIT |

Upstream commit IDs, original file hashes and pinned submodules are recorded in
`quantizers/SOURCES.json`. Tracked local changes are marked in each quantizer's
`LOCAL_CHANGES.patch`; additional `local_setup/` sources are identified in the
snapshot manifest. Unrelated untracked experiments and results are not included.
Submodule code and vendored kernel headers retain their own notices and licenses.

The FlatQuant inference runtime snapshot is from the published checkpoint
`Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant` at revision
`65380056097d262d882e427359c9c8be2d63abf7`. Credit its publisher and the
FlatQuant authors; this project did not produce those quantized weights.

All three model artifacts derive from Meta Llama 3 and carry the applicable
Meta Llama 3 agreement and notice in their HF repositories. Built with Meta
Llama 3. Model weights are not in this GitHub repository. Dataset and judge
downloads remain subject to their original licenses/access conditions.
