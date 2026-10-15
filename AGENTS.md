# Repository Instructions

- Never run GPU or model inference directly on NCSA Delta login nodes.
- Use Slurm for GPU experiments.
- Never submit Slurm jobs on the user's behalf; provide commands for the user to run.
- Treat quantized models as immutable evaluation inputs; do not quantize inside this repository.
- Prefer minimal changes over large refactors.
- Do not build abstractions for hypothetical future requirements.
- Preserve identical evaluation settings across full-precision and quantized models.
- Save per-example predictions separately from aggregate metrics.
- Do not silently change scientific evaluation choices.
- Do not expand the research scope unless explicitly requested.
- Refer to the initial quantized condition as SpinQuant W4A4KV4.
