# Selected completed aggregate results — 2026-10-15

These are exact copies of an explicit allowlist of completed `metrics.json`
files, not a dump of the original `outputs/` tree. `INDEX.json` records source
run paths, published paths, byte sizes and SHA256 hashes. Per-example predictions,
model responses, prompts, dataset files, API exchanges and raw job logs are not
published here; they remain separate local evaluation records.

The snapshot includes BF16, SpinQuant W4A4KV4 and FlatQuant W4A4KV4 general/PPL,
FinGuard, WildGuard, HarmBench DirectRequest, StrongREJECT, and AdvBench proxy
metrics from the listed full runs. It is a baseline archive, not a new experiment
or a claim that every runtime choice is identical across model implementations.
FinGuard FlatQuant uses batch 4 and the selected FlatQuant WildGuard rerun uses
batch 2; consult the frozen configs and operational guide for other settings.

Interpret the original metric names literally:

- General tasks use answer-likelihood accuracy; the mean is unweighted over
  ARC-E, ARC-C, PIQA and WinoGrande. PPL is separate.
- FinGuard/WildGuard are the project's frozen likelihood-based safety adapters,
  not fine-tuned guard models. WildGuard's FlatQuant gap remains unresolved.
- HarmBench `asr` uses the separate HarmBench classifier on saved responses.
- StrongREJECT `mean_score` is the separate API rubric's continuous score, not ASR.
- AdvBench `non_refusal_rate` is the original refusal-string proxy, **not**
  HarmBench-classifier ASR or semantic harmful-compliance success.

No XSTest result is included in this allowlist. QuaRot's native PPL and producing
provenance are in its HF artifact, not presented as validated benchmark-loader
results. Checkpoint runtime differences and pending training/portability work
are documented in `SHARING.md` and the repository README.
