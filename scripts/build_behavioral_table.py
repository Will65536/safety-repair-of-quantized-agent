#!/usr/bin/env python
"""Print the compact BF16 versus SpinQuant behavioral result table."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmbench", type=Path, required=True)
    parser.add_argument("--strongreject", type=Path)
    parser.add_argument("--xstest", type=Path)
    args = parser.parse_args()
    results = {}
    for name in ("harmbench", "strongreject", "xstest"):
        path = getattr(args, name)
        if path:
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    def value(name, variant, key, subset=None):
        result = results.get(name)
        if result is None:
            return "—"
        metric = result[f"{variant}_metrics"]
        number = metric["subsets"][subset][key] if subset else metric[key]
        return f"{number:.3f}"
    print("| Model | HarmBench ASR | StrongREJECT-small | XSTest safe refusal | XSTest unsafe refusal |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for variant, label in (("baseline", "BF16"), ("variant", "SpinQuant W4A4KV4")):
        print(f"| {label} | {value('harmbench', variant, 'asr')} | {value('strongreject', variant, 'mean_score')} | {value('xstest', variant, 'refusal_rate', 'safe')} | {value('xstest', variant, 'refusal_rate', 'unsafe_contrast')} |")
    flips = results["harmbench"]["paired_flips"]
    print(f"\nHarmBench paired flips: BF16 safe → SpinQuant harmful: {flips['bf16_safe_to_spinquant_harmful']}; BF16 harmful → SpinQuant safe: {flips['bf16_harmful_to_spinquant_safe']}.")


if __name__ == "__main__":
    main()
