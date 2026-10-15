#!/usr/bin/env python3
"""Recompute aggregate metrics from saved per-example predictions."""

import argparse
import json
from pathlib import Path

from safe_quant.metrics import summarize_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = summarize_predictions(args.predictions)
    if args.output:
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
