"""Compare two JSON objects mapping dataset case IDs to saved responses."""

import argparse
import json
from pathlib import Path

from promptpulse.comparison import compare_runs
from promptpulse.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        parser.error("Both run files must be JSON objects keyed by case ID.")
    report = compare_runs(load_dataset(), baseline, candidate)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.decision != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
