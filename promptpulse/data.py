from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "tests" / "test_dataset.json"


def load_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation dataset must be a non-empty JSON array.")

    required = {"id", "scenario", "user_query", "reference_context", "expected_answer"}
    seen: set[str] = set()

    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {index} must be an object.")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Dataset row {index} is missing: {sorted(missing)}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate dataset id: {row['id']}")
        seen.add(row["id"])
        row.setdefault("required_terms", [])
        row.setdefault("forbidden_terms", [])

    return data
