"""Load and validate the golden evaluation set."""

from collections import Counter
from pathlib import Path

from src.schemas import GoldenItem

GOLDEN_PATH = Path("data/golden/golden_set.jsonl")


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    """Parse one GoldenItem per non-blank line; fail on schema errors or duplicate ids."""
    items = [
        GoldenItem.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    duplicates = [i for i, n in Counter(item.id for item in items).items() if n > 1]
    if duplicates:
        raise ValueError(f"Duplicate golden ids: {duplicates}")
    return items
