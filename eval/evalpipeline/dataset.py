import json
import random
from pathlib import Path

from .paths import DEV_JSON


def load_dev(limit: int | None = None, seed: int = 0) -> list[dict]:
    with open(DEV_JSON) as f:
        examples = json.load(f)
    if limit is not None and limit < len(examples):
        idx = sorted(random.Random(seed).sample(range(len(examples)), limit))
        examples = [examples[i] for i in idx]
    return examples


def write_gold(examples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for ex in examples:
            query = " ".join(ex["query"].split())
            f.write(f"{query}\t{ex['db_id']}\n")
