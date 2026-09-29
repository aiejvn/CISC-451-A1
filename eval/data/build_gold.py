"""Write a Spider gold file (`query \\t db_id` per line) from dev.json."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalpipeline.dataset import load_dev, write_gold  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("gold.txt"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    examples = load_dev(args.limit, args.seed)
    write_gold(examples, args.out)
    print(f"wrote {len(examples)} gold lines to {args.out}")


if __name__ == "__main__":
    main()
