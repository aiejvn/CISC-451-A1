import argparse
import csv
from datetime import datetime

from .config import get_model_config, load_model_configs
from .dataset import load_dev
from .evaluate import evaluate_run
from .generate import generate, make_run_dir
from .paths import RESULTS_DIR

LEVELS = ["easy", "medium", "hard", "extra", "all"]


def append_summary(cfg: dict, run_dir, n: int, metrics: dict) -> None:
    exact = metrics.get("exact matching accuracy", {}).get("exact match", {})
    execu = metrics.get("execution accuracy", {}).get("execution", {})
    path = RESULTS_DIR / "summary.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["model", "run", "n", "family", "quantization"]
                       + [f"exact_{l}" for l in LEVELS] + [f"exec_{l}" for l in LEVELS])
        w.writerow([cfg["repo_id"], run_dir.name, n, cfg.get("family"), cfg.get("quantization")]
                   + [exact.get(l, "") for l in LEVELS] + [execu.get(l, "") for l in LEVELS])


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate + evaluate on Spider dev for one or more models.")
    ap.add_argument("--models", nargs="*", help="HF repo ids from models.yaml (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="subset size (default: full dev, 1034)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--etype", default="match", choices=["match", "exec", "all"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--skip-generate", action="store_true", help="only re-evaluate existing run dirs with --run-id")
    args = ap.parse_args()

    repo_ids = args.models or list(load_model_configs())
    examples = load_dev(args.limit, args.seed)
    for repo_id in repo_ids:
        cfg = get_model_config(repo_id)
        if args.batch_size:
            cfg["batch_size"] = args.batch_size
        run_dir = make_run_dir(repo_id, args.run_id)
        if not args.skip_generate:
            generate(cfg, examples, run_dir, args.seed)
        metrics = evaluate_run(run_dir, args.etype)
        append_summary(cfg, run_dir, len(examples), metrics)
        print(f"{repo_id}: exact match (all) = "
              f"{metrics.get('exact matching accuracy', {}).get('exact match', {}).get('all')}")


if __name__ == "__main__":
    main()
