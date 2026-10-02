import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .common import DB_DIR, EVALUATION_PY, SPIDER_DIR, TABLES_JSON

ROW_RE = re.compile(r"^(\S.*?)\s{2,}(-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)*)\s*$")
LEVELS = ["easy", "medium", "hard", "extra", "all"]
SECTION_RE = re.compile(r"^[=\-]+\s*([A-Za-z0-9 ]+?)\s*[=\-]+$")


def parse_eval_output(text: str) -> dict:
    """Parse evaluation.py stdout into {section: {row_label: {level: value}}}."""
    levels = None
    section = "header"
    metrics: dict[str, dict] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if levels is None and stripped.split() == LEVELS:  # not the "easy pred: ..." error lines
            levels = stripped.split()
            continue
        sec = SECTION_RE.match(stripped)
        if sec:
            section = sec.group(1).strip().lower()
            continue
        row = ROW_RE.match(line)
        if row and levels:
            values = row.group(2).split()
            if len(values) == len(levels):
                label = row.group(1).strip()
                metrics.setdefault(section if label != "count" else "count", {})[label] = dict(
                    zip(levels, (float(v) for v in values))
                )
    return metrics


def evaluate_run(run_dir: Path, etype: str = "match") -> dict:
    run_dir = Path(run_dir).resolve()  # evaluation.py runs with cwd=SPIDER_DIR
    if not DB_DIR.exists():
        raise FileNotFoundError(f"{DB_DIR} missing; check the spider submodule is checked out")
    proc = subprocess.run(
        [
            sys.executable, str(EVALUATION_PY),
            "--gold", str(run_dir / "gold.txt"),
            "--pred", str(run_dir / "pred.txt"),
            "--db", str(DB_DIR),
            "--table", str(TABLES_JSON),
            "--etype", etype,
        ],
        cwd=SPIDER_DIR,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SPIDER_DIR), os.environ.get("PYTHONPATH")]))},
        capture_output=True,
        text=True,
    )
    (run_dir / "eval_raw.txt").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluation.py failed, see {run_dir / 'eval_raw.txt'}")
    metrics = parse_eval_output(proc.stdout)
    metrics["etype"] = etype
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def ensure_inputs(run_dir: Path) -> int:
    """Rebuild gold.txt / pred.txt from generations.jsonl when missing; return the example count."""
    rows = [json.loads(line) for line in (run_dir / "generations.jsonl").read_text().splitlines() if line.strip()]
    if not (run_dir / "gold.txt").exists():
        (run_dir / "gold.txt").write_text(
            "".join(f"{' '.join(r['gold'].split())}\t{r['db_id']}\n" for r in rows))
    if not (run_dir / "pred.txt").exists():
        (run_dir / "pred.txt").write_text("".join(f"{r['pred']}\n" for r in rows))
    return len(rows)


def find_runs(path: Path) -> list[Path]:
    """`path` is either one run dir (has generations.jsonl) or a model dir whose subdirs are runs."""
    def has_gens(d: Path) -> bool:  # skip aborted runs whose generations.jsonl is empty
        g = d / "generations.jsonl"
        return g.exists() and g.stat().st_size > 0

    if has_gens(path):
        return [path]
    return sorted(d for d in path.iterdir() if d.is_dir() and has_gens(d))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate saved generations for one model dir (or a single run dir).")
    ap.add_argument("model_dir", type=Path, help="e.g. eval/results/Qwen__Qwen3-0.6B (all runs) or one run dir")
    ap.add_argument("--etype", default="all", choices=["match", "exec", "all"])
    ap.add_argument("--force", action="store_true", help="re-evaluate runs that already have metrics.json")
    args = ap.parse_args()

    runs = find_runs(args.model_dir)
    if not runs:
        raise SystemExit(f"no runs with generations.jsonl under {args.model_dir}")
    print(f"{'run':<20} {'n':>5} " + " ".join(f"{l:>7}" for l in LEVELS))
    for run_dir in runs:
        if (run_dir / "metrics.json").exists() and not args.force:
            m = json.loads((run_dir / "metrics.json").read_text())
            note = " (cached)"
        else:
            ensure_inputs(run_dir)
            m, note = evaluate_run(run_dir, args.etype), ""
        n = ensure_inputs(run_dir)
        exact = m.get("exact matching accuracy", {}).get("exact match", {})
        execu = m.get("execution accuracy", {}).get("execution", {})
        scores = exact or execu
        print(f"{run_dir.name:<20} {n:>5} " + " ".join(f"{scores.get(l, float('nan')):>7.3f}" for l in LEVELS) + note)


if __name__ == "__main__":
    main()
