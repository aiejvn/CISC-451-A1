import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .paths import DB_DIR, EVALUATION_PY, SPIDER_DIR, TABLES_JSON

ROW_RE = re.compile(r"^(\S.*?)\s{2,}(-?\d+(?:\.\d+)?(?:\s+-?\d+(?:\.\d+)?)*)\s*$")
SECTION_RE = re.compile(r"^[=\-]+\s*([A-Za-z0-9 ]+?)\s*[=\-]+$")


def parse_eval_output(text: str) -> dict:
    """Parse evaluation.py stdout into {section: {row_label: {level: value}}}."""
    levels = None
    section = "header"
    metrics: dict[str, dict] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if levels is None and stripped.startswith("easy"):
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
    run_dir = Path(run_dir)
    if not DB_DIR.exists():
        raise FileNotFoundError(f"{DB_DIR} missing; run eval/data/build_spider_dbs.py first")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--etype", default="match", choices=["match", "exec", "all"])
    args = ap.parse_args()
    m = evaluate_run(args.run, args.etype)
    print(json.dumps(m.get("exact matching accuracy", m), indent=2))


if __name__ == "__main__":
    main()
