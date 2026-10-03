"""Evaluate adapters on the full Spider dev set with the Task 1 pipeline (Spider evaluation.py), on the mock databases.

    SPIDER_DB_DIR=mock_spider/database python -m train.evaluate                    # all checkpoints below
    SPIDER_DB_DIR=mock_spider/database python -m train.evaluate --only gold_control
    SPIDER_DB_DIR=mock_spider/database python -m train.evaluate --summary          # table from saved metrics
    SPIDER_DB_DIR=mock_spider/database python -m train.evaluate --zero-shot 300    # base model on validation

Each run is written to eval/results/Qwen__Qwen3-0.6B/<name>/ in the same layout as Task 1.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

from eval.evalpipeline.adapters import extract_sql
from eval.evalpipeline.common import REPO_ROOT, RESULTS_DIR, load_dev, write_gold
from eval.evalpipeline.evaluate import evaluate_run
from .common import MAX_NEW, RUNS_DIR as RUNS, build_splits, chat_prompt, eval_examples, generate, load_base, load_lora

MODEL_DIR = RESULTS_DIR / "Qwen__Qwen3-0.6B"
CHECKPOINTS = [  # name, adapter dir, prompt format
    ("sft_sql", RUNS / "e1_sql/adapter", "sql"),
    ("sft_plan", RUNS / "e1_plan/adapter", "plan"),
    ("gold_control", RUNS / "c1_gold/r1_gold", "plan"),
    ("vbi", RUNS / "c1_vbi/r1_vbi", "plan"),
    ("pp1_vbi", RUNS / "pp1/r1_vbi", "plan"),
    ("pp1_selfplay", RUNS / "pp1/r1_sp", "plan"),
]
TASK1_RUN = REPO_ROOT / "451_gen_results_2026-10-02/Qwen__Qwen3-0.6B/20261002-154642"  # Task 1 greedy run


def baseline_on_mock_db() -> None:
    """Re-score the existing Task 1 generations on the mock databases (copy; the committed results stay untouched)."""
    out = MODEL_DIR / "base_task1"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("generations.jsonl", "gold.txt", "pred.txt", "config_used.yaml"):
        shutil.copy(TASK1_RUN / f, out / f)
    evaluate_run(out, "all")


def run_one(name: str, adapter: Path, fmt: str) -> None:
    dev = load_dev()
    tok, base = load_base(checkpoint=False)
    model = load_lora(base, adapter, trainable=False)
    prompts = [chat_prompt(tok, ex["question"], ex["db_id"], fmt) for ex in dev]
    raws = [o[0] for o in generate(model, tok, prompts, MAX_NEW[fmt])]
    out = MODEL_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    write_gold(dev, out / "gold.txt")
    preds = [extract_sql(r) for r in raws]
    (out / "pred.txt").write_text("\n".join(preds) + "\n")
    with open(out / "generations.jsonl", "w") as f:
        for ex, raw, pred in zip(dev, raws, preds):
            f.write(json.dumps({"id": ex["id"], "db_id": ex["db_id"], "question": ex["question"], "gold": ex["query"],
                                "raw": raw, "thinking": None, "pred": pred, "new_tokens": len(tok(raw)["input_ids"])}) + "\n")
    (out / "config_used.yaml").write_text(yaml.safe_dump({
        "model": {"repo_id": "Qwen/Qwen3-0.6B", "adapter": str(adapter), "prompt_format": fmt, "thinking": False,
                  "max_new_tokens": MAX_NEW[fmt]},
        "generation": {"do_sample": False}, "n_examples": len(dev)}))
    evaluate_run(out, "all")
    del model, base


def zero_shot(n: int) -> None:
    """Base model (no adapter) on n validation questions (held-out training databases), both prompt formats."""
    tok, base = load_base(checkpoint=False)
    _, val, _ = build_splits(tokenizer=tok)
    val = random.Random(0).sample(val, min(n, len(val)))
    for fmt in ("sql", "plan"):
        r = eval_examples(base, tok, val, fmt, MAX_NEW[fmt])
        print(fmt, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items() if k not in ("texts", "labels")},
              flush=True)


def summary() -> str:
    levels = ["easy", "medium", "hard", "extra", "all"]
    names = ["base_task1"] + [c[0] for c in CHECKPOINTS]
    rows = []
    for name in names:
        mpath = MODEL_DIR / name / "metrics.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text())
        n = {k: int(v) for k, v in m.get("count", {}).get("count", {}).items()}
        em = m["exact matching accuracy"]["exact match"]
        ex = m["execution accuracy"]["execution"]
        rows.append((name, em, ex))
    cnt = next((json.loads((MODEL_DIR / n / "metrics.json").read_text()).get("count", {}).get("count") for n in names
                if (MODEL_DIR / n / "metrics.json").exists()), {})
    lines = ["| Run | " + " | ".join(levels) + " |", "|---|" + "---|" * len(levels)]
    if cnt:
        lines.append("| n | " + " | ".join(str(int(cnt.get(l, 0))) for l in levels) + " |")
    for title, idx in (("exact match", 1), ("execution (mock DB)", 2)):
        lines.append(f"| **{title}** |" + " |" * len(levels))
        for r in rows:
            lines.append(f"| {r[0]} | " + " | ".join(f"{r[idx].get(l, float('nan')):.3f}" for l in levels) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--only", default="", help="comma list of checkpoint names")
    ap.add_argument("--zero-shot", type=int, default=0, metavar="N", help="evaluate the base model on N validation questions")
    args = ap.parse_args()
    if args.zero_shot:
        return zero_shot(args.zero_shot)
    if not args.summary:
        baseline_on_mock_db()
        only = set(filter(None, args.only.split(",")))
        for name, adapter, fmt in CHECKPOINTS:
            if only and name not in only:
                continue
            print(f"== {name} ({fmt}) from {adapter}", flush=True)
            run_one(name, adapter, fmt)
            print(summary(), flush=True)
    print(summary())
    (MODEL_DIR / "summary.md").write_text(summary() + "\n")


if __name__ == "__main__":
    main()
