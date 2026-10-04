"""Why does a reasoning model score low? Splits a reasoning model's run by generation outcome and scores each part.

    SPIDER_DB_DIR=mock_spider/database python -m eval.evalpipeline.breakdown \
        --reasoning 451_gen_results_2026-10-02/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/20261002-154642 \
        --others 451_gen_results_2026-10-02/Qwen__Qwen3-0.6B/20261002-154642 \
                 451_gen_results_2026-10-02/allenai__OLMo-2-0425-1B-Instruct/20261002-154642

Outcomes: finished with SQL | finished, no usable SQL | cut while thinking (no </think>) | cut while answering.
A generation is "cut" if it reached max_new_tokens or was stopped as a repetition loop.
"""
import argparse
import json
import statistics
import tempfile
import zlib
from pathlib import Path

import yaml

from .evaluate import evaluate_run

LEVELS = ["easy", "medium", "hard", "extra", "all"]
OUTCOMES = ["finished with SQL", "finished, no usable SQL", "cut while answering", "cut while thinking"]


def load(run: Path) -> tuple[list[dict], int]:
    rows = [json.loads(l) for l in (run / "generations.jsonl").read_text().splitlines() if l.strip()]
    limit = yaml.safe_load((run / "config_used.yaml").read_text())["model"]["max_new_tokens"]
    return sorted(rows, key=lambda r: r["id"]), limit


def outcome(row: dict, limit: int) -> str:
    cut = row["new_tokens"] >= limit or row.get("loop_stopped")
    if cut:
        return "cut while answering" if "</think>" in row["raw"] else "cut while thinking"
    return "finished with SQL" if len(row["pred"].split()) >= 3 else "finished, no usable SQL"


def score(rows: list[dict]) -> dict | None:
    """Spider metrics on a subset of rows (gold/pred written to a temp run dir)."""
    if not rows:
        return None
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "gold.txt").write_text("".join(f"{' '.join(r['gold'].split())}\t{r['db_id']}\n" for r in rows))
        (d / "pred.txt").write_text("".join(f"{r['pred']}\n" for r in rows))
        m = evaluate_run(d, "all")
    acc, rec = m["partial matching accuracy"], m["partial matching recall"]
    a = sum(v["all"] for v in acc.values()) / len(acc)
    r = sum(v["all"] for v in rec.values()) / len(rec)
    return {"n": len(rows), "exact": m["exact matching accuracy"]["exact match"]["all"],
            "exec": m["execution accuracy"]["execution"]["all"], "comp_f1": 2 * a * r / (a + r) if a + r else 0.0,
            "by_level": {l: int(m["count"]["count"][l]) for l in LEVELS}}


def tail_stats(rows: list[dict]) -> tuple[float, float, float]:
    """(median compression ratio of the last 3000 chars, share below 0.15, share whose last 150 chars recur >= 3x)."""
    ratios = [len(zlib.compress(r["raw"][-3000:].encode())) / max(1, len(r["raw"][-3000:].encode())) for r in rows]
    recur = [len(r["raw"]) >= 150 and r["raw"].count(r["raw"][-150:]) >= 3 for r in rows]
    return statistics.median(ratios), sum(x < 0.15 for x in ratios) / len(rows), sum(recur) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", type=Path, required=True, help="run dir of the reasoning model")
    ap.add_argument("--others", type=Path, nargs="*", default=[], help="run dirs of other models (same questions)")
    args = ap.parse_args()

    rows, limit = load(args.reasoning)
    for r in rows:
        r["outcome"] = outcome(r, limit)
    n = len(rows)
    print(f"max_new_tokens = {limit}, {n} generations\n")

    print("| Outcome | Problems (n) | Share | Avg. tokens | Exact match | Component F1 | Execution (mock DB) |")
    print("|---|---|---|---|---|---|---|")
    for o in OUTCOMES:
        part = [r for r in rows if r["outcome"] == o]
        s = score(part)
        tok = sum(r["new_tokens"] for r in part) / len(part) if part else 0
        print(f"| {o} | {len(part)} | {len(part) / n:.1%} | {tok:.0f} | "
              + (f"{s['exact']:.3f} | {s['comp_f1']:.3f} | {s['exec']:.3f} |" if s else "- | - | - |"))
    s = score(rows)
    print(f"| all | {n} | 100% | {sum(r['new_tokens'] for r in rows) / n:.0f} | {s['exact']:.3f} | {s['comp_f1']:.3f} | {s['exec']:.3f} |\n")

    fin = {r["id"] for r in rows if r["outcome"] == "finished with SQL"}
    print(f"Same {len(fin)} questions (reasoning model finished with SQL), every model:\n")
    print("| Model | Exact match | Execution (mock DB) | Component F1 |\n|---|---|---|---|")
    for run in [args.reasoning, *args.others]:
        rr, _ = load(run)
        s = score([r for r in rr if r["id"] in fin])
        print(f"| {run.parent.name} | {s['exact']:.3f} | {s['exec']:.3f} | {s['comp_f1']:.3f} |")

    print("\nAre the cut-off generations repetition loops? (last 3000 characters of the raw output)\n")
    print("| Generations | n | Median compression ratio | Compression ratio < 0.15 | Last 150 chars repeat >= 3x |\n|---|---|---|---|---|")
    for name, part in (("cut off", [r for r in rows if r["outcome"].startswith("cut")]),
                       ("finished", [r for r in rows if not r["outcome"].startswith("cut")])):
        m, lo, rc = tail_stats(part)
        print(f"| {name} | {len(part)} | {m:.3f} | {lo:.1%} | {rc:.1%} |")

    print("\nShare of the reasoning model's generations that did not finish with SQL, by difficulty:\n")
    bad = score([r for r in rows if r["outcome"] != "finished with SQL"])
    tot = score(rows)["by_level"]
    print("| | " + " | ".join(LEVELS) + " |\n|---|" + "---|" * len(LEVELS))
    print("| Problems (n) | " + " | ".join(str(tot[l]) for l in LEVELS) + " |")
    print("| Not finished with SQL (n) | " + " | ".join(str(bad["by_level"][l]) for l in LEVELS) + " |")
    print("| Share | " + " | ".join(f"{bad['by_level'][l] / tot[l]:.1%}" for l in LEVELS) + " |")


if __name__ == "__main__":
    main()
