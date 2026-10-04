"""Task 2 (RQ2): relate reasoning-trace Token Count and Information Retention to Text-to-SQL performance and difficulty.

    SPIDER_DB_DIR=mock_spider/database python -m eval.evalpipeline.reasoning_metrics \\
        --run eval/results/deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B/task1_orig \\
        --out eval/results/task2_analysis

Token Count: number of tokens in the "thinking" segment of the trace, counted with the model's own tokenizer
(not generations.jsonl's "new_tokens", which also includes the SQL answer).

Information Retention: of the schema identifiers (table/column names) a trace mentions -- quoted, e.g.
"Singer_ID", or written with an underscore, e.g. concert_id -- the share that actually exist in that
question's schema. An identifier that matches neither is a fact not in the prompt and not derivable from it,
i.e. a hallucination (see README Task 2). Traces that mention no schema identifiers get no score (NaN) and
are dropped from Information Retention statistics, but are kept for Token Count.

Correctness: rather than a single correct/incorrect label (too sparse per difficulty level on hard/extra
problems, see README Task 2), we keep every Task 1 submetric separate: accuracy, recall and F1 for each of
the 10 Spider components, plus exact matching and execution accuracy -- 32 per-question scores, each
correlated against each reasoning metric with Spearman's Rank Correlation, per difficulty level.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .common import DB_DIR, REPO_ROOT, TABLES_JSON

sys.path.insert(0, str(REPO_ROOT / "spider"))
from process_sql import Schema, get_schema, get_sql  # noqa: E402
from evaluation import (  # noqa: E402
    Evaluator,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    eval_exec_match,
    rebuild_sql_col,
    rebuild_sql_val,
)

PARTIAL_TYPES = ["select", "select(no AGG)", "where", "where(no OP)", "group(no Having)",
                 "group", "order", "and/or", "IUEN", "keywords"]
LEVELS = ["easy", "medium", "hard", "extra"]
LEVEL_RANK = {l: i for i, l in enumerate(LEVELS, start=1)}
EMPTY_SQL = {"except": None, "from": {"conds": [], "table_units": []}, "groupBy": [], "having": [],
             "intersect": None, "limit": None, "orderBy": [], "select": [False, []], "union": None, "where": []}
IDENT_RE = re.compile(r"[\"'`]([A-Za-z][A-Za-z0-9_]*)[\"'`]|\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b")

SUBMETRICS = [f"{stat}::{t}" for t in PARTIAL_TYPES for stat in ("acc", "rec", "f1")] + ["exact", "execution"]


def load_rows(run_dir: Path) -> list[dict]:
    rows = [json.loads(l) for l in (run_dir / "generations.jsonl").read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: r["id"])


def thinking_tokens(rows: list[dict], repo_id: str) -> dict[int, int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo_id, local_files_only=True)
    return {r["id"]: len(tok(r["thinking"], add_special_tokens=False)["input_ids"]) for r in rows}


_schema_ident_cache: dict[str, set[str]] = {}


def schema_identifiers(db_id: str) -> set[str]:
    if db_id not in _schema_ident_cache:
        schema = get_schema(str(DB_DIR / db_id / f"{db_id}.sqlite"))
        idents = set(schema.keys())
        for cols in schema.values():
            idents.update(cols)
        _schema_ident_cache[db_id] = idents
    return _schema_ident_cache[db_id]


def mentioned_identifiers(thinking: str) -> set[str]:
    found = set()
    for quoted, bare in IDENT_RE.findall(thinking):
        tok = (quoted or bare).strip().lower()
        if len(tok) >= 3:
            found.add(tok)
    return found


def information_retention(rows: list[dict]) -> dict[int, float]:
    out = {}
    for r in rows:
        idents = mentioned_identifiers(r["thinking"])
        if not idents:
            out[r["id"]] = float("nan")
            continue
        grounded = idents & schema_identifiers(r["db_id"])
        out[r["id"]] = len(grounded) / len(idents)
    return out


def per_question_eval(rows: list[dict]) -> list[dict]:
    """Per-question hardness, exact match, 10-component partial scores and execution match (no subprocess)."""
    kmaps = build_foreign_key_map_from_json(TABLES_JSON)
    ev = Evaluator()
    out = []
    for r in rows:
        db_path = str(DB_DIR / r["db_id"] / f"{r['db_id']}.sqlite")
        schema = Schema(get_schema(db_path))
        g_sql = get_sql(schema, r["gold"])
        hardness = ev.eval_hardness(g_sql)
        try:
            p_sql = get_sql(schema, r["pred"])
        except Exception:
            p_sql = EMPTY_SQL
        kmap = kmaps[r["db_id"]]
        g_valid = build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = rebuild_sql_col(g_valid, rebuild_sql_val(g_sql), kmap)
        p_valid = build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = rebuild_sql_col(p_valid, rebuild_sql_val(p_sql), kmap)

        exact = ev.eval_exact_match(p_sql, g_sql)
        partial = ev.partial_scores
        execution = eval_exec_match(db_path, r["pred"], r["gold"], p_sql, g_sql)

        row = {"id": r["id"], "difficulty": hardness, "exact": float(exact), "execution": float(bool(execution))}
        for t in PARTIAL_TYPES:
            s = partial[t]
            row[f"acc::{t}"] = s["acc"] if s["pred_total"] > 0 else np.nan
            row[f"rec::{t}"] = s["rec"] if s["label_total"] > 0 else np.nan
            row[f"f1::{t}"] = s["f1"]
        out.append(row)
    return out


def build_table(run_dir: Path, repo_id: str) -> pd.DataFrame:
    rows = load_rows(run_dir)
    tok = thinking_tokens(rows, repo_id)
    ret = information_retention(rows)
    scored = {s["id"]: s for s in per_question_eval(rows)}
    records = []
    for r in rows:
        rec = {"id": r["id"], "db_id": r["db_id"], "token_count": tok[r["id"]], "retention": ret[r["id"]]}
        rec.update(scored[r["id"]])
        records.append(rec)
    df = pd.DataFrame(records)
    df["difficulty"] = pd.Categorical(df["difficulty"], categories=LEVELS, ordered=True)
    return df


def bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def descriptive_stats(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for metric in ("token_count", "retention"):
        for by, groups in (("difficulty", LEVELS), ("exact", [0.0, 1.0])):
            for g in groups:
                vals = df.loc[df[by] == g, metric].dropna()
                if len(vals) == 0:
                    continue
                rows.append({"metric": metric, "grouped_by": by, "group": g, "n": len(vals),
                             "mean": vals.mean(), "median": vals.median(), "std": vals.std(),
                             "min": vals.min(), "max": vals.max()})
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / "descriptive_stats.csv", index=False)
    print("\n=== Descriptive statistics (Token Count & Information Retention, by difficulty and by exact-match correctness) ===")
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def correlation_tests(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for metric in ("token_count", "retention"):
        rho, p = stats.spearmanr(df[metric], df["difficulty"].cat.codes, nan_policy="omit")
        rows.append({"metric": metric, "submetric": "difficulty", "level": "all", "n": df[metric].notna().sum(),
                     "rho": rho, "p": p})
        for level in LEVELS:
            sub = df[df["difficulty"] == level]
            for submetric in SUBMETRICS:
                pair = sub[[metric, submetric]].dropna()
                if len(pair) < 5 or pair[submetric].nunique() < 2 or pair[metric].nunique() < 2:
                    continue
                rho, p = stats.spearmanr(pair[metric], pair[submetric])
                rows.append({"metric": metric, "submetric": submetric, "level": level, "n": len(pair),
                             "rho": rho, "p": p})
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_correct(res["p"].to_numpy())
    res = res.sort_values("q_bh")
    res.to_csv(out_dir / "spearman_correlations.csv", index=False)

    print(f"\n=== Spearman correlations: {len(res)} tests, Benjamini-Hochberg corrected ===")
    print(f"Significant at q < 0.05: {(res['q_bh'] < 0.05).sum()} / {len(res)}")
    print("\nTop 15 by q-value:")
    print(res.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return res


def make_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    specs = [("token_count", "Token Count", "difficulty", LEVELS),
             ("token_count", "Token Count", "exact", [0.0, 1.0]),
             ("retention", "Information Retention", "difficulty", LEVELS),
             ("retention", "Information Retention", "exact", [0.0, 1.0])]
    for ax, (metric, label, by, groups) in zip(axes.flat, specs):
        data = [df.loc[df[by] == g, metric].dropna() for g in groups]
        labels = [str(g) if by == "difficulty" else ("incorrect" if g == 0.0 else "correct") for g in groups]
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_title(f"{label} by {'difficulty' if by == 'difficulty' else 'exact-match correctness'}")
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(out_dir / "boxplots.png", dpi=150)
    print(f"\nSaved box plots to {out_dir / 'boxplots.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir with generations.jsonl (reasoning model)")
    ap.add_argument("--repo-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="tokenizer for Token Count")
    ap.add_argument("--out", type=Path, default=Path("eval/results/task2_analysis"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = build_table(args.run, args.repo_id)
    df.to_csv(args.out / "per_question_metrics.csv", index=False)
    print(f"Loaded {len(df)} questions from {args.run}; saved per-question metrics to {args.out / 'per_question_metrics.csv'}")

    descriptive_stats(df, args.out)
    correlation_tests(df, args.out)
    make_boxplots(df, args.out)


if __name__ == "__main__":
    main()
