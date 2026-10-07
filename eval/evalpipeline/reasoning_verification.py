"""Task 2 (RQ2) verification: confirmatory tests on top of the Spearman correlation analysis in
reasoning_metrics.py.

    python -m eval.evalpipeline.reasoning_verification \\
        --per-question eval/results/task2_analysis/per_question_metrics.csv \\
        --out eval/results/task2_analysis

Four primary tests (one per feature x correctness/difficulty cell):
  - Mann-Whitney U: Token Count / Information Retention, correct vs. incorrect (exact match),
    overall and within each difficulty stratum.
  - Kendall's tau-b: Token Count / Information Retention vs. difficulty, overall and within each
    correctness stratum. Preferred here over a second Spearman run because it is the standard
    choice when one variable has many ties (difficulty has only four distinct values, and
    Information Retention has a large tied mass at 1.0), and is the ordered-groups analogue of the
    Jonckheere-Terpstra trend test.

Two robustness checks (do the bivariate relationships survive controlling for the other variable?):
  - Logistic regression: exact match ~ standardized log Token Count + standardized Information
    Retention + standardized difficulty (jointly).
  - Ordinal (proportional-odds) regression: difficulty ~ standardized log Token Count +
    standardized Information Retention (jointly).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

LEVELS = ["easy", "medium", "hard", "extra"]


def load(per_question_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_question_csv)
    df["difficulty"] = pd.Categorical(df["difficulty"], categories=LEVELS, ordered=True)
    df["difficulty_code"] = df["difficulty"].cat.codes.astype(float)
    return df


def mann_whitney_by_correctness(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """metric by exact-match correctness, overall and within each difficulty stratum."""
    rows = []
    for level in ["all"] + LEVELS:
        sub = df if level == "all" else df[df["difficulty"] == level]
        g0 = sub.loc[sub["exact"] == 0.0, metric].dropna()
        g1 = sub.loc[sub["exact"] == 1.0, metric].dropna()
        if len(g0) < 2 or len(g1) < 2:
            rows.append({"level": level, "n_incorrect": len(g0), "n_correct": len(g1),
                         "U": np.nan, "p": np.nan, "rank_biserial_r": np.nan})
            continue
        U, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        r = 1 - (2 * U) / (len(g0) * len(g1))
        rows.append({"level": level, "n_incorrect": len(g0), "n_correct": len(g1),
                     "U": U, "p": p, "rank_biserial_r": r})
    return pd.DataFrame(rows)


def kendall_by_difficulty(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """metric vs. difficulty (tau-b), overall and within each correctness stratum."""
    rows = []
    for group in ["all", "incorrect", "correct"]:
        if group == "all":
            sub = df
        else:
            sub = df[df["exact"] == (0.0 if group == "incorrect" else 1.0)]
        pair = sub[[metric, "difficulty_code"]].dropna()
        if len(pair) < 5 or pair["difficulty_code"].nunique() < 2:
            rows.append({"group": group, "n": len(pair), "tau_b": np.nan, "p": np.nan})
            continue
        tau, p = stats.kendalltau(pair[metric], pair["difficulty_code"])
        rows.append({"group": group, "n": len(pair), "tau_b": tau, "p": p})
    return pd.DataFrame(rows)


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def logistic_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """exact ~ log(token_count) + retention + difficulty, all standardized, fit jointly."""
    work = df.copy()
    work["z_log_token_count"] = _zscore(np.log1p(work["token_count"]))
    work["z_retention"] = _zscore(work["retention"])
    work["z_difficulty"] = _zscore(work["difficulty_code"])
    reg = work[["exact", "z_log_token_count", "z_retention", "z_difficulty"]].dropna()
    X = sm.add_constant(reg[["z_log_token_count", "z_retention", "z_difficulty"]])
    res = sm.Logit(reg["exact"], X).fit(disp=0)
    ci = res.conf_int()
    out = pd.DataFrame({
        "predictor": res.params.index, "coef": res.params.values, "std_err": res.bse.values,
        "z": res.tvalues.values, "p": res.pvalues.values, "odds_ratio": np.exp(res.params.values),
        "or_ci_low": np.exp(ci[0].values), "or_ci_high": np.exp(ci[1].values),
    })
    out.attrs["n"] = len(reg)
    out.attrs["pseudo_r2"] = res.prsquared
    return out


def ordinal_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """difficulty ~ log(token_count) + retention, both standardized, fit jointly (proportional odds)."""
    work = df.copy()
    work["z_log_token_count"] = _zscore(np.log1p(work["token_count"]))
    work["z_retention"] = _zscore(work["retention"])
    reg = work[["difficulty_code", "z_log_token_count", "z_retention"]].dropna()
    mod = OrderedModel(reg["difficulty_code"], reg[["z_log_token_count", "z_retention"]], distr="logit")
    res = mod.fit(method="bfgs", disp=0)
    exog_names = ["z_log_token_count", "z_retention"]
    params = res.params[exog_names]
    bse = res.bse[exog_names]
    ci = res.conf_int().loc[exog_names]
    out = pd.DataFrame({
        "predictor": exog_names, "coef": params.values, "std_err": bse.values,
        "z": (params / bse).values, "p": res.pvalues[exog_names].values,
        "odds_ratio": np.exp(params.values), "or_ci_low": np.exp(ci[0].values), "or_ci_high": np.exp(ci[1].values),
    })
    out.attrs["n"] = len(reg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-question", type=Path, default=Path("eval/results/task2_analysis/per_question_metrics.csv"))
    ap.add_argument("--out", type=Path, default=Path("eval/results/task2_analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load(args.per_question)

    mw_tc = mann_whitney_by_correctness(df, "token_count")
    mw_ret = mann_whitney_by_correctness(df, "retention")
    mw_tc.to_csv(args.out / "mann_whitney_tokencount.csv", index=False)
    mw_ret.to_csv(args.out / "mann_whitney_retention.csv", index=False)
    print("=== Mann-Whitney U: Token Count by correctness ===")
    print(mw_tc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Mann-Whitney U: Information Retention by correctness ===")
    print(mw_ret.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    kt_tc = kendall_by_difficulty(df, "token_count")
    kt_ret = kendall_by_difficulty(df, "retention")
    kt_tc.to_csv(args.out / "kendall_tokencount.csv", index=False)
    kt_ret.to_csv(args.out / "kendall_retention.csv", index=False)
    print("\n=== Kendall's tau-b: Token Count vs. difficulty ===")
    print(kt_tc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n=== Kendall's tau-b: Information Retention vs. difficulty ===")
    print(kt_ret.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    logit = logistic_robustness(df)
    logit.to_csv(args.out / "logistic_regression.csv", index=False)
    print(f"\n=== Logistic regression: exact ~ token_count + retention + difficulty (n={logit.attrs['n']}, "
          f"pseudo R2={logit.attrs['pseudo_r2']:.3f}) ===")
    print(logit.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    ordinal = ordinal_robustness(df)
    ordinal.to_csv(args.out / "ordinal_regression.csv", index=False)
    print(f"\n=== Ordinal regression: difficulty ~ token_count + retention (n={ordinal.attrs['n']}) ===")
    print(ordinal.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
