"""Task 2 (RQ2) verification: regression-based confirmatory checks on top of the Spearman
correlation analysis in reasoning_metrics.py.

    python -m eval.evalpipeline.reasoning_verification \\
        --per-question eval/results/task2_analysis/per_question_metrics.csv \\
        --out eval/results/task2_analysis

Two robustness checks (do the bivariate relationships survive controlling for the other variable?):
  - Logistic regression: exact match ~ standardized log Token Count + standardized Information
    Retention + standardized difficulty (jointly).
  - Ordinal (proportional-odds) regression: difficulty ~ standardized log Token Count +
    standardized Information Retention (jointly).

Regression diagnostics on top of the above:
  - Variance inflation factors for the three logistic-regression predictors (checks whether the
    robustness-check coefficients in logistic_robustness() are distorted by collinearity between
    Token Count, Retention and difficulty).
  - A proportional-odds check for the ordinal model (compares the per-cutoff cumulative-logit
    coefficients against the single pooled coefficient OrderedModel assumes is constant across
    cutoffs).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

LEVELS = ["easy", "medium", "hard", "extra"]


def load(per_question_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(per_question_csv)
    df["difficulty"] = pd.Categorical(df["difficulty"], categories=LEVELS, ordered=True)
    df["difficulty_code"] = df["difficulty"].cat.codes.astype(float)
    return df


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors for the three predictors used in logistic_robustness()."""
    work = df.copy()
    work["z_log_token_count"] = _zscore(np.log1p(work["token_count"]))
    work["z_retention"] = _zscore(work["retention"])
    work["z_difficulty"] = _zscore(work["difficulty_code"])
    reg = work[["z_log_token_count", "z_retention", "z_difficulty"]].dropna()
    X = sm.add_constant(reg)
    rows = [{"predictor": col, "VIF": variance_inflation_factor(X.values, i)}
            for i, col in enumerate(X.columns) if col != "const"]
    out = pd.DataFrame(rows)
    out.attrs["n"] = len(reg)
    return out


def proportional_odds_check(df: pd.DataFrame) -> pd.DataFrame:
    """Compares the pooled ordinal-regression coefficients (which assume a single, constant effect
    across all difficulty cutoffs) against separately-fit cumulative-logit coefficients at each
    cutoff. Large, systematic drift across cutoffs is evidence against the proportional-odds
    assumption that ordinal_robustness() relies on."""
    work = df.copy()
    work["z_log_token_count"] = _zscore(np.log1p(work["token_count"]))
    work["z_retention"] = _zscore(work["retention"])
    reg = work[["difficulty_code", "z_log_token_count", "z_retention"]].dropna()

    mod = OrderedModel(reg["difficulty_code"], reg[["z_log_token_count", "z_retention"]], distr="logit")
    pooled = mod.fit(method="bfgs", disp=0)
    rows = [{"cutoff": "pooled (proportional-odds)",
             "coef_token_count": pooled.params["z_log_token_count"], "se_token_count": pooled.bse["z_log_token_count"],
             "coef_retention": pooled.params["z_retention"], "se_retention": pooled.bse["z_retention"]}]

    X = sm.add_constant(reg[["z_log_token_count", "z_retention"]])
    for k in (1, 2, 3):
        y = (reg["difficulty_code"] >= k).astype(int)
        res = sm.Logit(y, X).fit(disp=0)
        rows.append({"cutoff": f"P(difficulty >= {LEVELS[k]})",
                     "coef_token_count": res.params["z_log_token_count"], "se_token_count": res.bse["z_log_token_count"],
                     "coef_retention": res.params["z_retention"], "se_retention": res.bse["z_retention"]})
    out = pd.DataFrame(rows)
    out.attrs["n"] = len(reg)
    return out


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

    logit = logistic_robustness(df)
    logit.to_csv(args.out / "logistic_regression.csv", index=False)
    print(f"\n=== Logistic regression: exact ~ token_count + retention + difficulty (n={logit.attrs['n']}, "
          f"pseudo R2={logit.attrs['pseudo_r2']:.3f}) ===")
    print(logit.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    ordinal = ordinal_robustness(df)
    ordinal.to_csv(args.out / "ordinal_regression.csv", index=False)
    print(f"\n=== Ordinal regression: difficulty ~ token_count + retention (n={ordinal.attrs['n']}) ===")
    print(ordinal.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    vif = vif_table(df)
    vif.to_csv(args.out / "vif.csv", index=False)
    print(f"\n=== VIF: logistic regression predictors (n={vif.attrs['n']}) ===")
    print(vif.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    po_check = proportional_odds_check(df)
    po_check.to_csv(args.out / "proportional_odds_check.csv", index=False)
    print(f"\n=== Proportional-odds check: per-cutoff cumulative logits vs. pooled ordinal (n={po_check.attrs['n']}) ===")
    print(po_check.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
