"""H1-H3 starter analysis; run from your repo: python scripts/analyze_h1_h3.py.

Install: python -m pip install numpy pandas scipy matplotlib
Inputs: data/clean/survey_responses.csv and passage_answer_key.csv.
Outputs: outputs/h1_h3/ (source CSVs are never changed).
VS Code recognizes # %% sections as runnable Python cells.
"""

# %% 1. Setup
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # Save charts without needing an interactive window.
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SEED = 510
ALPHA_PRIMARY = 0.05
# Keep three secondary tests in the family: H2, H3 AND your partner's H4.
ALPHA_SECONDARY = 0.05 / 3
BLUE, ORANGE, GRAY = "#245A81", "#BD591D", "#626870"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140,
                     "savefig.dpi": 200, "axes.titleweight": "bold"})


def find_input(name, data_dir):
    """Allow an answer key in a different data subfolder; never guess duplicates."""
    direct = data_dir / name
    if direct.exists():
        return direct
    matches = list((ROOT / "data").rglob(name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Place {name} in {data_dir}, or pass --data-dir PATH.")


# %% 2. Validate inputs; preserve missing demographics and flag, not delete, effort issues
def prepare_data(response_path, key_path):
    df = pd.read_csv(response_path)
    key = pd.read_csv(key_path)
    origin_cols = [f"p{i}_origin_rating" for i in range(1, 7)]
    quality_cols = [f"p{i}_quality" for i in range(1, 7)]
    rating_cols = origin_cols + quality_cols
    required = ["response_id", "native_english", "english_proficiency"] + rating_cols
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.response_id.isna().any() or df.response_id.duplicated().any():
        raise ValueError("Response IDs must be present and unique; inspect duplicates.")
    if not {"passage_id", "correct_origin"}.issubset(key.columns):
        raise ValueError("Answer key needs passage_id and correct_origin.")
    if len(key) != 6 or set(key.passage_id) != set(range(1, 7)):
        raise ValueError("Answer key must have exactly one row for each passage 1-6.")
    key["correct_origin"] = key.correct_origin.astype("string").str.strip().str.lower()
    expected = {1: "human", 2: "ai", 3: "ai", 4: "human", 5: "ai", 6: "human"}
    if key.set_index("passage_id").correct_origin.to_dict() != expected:
        raise ValueError("Answer key differs from the supplied report. Reconcile before running.")
    for col in rating_cols:
        # errors='raise' prevents malformed text from silently becoming missing.
        df[col] = pd.to_numeric(df[col], errors="raise")
        present = df[col].dropna()
        high = 10 if col in origin_cols else 5
        if not (present.between(1, high) & (present % 1 == 0)).all():
            raise ValueError(f"{col} must contain integers from 1 to {high}.")
    df["native_english"] = df.native_english.astype("string").str.strip().str.lower()
    if not df.native_english.dropna().isin(["yes", "no"]).all():
        raise ValueError("Unexpected native_english value; inspect before recoding.")
    # Do not fill native status or proficiency: H1-H3 do not require these fields.
    incomplete = df[rating_cols].isna().any(axis=1)
    excluded = df.loc[incomplete, ["response_id"]].assign(reason="Missing passage rating")
    audit = {"input_rows": len(df), "excluded_incomplete": int(incomplete.sum()),
             "proficiency_missing": int(df.english_proficiency.isna().sum()),
             "duplicate_rating_patterns": int(df.duplicated(subset=rating_cols).sum())}
    df = df.loc[~incomplete].copy()
    df["flag_identical_quality"] = df[quality_cols].nunique(axis=1).eq(1)
    df["flag_all_origin_extremes"] = df[origin_cols].isin([1, 10]).all(axis=1)
    df["flag_low_effort"] = df.flag_identical_quality | df.flag_all_origin_extremes

    # Long format: one row for each participant-passage pair (32 * 6 = 192).
    long = pd.concat([
        df[["response_id", "native_english", "english_proficiency", "flag_low_effort",
            f"p{i}_quality", f"p{i}_origin_rating"]]
        .rename(columns={f"p{i}_quality": "quality", f"p{i}_origin_rating": "origin_rating"})
        .assign(passage_id=i) for i in range(1, 7)
    ], ignore_index=True).merge(key, on="passage_id", how="left", validate="many_to_one")
    # 1 = certainly AI; 10 = certainly human. Never reverse this scale.
    long["guessed_origin"] = np.where(long.origin_rating <= 5, "ai", "human")
    long["correct"] = long.guessed_origin.eq(long.correct_origin)
    # Absolute distance FIRST, then average. abs(mean(rating)-5.5) is NOT confidence.
    long["confidence"] = (long.origin_rating - 5.5).abs()
    long["human_guess"] = long.guessed_origin.eq("human")

    # Each person contributes ONE observation to H1, H2 and H3.
    people = long.groupby("response_id").agg(
        n_rated=("correct", "size"), n_correct=("correct", "sum"),
        accuracy=("correct", "mean"), mean_source_rating=("origin_rating", "mean"),
        confidence=("confidence", "mean"), human_guess_rate=("human_guess", "mean")
    ).reset_index().merge(
        df[["response_id", "native_english", "english_proficiency", "flag_low_effort",
            "flag_identical_quality", "flag_all_origin_extremes"]],
        on="response_id", validate="one_to_one")
    assert people.n_rated.eq(6).all() and len(long) == 6 * len(people)
    audit.update(analyzed_rows=len(people), participant_passage_rows=len(long),
                 native_counts=people.native_english.value_counts(dropna=False).to_dict(),
                 flagged_ids=people.loc[people.flag_low_effort, "response_id"].tolist())
    return long, people, excluded, audit


# %% 3. Main tests, confidence intervals and effect sizes
def mean_ci(values, level=0.95):
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf((1 + level) / 2, len(values) - 1) * stats.sem(values)
    return values.mean() - half, values.mean() + half


def run_tests(people, sample="all participants"):
    if len(people) < 4:
        raise ValueError("At least four complete participants are needed.")
    rows = []
    for label, col, null, alternative, alpha in [
        ("H1", "accuracy", 0.5, "greater", ALPHA_PRIMARY),
        ("H2", "mean_source_rating", 5.5, "two-sided", ALPHA_SECONDARY)
    ]:
        values = people[col].to_numpy()
        if values.std(ddof=1) == 0:
            raise ValueError(f"No variation in {col}; usual t-test is undefined.")
        test = stats.ttest_1samp(values, popmean=null, alternative=alternative)
        lo, hi = mean_ci(values)  # TWO-sided 95% estimation CI, even for H1.
        d = (values.mean() - null) / values.std(ddof=1)
        # Participant bootstrap for Cohen's d interval (approximate percentile CI).
        draws = np.random.default_rng(SEED).choice(values, (10000, len(values)))
        sd = draws.std(axis=1, ddof=1)
        ds = (draws.mean(axis=1)[sd > 0] - null) / sd[sd > 0]
        dlo, dhi = np.quantile(ds, [0.025, 0.975])
        rows.append(dict(sample=sample, hypothesis=label, n=len(values), estimate=values.mean(),
                         ci95_low=lo, ci95_high=hi, statistic=test.statistic,
                         df=len(values)-1, p_raw=test.pvalue, alpha=alpha,
                         p_bonferroni=min(test.pvalue * 3, 1) if label == "H2" else np.nan,
                         effect_type="Cohen d", effect=d, effect_ci95_low=dlo,
                         effect_ci95_high=dhi, reject_null=bool(test.pvalue < alpha)))
    corr = stats.pearsonr(people.confidence, people.accuracy)
    ci = corr.confidence_interval()  # Fisher-transformed, approximate 95% CI.
    rows.append(dict(sample=sample, hypothesis="H3", n=len(people), estimate=corr.statistic,
                     ci95_low=ci.low, ci95_high=ci.high, statistic=corr.statistic,
                     df=len(people)-2, p_raw=corr.pvalue, alpha=ALPHA_SECONDARY,
                     p_bonferroni=min(corr.pvalue*3, 1), effect_type="Pearson r",
                     effect=corr.statistic, effect_ci95_low=ci.low, effect_ci95_high=ci.high,
                     reject_null=bool(corr.pvalue < ALPHA_SECONDARY)))
    return pd.DataFrame(rows)


# %% 4. Diagnostics and supplementary checks (do not replace the planned tests)
def diagnostics(people):
    normality = []
    for col in ["accuracy", "mean_source_rating", "confidence"]:
        test = stats.shapiro(people[col])
        normality.append({"measure": col, "W": test.statistic, "p": test.pvalue})
    checks = []
    for name, col, null, alt in [("H1", "accuracy", .5, "greater"),
                                ("H2", "mean_source_rating", 5.5, "two-sided")]:
        # Round differences to preserve ties in the discrete six-item scores.
        diff = np.round(people[col].to_numpy() - null, 10)
        result = stats.wilcoxon(diff, alternative=alt, zero_method="wilcox", method="approx")
        checks.append({"hypothesis": name, "check": "Wilcoxon; asymptotic, zeros removed",
                       "statistic": result.statistic, "p": result.pvalue})
    # Accuracy is bounded/discrete. A permutation check supplements Pearson inference.
    x = people.confidence.to_numpy(); y = people.accuracy.to_numpy()
    xc = x - x.mean(); yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    observed = np.dot(xc, yc) / denominator
    rng = np.random.default_rng(SEED)
    permuted = np.array([np.dot(xc, rng.permutation(yc)) / denominator for _ in range(9999)])
    p_perm = (1 + np.sum(abs(permuted) >= abs(observed))) / (1 + len(permuted))
    checks.append({"hypothesis": "H3", "check": "Pearson permutation; 9999 shuffles, two-sided",
                   "statistic": observed, "p": p_perm})
    return pd.DataFrame(normality), pd.DataFrame(checks)


# %% 5. Presentation charts and regression
def make_figures(people, long, results, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    # H1: exact discrete score frequencies; don't smooth six-item accuracy scores.
    ax = axes[0]
    counts = people.n_correct.value_counts().reindex(range(7), fill_value=0)
    ax.bar(np.arange(7) / 6, counts, width=.11, color=BLUE)
    ax.axvline(.5, color=GRAY, linestyle="--", label="50% chance")
    ax.set(xticks=np.arange(7)/6, xlabel="Participant accuracy", ylabel="Participants",
           title="H1 | Accuracy across six passages")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=10)
    h1 = results.iloc[0]
    ax.text(.03, .95, f"Mean {h1.estimate:.1%}\n95% CI {h1.ci95_low:.1%}–{h1.ci95_high:.1%}",
            transform=ax.transAxes, va="top")
    ax.set_ylim(0, max(counts) * 1.35)
    # H2: participant values plus the mean and 95% CI; neutral midpoint explicit.
    ax = axes[1]
    jitter = np.random.default_rng(SEED).uniform(-.13, .13, len(people))
    ax.scatter(people.mean_source_rating, jitter, color=BLUE, s=28, alpha=.7)
    h2 = results.iloc[1]
    ax.errorbar(h2.estimate, .4, xerr=[[h2.estimate-h2.ci95_low], [h2.ci95_high-h2.estimate]],
                fmt="D", color=ORANGE, capsize=6, markersize=7, label="Mean ± 95% CI")
    ax.axvline(5.5, color=GRAY, linestyle="--", label="Neutral midpoint: 5.5")
    ax.set(xlim=(1, 10), ylim=(-.4, .9), yticks=[], xticks=[1, 3, 5.5, 8, 10],
           xlabel="Mean source rating (1 = AI; 10 = human)",
           title="H2 | Direction of source judgments")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    # H3: use actual coordinates; bubble size reveals identical participant pairs.
    ax = axes[2]
    grouped = people.groupby(["confidence", "accuracy"]).size().reset_index(name="count")
    ax.scatter(grouped.confidence, grouped.accuracy, s=35*grouped["count"],
               color=BLUE, alpha=.65, edgecolors="white")
    fit = stats.linregress(people.confidence, people.accuracy)
    grid = np.linspace(people.confidence.min(), people.confidence.max(), 100)
    prediction = fit.intercept + fit.slope * grid
    residual = people.accuracy - (fit.intercept + fit.slope * people.confidence)
    s = np.sqrt(np.sum(residual**2)/(len(people)-2))
    sxx = np.sum((people.confidence-people.confidence.mean())**2)
    band = stats.t.ppf(.975, len(people)-2) * s * np.sqrt(
        1/len(people) + (grid-people.confidence.mean())**2/sxx)
    ax.plot(grid, prediction, color=ORANGE, label="Linear fit")
    ax.fill_between(grid, prediction-band, prediction+band, color=ORANGE, alpha=.14,
                    label="95% CI for fitted mean")
    ax.set(xlim=(.4, 4.6), ylim=(-.03, 1.03), xlabel="Mean confidence proxy (0.5–4.5)",
           ylabel="Participant accuracy", title="H3 | Confidence and accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.text(.97, .03, f"r = {fit.rvalue:.3f}\np = {fit.pvalue:.3f}",
            transform=ax.transAxes, ha="right", va="bottom")
    fig.suptitle(f"AI or human writing? H1–H3 • {len(people)} participants", fontsize=16)
    fig.savefig(out / "h1_h3_overview.png", bbox_inches="tight")
    plt.close(fig)

    # Descriptive passage comparison: each bar uses 32 people, not pooled inference.
    passage = long.groupby(["passage_id", "correct_origin"]).agg(
        n=("correct", "size"), n_correct=("correct", "sum"), accuracy=("correct", "mean"),
        mean_source_rating=("origin_rating", "mean"), mean_quality=("quality", "mean")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    labels = [f"P{r.passage_id} ({r.correct_origin.upper()})" for r in passage.itertuples()]
    bars = ax.bar(labels, passage.accuracy, color=BLUE)
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in passage.accuracy], padding=4)
    ax.axhline(.5, linestyle="--", color=GRAY, label="50% chance")
    ax.set(ylim=(0, 1), ylabel="Correct guesses", title="Descriptive accuracy by passage")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False)
    fig.savefig(out / "passage_accuracy.png", bbox_inches="tight"); plt.close(fig)
    passage.to_csv(out / "passage_summary.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, col in zip(axes, ["accuracy", "mean_source_rating", "confidence"]):
        stats.probplot(people[col], dist="norm", plot=ax)
        ax.set_title(col.replace("_", " ").title())
    fig.suptitle("Q–Q diagnostics: discreteness is expected for six-item accuracy")
    fig.savefig(out / "qq_diagnostics.png", bbox_inches="tight"); plt.close(fig)
    tcrit = stats.t.ppf(.975, len(people)-2)
    regression = pd.DataFrame([dict(
        intercept=fit.intercept, slope=fit.slope,
        slope_ci95_low=fit.slope-tcrit*fit.stderr, slope_ci95_high=fit.slope+tcrit*fit.stderr,
        slope_percentage_points=100*fit.slope, r_squared=fit.rvalue**2, p=fit.pvalue)])
    regression.to_csv(out / "h3_regression.csv", index=False)


# %% 6. Run and export
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "clean")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "h1_h3")
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    response_path = find_input("survey_responses.csv", args.data_dir)
    key_path = find_input("passage_answer_key.csv", args.data_dir)
    long, people, excluded, audit = prepare_data(response_path, key_path)
    main_results = run_tests(people)
    normality, checks = diagnostics(people)
    sensitivity = run_tests(people.loc[~people.flag_low_effort], "flagged responses excluded")
    desc = people[["accuracy", "mean_source_rating", "confidence", "human_guess_rate"]].agg(
        ["count", "mean", "median", "std", "min", "max"]).T
    for name, frame in {"participant_scores": people, "ratings_long": long,
                        "excluded_incomplete": excluded, "hypothesis_tests": main_results,
                        "sensitivity_tests": sensitivity, "descriptive_statistics": desc.reset_index(),
                        "shapiro_diagnostics": normality, "supplementary_tests": checks}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    make_figures(people, long, main_results, out)
    audit["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [response_path, key_path]}
    audit["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scipy": scipy.__version__,
                         "matplotlib": matplotlib.__version__}
    (out / "validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nData audit:\n", json.dumps(audit, indent=2))
    print("\nMain hypothesis tests (raw p-values; compare to alpha):")
    print(main_results[["hypothesis", "n", "estimate", "ci95_low", "ci95_high",
                        "statistic", "p_raw", "alpha", "reject_null"]].to_string(index=False))
    print("\nSupplementary checks:\n", checks.to_string(index=False))
    print(f"\nSaved figures and tables to: {out.resolve()}")
    print("Use README_H1_H3.md for interpretation and report corrections.")


if __name__ == "__main__":
    main()
