"""H4, E1 and post hoc analyses; run from the repo root:
python scripts/analyze_h4_e1_posthoc.py

Inputs: data/clean/survey_responses.csv and passage_answer_key.csv.
Outputs: outputs/h4_e1_posthoc/ (source CSVs are never changed).
Data preparation is shared with analyze_h1_h3.py so both scripts score
responses identically (1 = certainly AI, 10 = certainly human; 1-5 = "AI" guess).

Tests (all within the course toolkit):
- H4: paired t-test, mean quality of AI vs. human passages (Wilcoxon backup).
- E1 (exploratory): Welch's t-test, accuracy of native vs. non-native speakers
  (Levene's test reported, Mann-Whitney U backup).
- Post hoc A: H1 and H2 repeated within each language group (4 tests, Bonferroni).
- Post hoc B: one-way ANOVA of accuracy across English level
  (B2 / C1 / C2 / native) with Tukey's HSD, plus Kruskal-Wallis as a check.
- Post hoc C: passage-level descriptive table (accuracy, source rating, quality).
"""

# %% 1. Setup
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_h1_h3 import ROOT, find_input, prepare_data  # noqa: E402

ALPHA_PRIMARY = 0.05
ALPHA_SECONDARY = 0.05 / 3           # H2, H3, H4 share one Bonferroni family.
ALPHA_GROUP_POSTHOC = 0.05 / 4       # H1 and H2 within each of two groups.
BLUE, ORANGE, GRAY = "#245A81", "#BD591D", "#626870"
LEVELS = ["B2", "C1", "C2", "Native"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140,
                     "savefig.dpi": 200, "axes.titleweight": "bold"})


def english_level(row):
    if row.native_english == "yes":
        return "Native"
    text = str(row.english_proficiency)
    for level in ["B2", "C1", "C2"]:
        if f"({level})" in text:
            return level
    if "near-native" in text.lower():
        return "C2"  # Grouped with the highest non-native level.
    raise ValueError(f"Unrecognized proficiency: {text!r}")


def mean_ci(values, level=0.95):
    values = np.asarray(values, float)
    se = stats.sem(values)
    half = se * stats.t.ppf((1 + level) / 2, len(values) - 1)
    return values.mean(), values.mean() - half, values.mean() + half


# %% 2. H4: perceived quality of AI vs. human passages (paired)
def run_h4(long, people):
    quality = long.pivot_table(index="response_id", columns="correct_origin",
                               values="quality", aggfunc="mean")
    quality = quality.rename(columns={"ai": "quality_ai", "human": "quality_human"})
    quality["quality_diff"] = quality.quality_ai - quality.quality_human
    diff = quality.quality_diff.to_numpy()
    test = stats.ttest_rel(quality.quality_ai, quality.quality_human)
    est, lo, hi = mean_ci(diff)
    dz = diff.mean() / diff.std(ddof=1)
    shapiro = stats.shapiro(diff)
    nonzero = diff[diff != 0]
    wil = stats.wilcoxon(nonzero, zero_method="wilcox", method="approx")
    row = dict(hypothesis="H4", test="Paired t-test (ttest_rel)", n=len(diff),
               mean_quality_ai=quality.quality_ai.mean(),
               mean_quality_human=quality.quality_human.mean(),
               estimate=est, ci95_low=lo, ci95_high=hi, statistic=test.statistic,
               df=len(diff) - 1, p_raw=test.pvalue, alpha=ALPHA_SECONDARY,
               p_bonferroni=min(test.pvalue * 3, 1), effect_type="Cohen d_z",
               effect=dz, reject_null=bool(test.pvalue < ALPHA_SECONDARY),
               shapiro_W=shapiro.statistic, shapiro_p=shapiro.pvalue,
               wilcoxon_stat=wil.statistic, wilcoxon_p=wil.pvalue,
               wilcoxon_note=f"asymptotic, {len(diff) - len(nonzero)} zero differences removed")
    return pd.DataFrame([row]), quality.reset_index()


# %% 3. E1: native vs. non-native accuracy (exploratory)
def run_e1(people):
    rows, desc = [], []
    for measure in ["accuracy", "mean_source_rating", "confidence"]:
        native = people.loc[people.native_english == "yes", measure].to_numpy(float)
        non = people.loc[people.native_english == "no", measure].to_numpy(float)
        welch = stats.ttest_ind(native, non, equal_var=False)
        levene = stats.levene(native, non)
        mwu = stats.mannwhitneyu(native, non, alternative="two-sided")
        v1, v2 = native.var(ddof=1) / len(native), non.var(ddof=1) / len(non)
        se = np.sqrt(v1 + v2)
        dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(native) - 1) + v2 ** 2 / (len(non) - 1))
        diff = native.mean() - non.mean()
        tcrit = stats.t.ppf(.975, dof)
        pooled = np.sqrt(((len(native) - 1) * native.var(ddof=1) +
                          (len(non) - 1) * non.var(ddof=1)) / (len(native) + len(non) - 2))
        rows.append(dict(
            analysis="E1 (exploratory)" if measure == "accuracy" else "E1 supplementary",
            measure=measure, test="Welch's t-test (ttest_ind, equal_var=False)",
            n_native=len(native), n_nonnative=len(non),
            mean_native=native.mean(), mean_nonnative=non.mean(),
            difference=diff, ci95_low=diff - tcrit * se, ci95_high=diff + tcrit * se,
            statistic=welch.statistic, df=dof, p_raw=welch.pvalue, alpha=ALPHA_PRIMARY,
            effect_type="Cohen d (pooled SD)", effect=diff / pooled,
            levene_W=levene.statistic, levene_p=levene.pvalue,
            mannwhitney_U=mwu.statistic, mannwhitney_p=mwu.pvalue))
        for group, values in [("Native", native), ("Non-native", non)]:
            desc.append(dict(measure=measure, group=group, n=len(values), mean=values.mean(),
                             median=np.median(values), sd=values.std(ddof=1),
                             min=values.min(), max=values.max()))
    return pd.DataFrame(rows), pd.DataFrame(desc)


# %% 4. Post hoc A: H1 and H2 within each language group
def run_group_posthoc(people):
    rows = []
    for group, label in [("yes", "Native"), ("no", "Non-native")]:
        sub = people.loc[people.native_english == group]
        for hyp, col, null, alt in [("H1 within group", "accuracy", 0.5, "greater"),
                                    ("H2 within group", "mean_source_rating", 5.5, "two-sided")]:
            values = sub[col].to_numpy(float)
            test = stats.ttest_1samp(values, popmean=null, alternative=alt)
            est, lo, hi = mean_ci(values)
            rows.append(dict(group=label, check=hyp, measure=col, null_value=null,
                             alternative=alt, n=len(values), estimate=est,
                             ci95_low=lo, ci95_high=hi, statistic=test.statistic,
                             df=len(values) - 1, p_raw=test.pvalue,
                             alpha=ALPHA_GROUP_POSTHOC, p_bonferroni=min(4 * test.pvalue, 1),
                             cohen_d=(values.mean() - null) / values.std(ddof=1),
                             reject_null=bool(test.pvalue < ALPHA_GROUP_POSTHOC)))
    return pd.DataFrame(rows)


# %% 5. Post hoc B: accuracy across English level (ANOVA + Tukey's HSD)
def run_level_posthoc(people):
    groups = [people.loc[people.level == lv, "accuracy"].to_numpy(float) for lv in LEVELS]
    anova = stats.f_oneway(*groups)
    kruskal = stats.kruskal(*groups)
    levene = stats.levene(*groups)
    grand = people.accuracy.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((people.accuracy - grand) ** 2).sum()
    omnibus = pd.DataFrame([dict(
        test="One-way ANOVA (f_oneway)", F=anova.statistic,
        df_between=len(groups) - 1, df_within=len(people) - len(groups),
        p=anova.pvalue, eta_squared=ss_between / ss_total,
        levene_W=levene.statistic, levene_p=levene.pvalue,
        kruskal_H=kruskal.statistic, kruskal_p=kruskal.pvalue)])
    tukey = stats.tukey_hsd(*groups)
    ci = tukey.confidence_interval(0.95)
    pairs = []
    for i in range(len(LEVELS)):
        for j in range(i + 1, len(LEVELS)):
            pairs.append(dict(group_a=LEVELS[i], group_b=LEVELS[j],
                              n_a=len(groups[i]), n_b=len(groups[j]),
                              mean_diff=groups[i].mean() - groups[j].mean(),
                              ci95_low=ci.low[i, j], ci95_high=ci.high[i, j],
                              p_tukey=tukey.pvalue[i, j],
                              significant=bool(tukey.pvalue[i, j] < 0.05)))
    level_desc = people.groupby("level").agg(
        n=("accuracy", "size"), mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"), mean_source_rating=("mean_source_rating", "mean"),
        mean_confidence=("confidence", "mean")).reindex(LEVELS).reset_index()
    return omnibus, pd.DataFrame(pairs), level_desc


# %% 6. Post hoc C: passage-level descriptive table
def passage_table(long):
    table = long.groupby(["passage_id", "correct_origin"]).agg(
        n=("correct", "size"), n_correct=("correct", "sum"),
        accuracy=("correct", "mean"), human_guess_rate=("human_guess", "mean"),
        mean_source_rating=("origin_rating", "mean"), mean_quality=("quality", "mean"),
        accuracy_native=("correct", lambda s: s[long.loc[s.index, "native_english"] == "yes"].mean()),
        accuracy_nonnative=("correct", lambda s: s[long.loc[s.index, "native_english"] == "no"].mean()),
    ).reset_index()
    return table


# %% 7. Figures
def make_figures(quality, people, level_desc, out):
    # H4: paired quality, one line per participant.
    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    rng = np.random.default_rng(510)
    jit = rng.uniform(-.04, .04, len(quality))
    for (_, r), j in zip(quality.iterrows(), jit):
        ax.plot([0 + j, 1 + j], [r.quality_human, r.quality_ai], color=GRAY, alpha=.35, lw=1)
    ax.scatter(np.zeros(len(quality)) + jit, quality.quality_human, color=BLUE, s=22, alpha=.7)
    ax.scatter(np.ones(len(quality)) + jit, quality.quality_ai, color=ORANGE, s=22, alpha=.7)
    for x, col, c in [(0, "quality_human", BLUE), (1, "quality_ai", ORANGE)]:
        m, lo, hi = mean_ci(quality[col])
        ax.errorbar(x + .18, m, yerr=[[m - lo], [hi - m]], fmt="D", color=c, capsize=5, ms=8)
    ax.set(xticks=[0, 1], xticklabels=["Human passages", "AI passages"], xlim=(-.4, 1.5),
           ylim=(0.8, 5.2), ylabel="Mean quality rating (1 = awful, 5 = excellent)",
           title="H4: perceived writing quality")
    fig.savefig(out / "h4_quality_paired.png", bbox_inches="tight"); plt.close(fig)

    # E1: accuracy by language group.
    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    for x, (g, label, c) in enumerate([("yes", "Native", BLUE), ("no", "Non-native", ORANGE)]):
        vals = people.loc[people.native_english == g, "accuracy"]
        counts = vals.round(4).value_counts()
        for v, k in counts.items():
            ax.scatter(np.full(k, x) + np.linspace(-.12, .12, k), np.full(k, v),
                       color=c, s=26, alpha=.75)
        m, lo, hi = mean_ci(vals)
        ax.errorbar(x + .28, m, yerr=[[m - lo], [hi - m]], fmt="D", color=c, capsize=5, ms=8)
    ax.axhline(.5, color=GRAY, ls="--", label="Chance (50%)")
    ax.set(xticks=[0, 1], xticklabels=["Native (n = 16)", "Non-native (n = 16)"],
           xlim=(-.5, 1.6), ylim=(-.03, 1.03), ylabel="Accuracy (6 passages)",
           title="E1: accuracy by language group")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False, loc="lower right")
    fig.savefig(out / "e1_accuracy_by_group.png", bbox_inches="tight"); plt.close(fig)

    # Post hoc B: accuracy by English level.
    fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
    for x, lv in enumerate(LEVELS):
        vals = people.loc[people.level == lv, "accuracy"]
        m, lo, hi = mean_ci(vals) if len(vals) > 1 else (vals.mean(), np.nan, np.nan)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=BLUE, capsize=5, ms=8)
        ax.annotate(f"n = {len(vals)}", (x, -.0), ha="center", va="bottom", color=GRAY, fontsize=9)
    ax.axhline(.5, color=GRAY, ls="--", label="Chance (50%)")
    ax.set(xticks=range(len(LEVELS)), xticklabels=LEVELS, ylim=(-.03, 1.03),
           ylabel="Mean accuracy (95% CI)", xlabel="Self-rated English level",
           title="Post hoc: accuracy by English level")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False)
    fig.savefig(out / "posthoc_accuracy_by_level.png", bbox_inches="tight"); plt.close(fig)


# %% 8. Run and export
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "clean")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "h4_e1_posthoc")
    args = parser.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    long, people, excluded, audit = prepare_data(
        find_input("survey_responses.csv", args.data_dir),
        find_input("passage_answer_key.csv", args.data_dir))
    people["level"] = people.apply(english_level, axis=1)

    h4, quality = run_h4(long, people)
    h4_sens, _ = run_h4(long.loc[~long.flag_low_effort], people.loc[~people.flag_low_effort])
    h4_sens.insert(0, "sample", "flagged responses excluded")
    e1, e1_desc = run_e1(people)
    group_posthoc = run_group_posthoc(people)
    omnibus, tukey, level_desc = run_level_posthoc(people)
    passages = passage_table(long)

    tables = {"h4_test": h4, "h4_sensitivity": h4_sens, "h4_participant_quality": quality,
              "e1_tests": e1, "e1_descriptives": e1_desc,
              "posthoc_group_tests": group_posthoc, "posthoc_level_anova": omnibus,
              "posthoc_level_tukey": tukey, "posthoc_level_descriptives": level_desc,
              "posthoc_passage_table": passages}
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    make_figures(quality, people, level_desc, out)
    audit["level_counts"] = people.level.value_counts().reindex(LEVELS).to_dict()
    (out / "validation.json").write_text(json.dumps(audit, indent=2, default=int), encoding="utf-8")

    pd.set_option("display.width", 160)
    print("\nH4:\n", h4[["n", "mean_quality_ai", "mean_quality_human", "estimate", "ci95_low",
                         "ci95_high", "statistic", "p_raw", "alpha", "effect", "shapiro_p",
                         "wilcoxon_p", "reject_null"]].round(4).to_string(index=False))
    print("\nE1:\n", e1[["measure", "mean_native", "mean_nonnative", "difference", "ci95_low",
                         "ci95_high", "statistic", "df", "p_raw", "effect", "levene_p",
                         "mannwhitney_p"]].round(4).to_string(index=False))
    print("\nPost hoc A (alpha = 0.0125):\n", group_posthoc[[
        "group", "check", "estimate", "ci95_low", "ci95_high", "statistic", "p_raw",
        "p_bonferroni", "cohen_d", "reject_null"]].round(4).to_string(index=False))
    print("\nPost hoc B ANOVA:\n", omnibus.round(4).to_string(index=False))
    print("\nTukey HSD:\n", tukey.round(4).to_string(index=False))
    print("\nPassages:\n", passages.round(3).to_string(index=False))
    print(f"\nSaved to {out.resolve()}")


if __name__ == "__main__":
    main()
