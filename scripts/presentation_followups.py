"""Slide-sized follow-ups. Run: python scripts/presentation_followups.py
Uses original cleaned CSVs; does not change the H1-H3 analysis or inputs.
Requires numpy, pandas, scipy and matplotlib (same as the starter script).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "presentation_followups"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GRAY = "#245A81", "#BD591D", "#626870"
plt.rcParams.update({"font.size": 14, "axes.titleweight": "bold", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})

# 1. Load, validate, and score (do not drop blank demographic fields).
df = pd.read_csv(ROOT / "data" / "clean" / "survey_responses.csv")
keys = list((ROOT / "data").rglob("passage_answer_key.csv"))
if len(keys) != 1:
    raise ValueError("Expected exactly one passage_answer_key.csv under data/.")
key = pd.read_csv(keys[0]).set_index("passage_id").correct_origin.to_dict()
assert key == {1: "human", 2: "ai", 3: "ai", 4: "human", 5: "ai", 6: "human"}
assert df.response_id.notna().all() and df.response_id.is_unique
cols = [f"p{i}_origin_rating" for i in range(1, 7)]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="raise")
    assert (df[col].dropna().between(1, 10) & (df[col].dropna() % 1 == 0)).all()
# Match main analysis complete-case rule for all twelve passage ratings.
all_ratings = cols + [f"p{i}_quality" for i in range(1, 7)]
if df[all_ratings].isna().any().any():
    raise ValueError("Incomplete ratings detected; reconcile exclusions with the main script first.")
long = pd.concat([pd.DataFrame({"response_id": df.response_id, "passage": i,
    "true_source": key[i], "rating": df[f"p{i}_origin_rating"]})
    for i in range(1, 7)], ignore_index=True)
long["guess"] = np.where(long.rating <= 5, "ai", "human")
long["correct"] = long.guess.eq(long.true_source)
long["confidence"] = abs(long.rating - 5.5)


def save(fig, name, caption):
    fig.subplots_adjust(bottom=.2, top=.83, left=.13, right=.96)
    fig.text(.13, .035, caption, fontsize=10, color=GRAY, va="bottom")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# 2. Main-slide candidate: descriptive confusion counts, not an independence test.
counts = pd.crosstab(long.true_source, long.guess).reindex(
    index=["ai", "human"], columns=["ai", "human"], fill_value=0)
counts.to_csv(OUT / "confusion_counts.csv")
proportions = counts.div(counts.sum(axis=1), axis=0)
human_share = long.guess.eq("human").mean()
fig, ax = plt.subplots(figsize=(10.5, 5.9))
y = np.arange(2)
ax.barh(y, proportions.ai, color=ORANGE, label='Guessed "AI"', height=.48)
ax.barh(y, proportions.human, left=proportions.ai, color=BLUE,
        label='Guessed "human"', height=.48)
for i, actual in enumerate(counts.index):
    for source, start in [("ai", 0), ("human", proportions.loc[actual, "ai"])]:
        frac = proportions.loc[actual, source]
        ax.text(start + frac/2, i, f"{frac:.1%}\n{counts.loc[actual, source]} judgments",
                ha="center", va="center", color="white", fontweight="bold")
ax.set(yticks=y, yticklabels=["Actually AI", "Actually human"], xlim=(0,1),
       xlabel="Share of judgments within each true source")
ax.invert_yaxis(); ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.16), ncol=2, frameon=False)
fig.suptitle(f'Participants chose "human" on {human_share:.1%} of judgments', fontsize=19)
save(fig, "01_source_judgments",
     f"Descriptive: {len(df)} people × 6 passages; 3 AI and 3 human passages.\n"
     "These are repeated judgments, not independent participants. H2 uses participant mean ratings.")

# 3. Passage follow-ups: show all six; test family includes all six, not only surprises.
passage_rows = []
for i, group in long.groupby("passage"):
    k, n = int(group.correct.sum()), len(group)
    exact = stats.binomtest(k, n, .5, alternative="two-sided")
    ci = exact.proportion_ci(confidence_level=.95, method="exact")
    chi = stats.chisquare([k, n-k], f_exp=[n/2, n/2])
    passage_rows.append(dict(passage=i, true_source=key[i], n=n, n_correct=k,
        accuracy=k/n, ci95_low=ci.low, ci95_high=ci.high,
        chi2=chi.statistic, chi2_df=1, chi2_p_raw=chi.pvalue,
        chi2_p_bonferroni=min(6*chi.pvalue, 1), exact_p_raw=exact.pvalue,
        exact_p_bonferroni=min(6*exact.pvalue, 1), alpha=.05/6))
passages = pd.DataFrame(passage_rows)
passages.to_csv(OUT / "passage_followups.csv", index=False)
fig, ax = plt.subplots(figsize=(10.5, 6.4))
for row in passages.itertuples():
    ax.errorbar(row.accuracy, row.passage, xerr=[[row.accuracy-row.ci95_low],
        [row.ci95_high-row.accuracy]], fmt="o", markersize=9, capsize=5,
        color=ORANGE if row.true_source == "ai" else BLUE)
    ax.text(1.02, row.passage, f"{row.n_correct}/{row.n} · {row.accuracy:.1%}",
            va="center", transform=ax.get_yaxis_transform(), fontsize=12)
ax.axvline(.5, color=GRAY, linestyle="--", label="50% chance")
ax.set(yticks=passages.passage,
       yticklabels=[f"P{r.passage} · {r.true_source.upper()}" for r in passages.itertuples()],
       xlim=(0,1), ylim=(.5,6.5), xlabel="Correct source identification")
ax.invert_yaxis(); ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.legend(frameon=False, loc="upper left")
p3 = passages.loc[passages.passage.eq(3)].iloc[0]
fig.suptitle(f"AI passage 3 fooled {int(p3.n-p3.n_correct)} of {int(p3.n)} participants", fontsize=19)
save(fig, "02_passage_accuracy",
     "Post hoc: all six passages shown. Whiskers are individual 95% exact binomial intervals,\n"
     "not multiplicity-adjusted intervals; test decisions use Bonferroni alpha = .05/6.")

# 4. Optional H3 illustration: judgment-level confidence; no new independent-trial p-value.
confidence = long.groupby("confidence").agg(n_judgments=("correct", "size"),
    n_correct=("correct", "sum"), n_people=("response_id", "nunique"),
    accuracy=("correct", "mean")).reset_index()
confidence.to_csv(OUT / "confidence_descriptive.csv", index=False)
fig, ax = plt.subplots(figsize=(10.5, 5.9))
ax.bar(confidence.confidence, confidence.accuracy, width=.65, color=BLUE)
for row in confidence.itertuples():
    ax.text(row.confidence, row.accuracy+.025, f"{row.accuracy:.1%}\n{row.n_correct}/{row.n_judgments}",
            ha="center", fontsize=12)
ax.axhline(.5, color=GRAY, linestyle="--", label="50% chance")
ax.set(ylim=(0,1), xticks=confidence.confidence, xlabel="Distance from 5.5 (confidence proxy)",
       ylabel="Correct judgments")
ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False, loc="upper right")
extreme = confidence.loc[confidence.confidence.eq(4.5)].iloc[0]
fig.suptitle(f'The most certain judgments were {extreme.accuracy:.0%} correct', fontsize=19)
save(fig, "03_confidence_judgments",
     f"Descriptive follow-up; endpoint ratings 1 or 10: {int(extreme.n_judgments)} judgments from "
     f"{int(extreme.n_people)} people.\n"
     "Repeated judgments; this is separate from H3's participant-level correlation, not a calibration curve.")


# 5. Power under assumed effect sizes, not observed/post-hoc effect-size power.
def t_power(n, d, alpha=.05, two_sided=False):
    dfree = np.asarray(n)-1
    critical = stats.t.ppf(1-alpha/2 if two_sided else 1-alpha, dfree)
    noncentrality = d*np.sqrt(n)
    power = stats.nct.sf(critical, dfree, noncentrality)
    return power + (stats.nct.cdf(-critical, dfree, noncentrality) if two_sided else 0)


power_rows = []
for label, alpha, two_sided in [("H1", .05, False), ("H2/H4", .05/3, True)]:
    for d in [.2,.5]:
        candidates = np.arange(3,1001)
        smallest = int(candidates[np.flatnonzero(t_power(candidates,d,alpha,two_sided)>=.8)[0]])
        power_rows.append(dict(test=label, alpha=alpha, assumed_d=d, n=len(df),
            power=t_power(len(df),d,alpha,two_sided), n_for_80pct=smallest))
pd.DataFrame(power_rows).to_csv(OUT / "power_scenarios.csv", index=False)
fig, ax = plt.subplots(figsize=(10.5, 5.9))
sample_sizes = np.arange(10,201)
for d, color, linestyle in [(.2, ORANGE, "--"), (.5, BLUE, "-")]:
    power = t_power(sample_sizes, d)
    actual = t_power(len(df), d)
    ax.plot(sample_sizes, power, color=color, linestyle=linestyle, linewidth=2.5,
            label=f"Assumed d = {d} ({'small' if d==.2 else 'medium'})")
    ax.scatter(len(df), actual, color=color, s=60, zorder=3)
    ax.annotate(f"{actual:.0%} power at n = {len(df)}", (len(df), actual),
                xytext=(12, 12), textcoords="offset points", color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 2})
ax.axhline(.8, color=GRAY, linestyle=":", label="80% power target")
ax.axvline(len(df), color=GRAY, linestyle=":")
ax.set(xlim=(10,200), ylim=(0,1.04), xlabel="Participants", ylabel="Power")
ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(frameon=False, loc="lower right")
fig.suptitle("A small detection advantage could easily be missed", fontsize=19)
save(fig, "04_power_h1",
     "H1: one-sided one-sample t-test, alpha = .05. d = (mean accuracy − .5) / participant SD.\n"
     "Normal-model planning calculation at specified effect sizes; not power estimated from the observed d.")

print("Descriptive confusion counts:\n",counts)
print("\nPassage follow-ups:\n",passages.to_string(index=False))
print("\nConfidence groups:\n",confidence.to_string(index=False))
print("\nPower scenarios:\n",pd.DataFrame(power_rows).to_string(index=False))
print(f"\nFigures and tables saved to {OUT}")
