"""Step 4 - answer the research questions and draw the charts.

Q1  Do the models beat the simple baselines?
Q2  How much is decided on each day (Thursday -> Friday -> Saturday)?
Q3  Does a model survive a rule reset, and how many races does it take to recover?
Q4  Which features still work after a reset and which collapse?
Q5  Does retraining after every race help?

Input : outputs/race_scores.csv, data/processed/features.csv
Output: outputs/findings.md (all tables) and outputs/fig*.png

Run:  python src/04_analysis.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIRST_TRAIN_SEASON, OUTPUTS, PROCESSED, RESET_SEASONS

HEADLINE = "logistic"                      # the model the headline numbers use
STAGE_NAMES = {"thursday": "Thursday\n(history only)", "friday": "Friday\n(+ practice)",
               "saturday": "Saturday\n(+ qualifying)"}
MODEL_NAMES = {"baseline": "Simple baseline", "logistic": "Logistic regression",
               "boosting": "Gradient boosting"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 30)]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "xtick.major.size": 0,
    "ytick.major.size": 0, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False,
})


def pct(x):
    return f"{100 * x:.0f}%"


def bootstrap_gap(a, b, n=5000, seed=0):
    """90% interval for mean(a) - mean(b), resampling races with replacement."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    gaps = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(gaps, [5, 95])


def bucket(round_):
    for lo, hi in BUCKETS:
        if lo <= round_ <= hi:
            return f"{lo}-{hi}" if hi < 30 else f"{lo}+"


def save(fig, name):
    fig.savefig(OUTPUTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    scores = pd.read_csv(OUTPUTS / "race_scores.csv")
    scores["era"] = np.where(scores["season"].isin(RESET_SEASONS), "reset", "stable")
    scores["rounds"] = scores["round"].map(bucket)
    upd = scores[scores["mode"] == "updating"]
    last = scores.sort_values(["season", "round"])["race_id"].iloc[-1]
    n_races = scores["race_id"].nunique()
    md = [f"# Findings\n\nTest set: {n_races} races, 2014 to {last} "
          f"(round {int(last[-2:])} of the 2026 season). Every prediction was made with a model "
          "that had only seen earlier races.\n"]

    # ---- Q1 + Q2: models vs baselines on each day -------------------------------
    metrics = ["hit_win", "hit_podium", "hit_top10", "logloss_win", "logloss_podium"]
    q1 = upd.groupby(["stage", "model"], sort=False)[metrics].mean()
    table = q1.copy()
    for c in metrics[:3]:
        table[c] = table[c].map(pct)
    table[metrics[3:]] = table[metrics[3:]].round(3)
    table.index = [f"{s.title()} / {MODEL_NAMES[m]}" for s, m in table.index]
    table.columns = ["Winner picked", "Podium picks right", "Top-10 picks right",
                     "Log loss (win)", "Log loss (podium)"]
    md += ["## Q1 + Q2. Models vs baselines, day by day\n",
           "Retrained before every race. Log loss: lower is better.\n",
           table.to_markdown(), ""]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(3)
    for i, (m, colour) in enumerate(zip(MODEL_NAMES, [BLUE, ORANGE, AQUA])):
        vals = [q1.loc[(s, m), "hit_podium"] for s in STAGE_NAMES]
        bars = ax.bar(x + (i - 1) * 0.25, vals, 0.22, color=colour, label=MODEL_NAMES[m])
        ax.bar_label(bars, [pct(v) for v in vals], padding=2, fontsize=9, color=INK)
    ax.set_xticks(x, STAGE_NAMES.values())
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.grid(axis="x", visible=False)
    ax.set_title("How many podium finishers can you name in advance?")
    ax.set_ylabel("Share of top-3 picks that finished on the podium")
    ax.legend(loc="upper left", ncols=3, fontsize=9, handlelength=1)
    save(fig, "fig1_day_by_day.png")

    # ---- Q3: reset seasons vs stable seasons ------------------------------------
    head = upd[upd["model"] == HEADLINE]
    md += ["## Q3. Does the model survive a rule reset?\n",
           f"{MODEL_NAMES[HEADLINE]}, retrained before every race. Reset seasons in the test "
           f"set: {', '.join(str(s) for s in RESET_SEASONS if s >= scores['season'].min())}. "
           "'Gap' is reset minus stable, with a 90% bootstrap interval.\n",
           "| Day | Rounds | Podium picks right: stable | reset | Gap (90% interval) | Races: stable / reset |",
           "|---|---|---|---|---|---|"]
    for stage in STAGE_NAMES:
        for label, mask in [("1-4", head["round"] <= 4), ("5+", head["round"] > 4)]:
            part = head[(head["stage"] == stage) & mask]
            st = part.loc[part["era"] == "stable", "hit_podium"]
            rs = part.loc[part["era"] == "reset", "hit_podium"]
            lo, hi = bootstrap_gap(rs, st)
            md.append(f"| {stage.title()} | {label} | {pct(st.mean())} | {pct(rs.mean())} | "
                      f"{100 * (rs.mean() - st.mean()):+.0f} pts ({100 * lo:+.0f} to {100 * hi:+.0f}) | "
                      f"{len(st)} / {len(rs)} |")
    md.append("")

    order = [bucket(lo) for lo, _ in BUCKETS]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, stage in zip(axes, ["thursday", "saturday"]):
        part = head[head["stage"] == stage]
        curve = part.groupby(["era", "rounds"])["hit_podium"].mean().unstack("era").loc[order]
        for era, colour in [("stable", BLUE), ("reset", ORANGE)]:
            ax.plot(order, curve[era], color=colour, lw=2, marker="o", ms=8,
                    mec=SURFACE, mew=2, label=f"{era.title()} seasons")
            ax.annotate(pct(curve[era].iloc[0]), (0, curve[era].iloc[0]), xytext=(-8, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=9)
        ax.set_title(STAGE_NAMES[stage].replace("\n", " "))
        ax.set_xlabel("Round of the season")
        ax.set_xlim(-0.6, len(order) - 0.6)
        ax.set_ylim(0.3, 0.85)
        ax.yaxis.set_major_formatter(lambda v, _: pct(v))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Podium picks right")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Rule resets hurt the Thursday model all season, and most at the start",
                 x=0.06, ha="left", fontweight="bold", fontsize=12, y=1.02)
    save(fig, "fig2_reset_recovery.png")

    by_season = (upd[upd["stage"] == "thursday"]
                 .pivot_table(index="season", columns="model", values="hit_podium"))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    edge = by_season[HEADLINE] - by_season["baseline"]
    colours = [ORANGE if s in RESET_SEASONS else BLUE for s in by_season.index]
    bars = ax.bar(by_season.index.astype(str), by_season[HEADLINE], 0.6, color=colours)
    ax.bar_label(bars, [pct(v) for v in by_season[HEADLINE]], padding=2, fontsize=9, color=INK)
    ax.set_ylim(0, 0.85)
    ax.yaxis.set_major_formatter(lambda v, _: pct(v))
    ax.grid(axis="x", visible=False)
    ax.set_title("Thursday model, season by season")
    ax.set_ylabel("Podium picks right")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["Stable rules", "Rule-reset season"], loc="upper left", ncols=2,
              fontsize=9, handlelength=1)
    save(fig, "fig3_by_season.png")
    season_tbl = by_season[list(MODEL_NAMES)].map(pct).rename(columns=MODEL_NAMES)
    season_tbl["Logistic minus baseline"] = (100 * edge).map("{:+.0f} pts".format)
    md += ["Thursday podium picks by season (reset seasons: "
           + ", ".join(str(s) for s in RESET_SEASONS if s in by_season.index) + "):\n",
           season_tbl.to_markdown(), ""]

    # ---- Q4: which features survive a reset? -------------------------------------
    feats = pd.read_csv(PROCESSED / "features.csv")
    feats = feats[(feats["status"] != "UPCOMING") & (feats["season"] >= FIRST_TRAIN_SEASON)
                  & (feats["round"] <= 4)]
    names = {"d_last_season": "Driver: last season's average finish",
             "t_last_season": "Team: last season's average finish",
             "d_form5": "Driver: last 5 races", "t_form5": "Team: last 5 races",
             "t_quali5": "Team: qualifying, last 5 races",
             "practice_gap_pct": "Practice pace (this weekend)",
             "quali_gap_pct": "Qualifying pace (this weekend)", "grid": "Grid slot"}

    def link(g):   # rank correlation between each feature and the finishing order, one race
        return g[list(names)].corrwith(g["pos_filled"], method="spearman")

    per_race = feats.groupby(["season", "race_id"]).apply(link, include_groups=False)
    per_race["era"] = np.where(per_race.index.get_level_values("season").isin(RESET_SEASONS),
                               "reset", "stable")
    corr = per_race.groupby("era").mean().T.rename(index=names)
    corr["drop"] = corr["stable"] - corr["reset"]
    corr = corr.sort_values("drop")
    n_era = per_race["era"].value_counts()
    md += ["## Q4. Which features survive a reset?\n",
           f"Rounds 1-4 only, all seasons from {FIRST_TRAIN_SEASON}. Average rank correlation "
           "between the feature and the finishing order (1.0 = feature orders the field "
           f"perfectly, 0 = useless). {n_era['stable']} stable-season races, "
           f"{n_era['reset']} reset-season races.\n",
           corr.round(2).rename(columns={"stable": "Stable seasons", "reset": "Reset seasons",
                                         "drop": "Lost in a reset"}).to_markdown(), ""]

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(corr))
    ax.hlines(y, corr["reset"], corr["stable"], color=GRID, lw=3, zorder=1)
    ax.scatter(corr["stable"], y, s=90, color=BLUE, ec=SURFACE, lw=2, zorder=2, label="Stable seasons")
    ax.scatter(corr["reset"], y, s=90, color=ORANGE, ec=SURFACE, lw=2, zorder=2, label="Rule-reset seasons")
    ax.set_yticks(y, corr.index)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_xlim(0, 0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Link to the finishing order in rounds 1-4 (rank correlation)")
    ax.set_title("After a reset, old form fades. This weekend's pace holds.")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig4_features.png")

    # ---- Q5: does retraining after every race help? ------------------------------
    q5 = (scores[scores["model"] != "baseline"]
          .groupby(["era", "stage", "model", "mode"], sort=False)[["hit_podium", "logloss_podium"]]
          .mean().unstack("mode"))
    q5.columns = [f"{'Podium picks right' if a == 'hit_podium' else 'Log loss (podium)'}: {b}"
                  for a, b in q5.columns]
    for c in q5.columns[:2]:
        q5[c] = q5[c].map(pct)
    q5[q5.columns[2:]] = q5[q5.columns[2:]].round(3)
    q5.index = [f"{e.title()} / {s.title()} / {MODEL_NAMES[m]}" for e, s, m in q5.index]
    md += ["## Q5. Frozen model vs retraining after every race\n", q5.to_markdown(), ""]

    # ---- 2026 race by race --------------------------------------------------------
    this = head[head["season"] == head["season"].max()]
    wide = this.pivot_table(index="round", columns="stage", values=["hit_win", "hit_podium"])
    race_tbl = pd.DataFrame({
        "Thursday: winner": wide[("hit_win", "thursday")].map({1.0: "yes", 0.0: "no"}),
        "Thursday: podium picks right": (wide[("hit_podium", "thursday")] * 3).round().astype(int),
        "Saturday: winner": wide[("hit_win", "saturday")].map({1.0: "yes", 0.0: "no"}),
        "Saturday: podium picks right": (wide[("hit_podium", "saturday")] * 3).round().astype(int),
    })
    md += [f"## {int(head['season'].max())} race by race ({MODEL_NAMES[HEADLINE]})\n",
           race_tbl.to_markdown(), ""]

    (OUTPUTS / "findings.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
