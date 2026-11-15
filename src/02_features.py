"""Step 2 - build the model inputs (features).

Golden rule: every feature for a race uses ONLY information from before that race
starts. Anything else is "leakage" and would make the model look better than it is.

Features come in three stages, matching what you would know on each day:
  THURSDAY  history only (driver form, team form, reliability, track history)
  FRIDAY    + practice pace
  SATURDAY  + qualifying (grid slot, gap to pole)

Input : data/processed/results.csv, calendar.csv
Output: data/processed/features.csv   (finished races + one row per driver for the next race)

Run:  python src/02_features.py
"""
import numpy as np
import pandas as pd

from common import PROCESSED, TARGETS, lineage

THURSDAY = [
    "d_form5",            # driver: average finishing position, last 5 races
    "d_quali5",           # driver: average qualifying position, last 5 races
    "d_dnf10",            # driver: share of last 10 races not finished
    "d_last_season",      # driver: average finishing position last season
    "d_circuit",          # driver: average finish at this circuit in earlier years
    "d_vs_teammate5",     # driver: qualifying places gained on team-mate, last 5 races
    "d_starts",           # driver: experience (race starts, capped at 100)
    "t_form5",            # team: average finishing position of its cars, last 5 races
    "t_quali5",           # team: average qualifying position of its cars, last 5 races
    "t_dnf10",            # team: share of car-starts not finished, last 10 races
    "t_last_season",      # team: average finishing position last season
    "c_street",           # circuit: 1 if street circuit
    "c_shuffle",          # circuit: average places gained/lost per car in earlier years
    "round",              # how far into the season we are
]
FRIDAY = THURSDAY + ["practice_gap_pct"]              # % slower than the fastest car in practice
SATURDAY = FRIDAY + ["grid", "quali_gap_pct"]         # grid slot and % off the fastest qualifying lap
STAGES = {"thursday": THURSDAY, "friday": FRIDAY, "saturday": SATURDAY}


def add_next_race(results: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Add empty rows for the next race on the calendar so it gets features too."""
    todo = calendar[~calendar["has_results"]].sort_values(["season", "round"])
    if todo.empty:
        return results
    nxt = todo.iloc[0]
    last_race = results[results["race_id"] == results["race_id"].iloc[-1]]
    rows = last_race[["driver", "constructor", "engine"]].copy()   # assume the same line-up
    for col in ["season", "round", "race_id", "date", "grand_prix", "circuit",
                "circuit_type", "sprint_weekend"]:
        rows[col] = nxt[col]
    rows["status"] = "UPCOMING"
    return pd.concat([results, rows], ignore_index=True)


def past_mean(df, keys, col, window=None):
    """Average of `col` over a group's EARLIER rows (never the current one)."""
    def one_group(s):
        s = s.shift(1)                                   # drop the current race
        roll = s.rolling(window, min_periods=1) if window else s.expanding()
        return roll.mean()
    return df.groupby(keys, sort=False)[col].transform(one_group)


def main():
    results = pd.read_csv(PROCESSED / "results.csv")
    calendar = pd.read_csv(PROCESSED / "calendar.csv")
    df = add_next_race(results, calendar)
    df = df.sort_values(["season", "round", "grid"]).reset_index(drop=True)
    upcoming = df["status"] == "UPCOMING"

    # ---- helper columns describing what happened in each race -------------------
    df["team"] = df["constructor"].map(lineage)
    df["starters"] = df.groupby("race_id")["driver"].transform("size")
    df["dnf"] = np.where(upcoming, np.nan, (df["status"] != "FIN").astype(float))
    # A non-finisher counts as finishing last for "form" purposes.
    df["pos_filled"] = np.where(upcoming, np.nan, df["finish_pos"].fillna(df["starters"]))
    df["quali_filled"] = df["quali_pos"].fillna(df["grid"])   # no lap time -> use grid slot

    # ---- driver features -------------------------------------------------------
    df["d_form5"] = past_mean(df, "driver", "pos_filled", 5)
    df["d_quali5"] = past_mean(df, "driver", "quali_filled", 5)
    df["d_dnf10"] = past_mean(df, "driver", "dnf", 10)
    df["d_circuit"] = past_mean(df, ["driver", "circuit"], "pos_filled")
    df["d_starts"] = df.groupby("driver").cumcount().clip(upper=100)

    mates = df.groupby(["race_id", "constructor"])["quali_filled"]
    mate_avg = (mates.transform("sum") - df["quali_filled"]) / (mates.transform("count") - 1)
    df["beat_mate"] = mate_avg - df["quali_filled"]            # + means ahead of team-mate
    df["d_vs_teammate5"] = past_mean(df, "driver", "beat_mate", 5)

    season_avg = df.groupby(["driver", "season"])["pos_filled"].mean().rename("d_last_season")
    season_avg = season_avg.reset_index()
    season_avg["season"] += 1                                  # attach to the FOLLOWING season
    df = df.merge(season_avg, on=["driver", "season"], how="left")

    # ---- team features (one row per team per race, then merged back) -----------
    team = (df.groupby(["team", "season", "round"], sort=False)
              .agg(t_pos=("pos_filled", "mean"), t_quali=("quali_filled", "mean"),
                   t_dnf=("dnf", "mean"))
              .reset_index().sort_values(["season", "round"]))
    team["t_form5"] = past_mean(team, "team", "t_pos", 5)
    team["t_quali5"] = past_mean(team, "team", "t_quali", 5)
    team["t_dnf10"] = past_mean(team, "team", "t_dnf", 10)
    df = df.merge(team[["team", "season", "round", "t_form5", "t_quali5", "t_dnf10"]],
                  on=["team", "season", "round"], how="left")

    team_season = df.groupby(["team", "season"])["pos_filled"].mean().rename("t_last_season")
    team_season = team_season.reset_index()
    team_season["season"] += 1
    df = df.merge(team_season, on=["team", "season"], how="left")

    # ---- circuit features ------------------------------------------------------
    df["c_street"] = (df["circuit_type"] == "STREET").astype(int)
    df["moved"] = (df["grid"] - df["finish_pos"]).abs()        # places gained or lost
    race = (df.groupby(["circuit", "season", "round"], sort=False)["moved"].mean()
              .reset_index().sort_values(["season", "round"]))
    race["c_shuffle"] = past_mean(race, "circuit", "moved")
    df = df.merge(race[["circuit", "season", "round", "c_shuffle"]],
                  on=["circuit", "season", "round"], how="left")

    # ---- what we are trying to predict -----------------------------------------
    for name, cutoff in TARGETS.items():
        df[name] = (df["finish_pos"] <= cutoff).astype(int)    # a DNF counts as 0

    df = df.sort_values(["season", "round", "grid"]).reset_index(drop=True)
    keep = (["season", "race_id", "date", "grand_prix", "circuit", "driver",
             "constructor", "team", "status", "finish_pos", "pos_filled"]
            + SATURDAY + list(TARGETS))
    df[keep].to_csv(PROCESSED / "features.csv", index=False)
    done = df[df["status"] != "UPCOMING"]
    print(f"{len(done):,} rows with results, {len(SATURDAY)} features")
    if (df["status"] == "UPCOMING").any():
        print("next race:", df.loc[df["status"] == "UPCOMING", "race_id"].iloc[0],
              df.loc[df["status"] == "UPCOMING", "grand_prix"].iloc[0])


if __name__ == "__main__":
    main()
