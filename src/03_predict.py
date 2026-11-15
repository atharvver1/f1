"""Step 3 - make predictions the honest way: always train on the past, test on the future.

Two ways of running each test season (2014-2026):
  frozen    train once on every season before it, then predict the whole season.
            This is the "model built before the rule change" scenario.
  updating  retrain before every race, adding the races already run this season,
            and give recent races more weight.

Also produces a forecast for the next race on the calendar.

Input : data/processed/features.csv
Output: outputs/predictions.csv   one row per driver x race x stage x model x mode
        outputs/race_scores.csv   one row per race x stage x model x mode
        outputs/next_race.csv     forecast for the next race

Run:  python src/03_predict.py        (takes a few minutes)
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import FIRST_TEST_SEASON, FIRST_TRAIN_SEASON, OUTPUTS, PROCESSED, TARGETS
from models import predict_all, score_races

HALF_LIFE = 3.0   # updating mode: a race from 3 seasons ago counts half as much as one today


def recency_weights(train, season, round_):
    age = (season - train["season"]) + (round_ - train["round"]) / 24   # in seasons
    return np.power(0.5, age / HALF_LIFE).to_numpy()


def frozen_season(df, season):
    train = df[df["season"] < season]
    test = df[df["season"] == season]
    return predict_all(train, test).assign(mode="frozen")


def updating_race(df, season, round_):
    before = (df["season"] < season) | ((df["season"] == season) & (df["round"] < round_))
    train = df[before]
    test = df[(df["season"] == season) & (df["round"] == round_)]
    return predict_all(train, test, recency_weights(train, season, round_)).assign(mode="updating")


def main():
    feats = pd.read_csv(PROCESSED / "features.csv")
    feats = feats[feats["season"] >= FIRST_TRAIN_SEASON]
    done = feats[feats["status"] != "UPCOMING"]
    upcoming = feats[feats["status"] == "UPCOMING"]
    seasons = [s for s in sorted(done["season"].unique()) if s >= FIRST_TEST_SEASON]

    jobs = [delayed(frozen_season)(done, s) for s in seasons]
    races = done[done["season"].isin(seasons)][["season", "round"]].drop_duplicates()
    jobs += [delayed(updating_race)(done, s, r) for s, r in races.itertuples(index=False)]
    pred = pd.concat(Parallel(n_jobs=-1, verbose=1)(jobs), ignore_index=True)

    truth = done[["race_id", "driver", "constructor", "grid", "finish_pos", "status", *TARGETS]]
    pred = pred.merge(truth, on=["race_id", "driver"])
    OUTPUTS.mkdir(exist_ok=True)
    pred.round(4).to_csv(OUTPUTS / "predictions.csv", index=False)
    scores = score_races(pred)
    scores.round(4).to_csv(OUTPUTS / "race_scores.csv", index=False)
    print(f"{len(pred):,} predictions for {scores['race_id'].nunique()} races")

    if len(upcoming):
        s, r = int(upcoming["season"].iloc[0]), int(upcoming["round"].iloc[0])
        nxt = predict_all(done, upcoming, recency_weights(done, s, r))
        nxt = nxt[nxt["stage"] == "thursday"]     # no practice or qualifying data yet
        nxt = nxt.merge(upcoming[["driver", "constructor", "grand_prix", "date",
                                  "d_form5", "t_form5"]], on="driver")
        nxt.round(4).to_csv(OUTPUTS / "next_race.csv", index=False)
        print("forecast written for", upcoming["race_id"].iloc[0], upcoming["grand_prix"].iloc[0])


if __name__ == "__main__":
    main()
