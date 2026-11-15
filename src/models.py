"""The baselines and the two machine-learning models, plus the scoring functions."""
import importlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import TARGETS

STAGES = importlib.import_module("02_features").STAGES   # file name starts with a digit

# What each baseline ranks drivers by (smaller = better). No machine learning involved.
BASELINE_KEY = {
    "thursday": "d_form5",            # "whoever has been finishing best lately"
    "friday": "practice_gap_pct",     # "whoever was fastest in practice"
    "saturday": "grid",               # "whoever starts in front"
}
MODELS = ["baseline", "logistic", "boosting"]


def make_model(kind: str):
    if kind == "logistic":
        # Fill gaps with the median, put features on the same scale, fit a weighted sum.
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    if kind == "boosting":
        # Many small decision trees, each one fixing the mistakes of the ones before.
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                              min_samples_leaf=25, l2_regularization=1.0,
                                              random_state=0)
    raise ValueError(kind)


def race_rank(df: pd.DataFrame, key: str) -> pd.Series:
    """Rank drivers inside each race by `key` (1 = best). Missing values go last."""
    return df.groupby("race_id")[key].rank(method="first", na_option="bottom").astype(int)


def baseline_predict(train, test, stage, target):
    """Probability = how often a driver ranked r-th on the baseline key hit the target before."""
    key = BASELINE_KEY[stage]
    rate = train.groupby(race_rank(train, key))[target].mean()
    return race_rank(test, key).map(rate).fillna(0.0).to_numpy()


def model_predict(train, test, stage, target, kind, weights=None):
    model = make_model(kind)
    cols = STAGES[stage]
    if weights is None:
        model.fit(train[cols], train[target])
    elif kind == "logistic":
        model.fit(train[cols], train[target], logisticregression__sample_weight=weights)
    else:
        model.fit(train[cols], train[target], sample_weight=weights)
    return model.predict_proba(test[cols])[:, 1]


def normalise(test, prob, target):
    """Exactly 1 winner, 3 podium places, 10 top-ten places per race: rescale to match."""
    total = pd.Series(prob).groupby(test["race_id"].to_numpy()).transform("sum").to_numpy()
    return np.clip(prob * TARGETS[target] / np.maximum(total, 1e-9), 1e-4, 1 - 1e-4)


def predict_all(train, test, weights=None):
    """Every stage x model x target for the rows in `test`. Returns a long table."""
    out = []
    for stage in STAGES:
        for kind in MODELS:
            block = test[["season", "round", "race_id", "driver"]].copy()
            block["stage"], block["model"] = stage, kind
            for target in TARGETS:
                if kind == "baseline":
                    p = baseline_predict(train, test, stage, target)
                else:
                    p = model_predict(train, test, stage, target, kind, weights)
                block["p_" + target] = normalise(test, p, target)
            out.append(block)
    return pd.concat(out, ignore_index=True)


def score_races(pred: pd.DataFrame) -> pd.DataFrame:
    """One row per race (per stage/model/mode) with how good the predictions were.

    hit_win     1 if the driver we rated most likely to win did win
    hit_podium  share of our top-3 picks that finished on the podium
    hit_top10   share of our top-10 picks that finished in the top 10
    logloss_*   punishes confident wrong probabilities (lower is better)
    """
    def one_race(g):
        row = {}
        for target, k in TARGETS.items():
            p, y = g["p_" + target].to_numpy(), g[target].to_numpy()
            picks = np.argsort(-p, kind="stable")[:k]
            row["hit_" + target] = y[picks].sum() / k
            row["logloss_" + target] = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return pd.Series(row)

    keys = ["mode", "stage", "model", "season", "round", "race_id"]
    return pred.groupby(keys, sort=False).apply(one_race, include_groups=False).reset_index()
