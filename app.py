"""F1 Reset Test - the web app.

Run:  streamlit run app.py
It only reads the files in outputs/ and data/processed/, so run the pipeline first
(python run_all.py) if those folders are empty.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
RESET_SEASONS = [2014, 2017, 2022, 2026]
MODELS = {"logistic": "Logistic regression", "boosting": "Gradient boosting",
          "baseline": "Simple baseline (no ML)"}
DAYS = {"thursday": "Thursday (history only)", "friday": "Friday (+ practice)",
        "saturday": "Saturday (+ qualifying)"}
BLUE, ORANGE = "#2a78d6", "#eb6834"

st.set_page_config(page_title="F1 Reset Test", page_icon="🏁", layout="wide")


def nice(text: str) -> str:
    """'max-verstappen' -> 'Max Verstappen'"""
    special = {"mclaren": "McLaren", "rb": "RB"}
    return special.get(str(text), str(text).replace("-", " ").title())


@st.cache_data
def load():
    pred = pd.read_csv(OUT / "predictions.csv")
    pred = pred[pred["mode"] == "updating"]          # the model as it would run in real life
    scores = pd.read_csv(OUT / "race_scores.csv")
    cal = pd.read_csv(ROOT / "data" / "processed" / "calendar.csv")
    cal["label"] = "R" + cal["round"].astype(str) + "  " + cal["grand_prix"].map(nice)
    nxt_path = OUT / "next_race.csv"
    nxt = pd.read_csv(nxt_path) if nxt_path.exists() else None
    return pred, scores, cal, nxt


def prob_chart(df: pd.DataFrame, colour_by: str | None = None):
    """Horizontal bars of win probability, best first."""
    df = df.sort_values("p_win", ascending=False).head(12)
    base = alt.Chart(df).encode(
        y=alt.Y("Driver:N", sort=df["Driver"].tolist(), title=None),
        x=alt.X("p_win:Q", title="Chance of winning", axis=alt.Axis(format="%", tickCount=5)),
        tooltip=["Driver", "Team", alt.Tooltip("p_win:Q", title="Win", format=".0%"),
                 alt.Tooltip("p_podium:Q", title="Podium", format=".0%"),
                 alt.Tooltip("p_top10:Q", title="Top 10", format=".0%")],
    )
    if colour_by:
        colour = alt.Color(f"{colour_by}:N", title=None,
                           scale=alt.Scale(domain=["On the podium", "Off the podium"],
                                           range=[ORANGE, BLUE]),
                           legend=alt.Legend(orient="top"))
    else:
        colour = alt.value(BLUE)
    bars = base.mark_bar(size=16, cornerRadiusEnd=4).encode(color=colour)
    labels = base.mark_text(align="left", dx=5).encode(text=alt.Text("p_win:Q", format=".0%"))
    return (bars + labels).properties(height=28 * len(df) + 40)


def pct_table(df, cols):
    out = df[cols].copy()
    for c in ("p_win", "p_podium", "p_top10"):
        out[c] = (100 * out[c]).round(0).astype(int).astype(str) + "%"
    return out.rename(columns={"p_win": "Win", "p_podium": "Podium", "p_top10": "Top 10"})


pred, scores, cal, nxt = load()
st.title("🏁 F1 Reset Test")
st.caption("Can a model that learned F1 under the old rules still predict races after the "
           "rules are rewritten? Every forecast here was made using only earlier races.")

tab_next, tab_race, tab_reset = st.tabs(["Next race", "Race explorer", "The reset test"])

# ----------------------------------------------------------------------------- next race
with tab_next:
    if nxt is None or nxt.empty:
        st.info("The season is over, so there is no next race to forecast.")
    else:
        gp, rnd = nice(nxt["grand_prix"].iloc[0]), int(nxt["round"].iloc[0])
        st.subheader(f"{gp} Grand Prix · round {rnd} of {int(nxt['season'].iloc[0])}")
        st.caption("Thursday forecast: built from form and history only. Practice and "
                   "qualifying times are not in the data until after the weekend.")
        model = st.radio("Model", list(MODELS), format_func=MODELS.get, horizontal=True,
                         key="next_model")
        view = nxt[nxt["model"] == model].copy()
        view["Driver"], view["Team"] = view["driver"].map(nice), view["constructor"].map(nice)
        left, right = st.columns([3, 2])
        left.altair_chart(prob_chart(view), width="stretch")
        right.dataframe(pct_table(view.sort_values("p_win", ascending=False),
                                  ["Driver", "Team", "p_win", "p_podium", "p_top10"]),
                        hide_index=True, width="stretch", height=460)

# ------------------------------------------------------------------------- race explorer
with tab_race:
    c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
    seasons = sorted(pred["season"].unique(), reverse=True)
    season = c1.selectbox("Season", seasons)
    races = cal[(cal["season"] == season) & cal["has_results"]].sort_values("round")
    race_label = c2.selectbox("Race", races["label"], index=len(races) - 1)
    day = c3.selectbox("Forecast made on", list(DAYS), format_func=DAYS.get, index=2)
    model = c4.selectbox("Model", list(MODELS), format_func=MODELS.get)
    race_id = races.loc[races["label"] == race_label, "race_id"].iloc[0]

    view = pred[(pred["race_id"] == race_id) & (pred["stage"] == day)
                & (pred["model"] == model)].copy()
    view["Driver"], view["Team"] = view["driver"].map(nice), view["constructor"].map(nice)
    view["Result"] = view["podium"].map({1: "On the podium", 0: "Off the podium"})
    view["Finished"] = view["finish_pos"].map(lambda p: "DNF" if pd.isna(p) else f"P{int(p)}")
    view["Grid"] = "P" + view["grid"].astype(int).astype(str)

    s = scores[(scores["race_id"] == race_id) & (scores["stage"] == day)
               & (scores["model"] == model) & (scores["mode"] == "updating")].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actual winner", view.loc[view["finish_pos"] == 1, "Driver"].iloc[0])
    m2.metric("Winner picked?", "Yes" if s["hit_win"] == 1 else "No")
    m3.metric("Podium picks right", f"{round(3 * s['hit_podium'])} of 3")
    m4.metric("Top-10 picks right", f"{round(10 * s['hit_top10'])} of 10")
    if season in RESET_SEASONS:
        st.caption(f"{season} was a rule-reset season.")

    left, right = st.columns([3, 2])
    left.altair_chart(prob_chart(view, colour_by="Result"), width="stretch")
    right.dataframe(pct_table(view.sort_values("p_win", ascending=False),
                              ["Driver", "Team", "Grid", "Finished", "p_win", "p_podium", "p_top10"]),
                    hide_index=True, width="stretch", height=460)

# ---------------------------------------------------------------------------- reset test
with tab_reset:
    head = scores[(scores["mode"] == "updating") & (scores["model"] == "logistic")].copy()
    head["reset"] = head["season"].isin(RESET_SEASONS)
    early = head[(head["stage"] == "thursday") & (head["round"] <= 4)]
    sat = head[(head["stage"] == "saturday") & (head["round"] <= 4)]

    def rate(df, reset):
        return f"{100 * df.loc[df['reset'] == reset, 'hit_podium'].mean():.0f}%"

    st.subheader("First four races of a season: podium picks right")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Thursday model, stable rules", rate(early, False))
    k2.metric("Thursday model, after a reset", rate(early, True))
    k3.metric("Saturday model, stable rules", rate(sat, False))
    k4.metric("Saturday model, after a reset", rate(sat, True))
    st.caption(f"Logistic regression, {head['race_id'].nunique()} test races since 2014. "
               "Reset seasons: 2014, 2017, 2022, 2026.")

    for fig, note in [
        ("fig2_reset_recovery.png", "Before qualifying, a reset knocks about 20 points off the "
         "model's podium accuracy in the opening rounds. Once the grid is known, most of the gap closes."),
        ("fig4_features.png", "Why: features built on old results lose value after a reset, "
         "while pace measured this weekend barely changes."),
        ("fig1_day_by_day.png", "Machine learning helps most before qualifying. After "
         "qualifying, plain grid order is very hard to beat on picks."),
        ("fig3_by_season.png", "Season by season, before any car has turned a wheel."),
    ]:
        if (OUT / fig).exists():
            st.image(str(OUT / fig), width=820)
            st.caption(note)
    if (OUT / "findings.md").exists():
        with st.expander("All the tables"):
            st.markdown((OUT / "findings.md").read_text(encoding="utf-8"))
