"""Step 1 - turn the raw F1DB files into one clean table.

Input : a local copy of the open F1 database (https://github.com/f1db/f1db), folder src/data
Output: data/processed/results.csv   one row per driver per race (finished races)
        data/processed/calendar.csv  one row per race, including races not run yet

Run:  python src/01_build_dataset.py data/f1db/src/data
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

from common import FIRST_SEASON, PROCESSED

Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)  # the C loader is ~10x faster


def read_yaml(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=Loader)


def to_seconds(text):
    """'1:27.470' -> 87.47   (None if there is no time)"""
    if text is None or text == "":
        return None
    parts = str(text).split(":")
    try:
        seconds = float(parts[-1])
        if len(parts) >= 2:
            seconds += 60 * int(parts[-2])
        if len(parts) == 3:
            seconds += 3600 * int(parts[-3])
        return seconds
    except ValueError:
        return None


def best_times(rows, keys):
    """Each driver's fastest lap in a session -> {driverId: seconds}."""
    out = {}
    for row in rows or []:
        times = [to_seconds(row.get(k)) for k in keys]
        times = [t for t in times if t]
        if times:
            out[row["driverId"]] = min(times)
    return out


def gap_pct(times):
    """Turn lap times into '% slower than the fastest car in that session'."""
    if not times:
        return {}
    fastest = min(times.values())
    return {d: 100 * (t / fastest - 1) for d, t in times.items()}


def load_race(race_dir: Path, season: int):
    race = read_yaml(race_dir / "race.yml")
    info = {
        "season": season,
        "round": race["round"],
        "race_id": f"{season}-{race['round']:02d}",
        "date": str(race["date"]),
        "grand_prix": race["grandPrixId"],
        "circuit": race["circuitId"],
        "circuit_type": race.get("circuitType"),       # RACE / ROAD / STREET
        "sprint_weekend": (race_dir / "sprint-race-results.yml").exists(),
    }
    results = read_yaml(race_dir / "race-results.yml")
    info["has_results"] = bool(results)
    if not results:
        return info, []

    # Qualifying: finishing order of the session and each driver's best lap.
    quali = read_yaml(race_dir / "qualifying-results.yml") or []
    quali_pos = {q["driverId"]: q["position"] for q in quali if isinstance(q["position"], int)}
    quali_gap = gap_pct(best_times(quali, ["q1", "q2", "q3", "time"]))

    # Practice: a driver's best gap to the fastest car across FP1-FP3.
    practice_gap = {}
    for n in (1, 2, 3):
        fp = read_yaml(race_dir / f"free-practice-{n}-results.yml")
        for driver, gap in gap_pct(best_times(fp, ["time"])).items():
            practice_gap[driver] = min(gap, practice_gap.get(driver, 99))

    rows = []
    for r in results:
        pos = r["position"]
        finished = isinstance(pos, int)
        grid = r.get("gridPosition")
        rows.append({
            **info,
            "driver": r["driverId"],
            "constructor": r["constructorId"],
            "engine": r.get("engineManufacturerId"),
            "grid": grid if isinstance(grid, int) else None,   # pit-lane starts fixed below
            "quali_pos": quali_pos.get(r["driverId"]),
            "quali_gap_pct": quali_gap.get(r["driverId"]),
            "practice_gap_pct": practice_gap.get(r["driverId"]),
            "finish_pos": pos if finished else None,
            "status": "FIN" if finished else str(pos),          # DNF / DSQ / DNS / ...
            "laps": r.get("laps"),
            "points": r.get("points") or 0,
        })
    return info, rows


def main(f1db_data: Path):
    calendar, rows = [], []
    for season_dir in sorted((f1db_data / "seasons").iterdir()):
        if not season_dir.name.isdigit() or int(season_dir.name) < FIRST_SEASON:
            continue
        for race_dir in sorted((season_dir / "races").iterdir()):
            info, race_rows = load_race(race_dir, int(season_dir.name))
            calendar.append(info)
            rows.extend(race_rows)

    df = pd.DataFrame(rows)
    df = df[~df["status"].isin(["DNS", "DNQ", "DNPQ", "EX"])]          # never took the start
    # Pit-lane starters have no grid slot: put them behind the last car on the grid.
    last = df.groupby("race_id")["grid"].transform("max")
    df["grid"] = df["grid"].fillna(last + 1).astype(int)
    df = df.sort_values(["season", "round", "grid"]).reset_index(drop=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED / "results.csv", index=False)
    pd.DataFrame(calendar).to_csv(PROCESSED / "calendar.csv", index=False)
    print(f"{len(df):,} driver-race rows | {df['race_id'].nunique()} races | "
          f"{df['season'].min()}-{df['season'].max()} | last race: {df['race_id'].iloc[-1]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/01_build_dataset.py <path to f1db/src/data>")
    main(Path(sys.argv[1]))
