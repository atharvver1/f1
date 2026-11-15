"""Shared settings and helpers used by every script in this project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"

FIRST_SEASON = 2006        # first season we load (knockout qualifying starts here)
FIRST_TRAIN_SEASON = 2008  # first season used to train models (2006-07 only warm up the form features)
FIRST_TEST_SEASON = 2014   # first season we test on

# Seasons where the technical rules were rewritten and the pecking order could reshuffle.
#   2009: aero overhaul + slick tyres      2014: turbo-hybrid engines
#   2017: wider, faster cars               2022: ground-effect cars
#   2026: new chassis AND new engines
RESET_SEASONS = [2009, 2014, 2017, 2022, 2026]

# The three things we predict for every driver in every race.
TARGETS = {"win": 1, "podium": 3, "top10": 10}  # name -> finishing position cut-off

# Teams change names. For "team form" we follow the factory, not the badge.
TEAM_LINEAGE = {
    "toro-rosso": "faenza", "alphatauri": "faenza", "rb": "faenza", "racing-bulls": "faenza",
    "renault": "enstone", "lotus-f1": "enstone", "alpine": "enstone",
    "midland": "silverstone", "spyker": "silverstone", "force-india": "silverstone",
    "racing-point": "silverstone", "aston-martin": "silverstone",
    "bmw-sauber": "hinwil", "sauber": "hinwil", "alfa-romeo": "hinwil",
    "kick-sauber": "hinwil", "audi": "hinwil",
    "honda": "brackley", "brawn": "brackley", "mercedes": "brackley",
    "virgin": "banbury", "marussia": "banbury", "manor": "banbury",
    "lotus-racing": "leafield", "team-lotus": "leafield", "caterham": "leafield",
}


def lineage(constructor_id: str) -> str:
    """Return a stable team id that survives rebrands."""
    return TEAM_LINEAGE.get(constructor_id, constructor_id)
