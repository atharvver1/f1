"""Step 0 - download (or refresh) the raw data.

The data is F1DB, an open F1 database (https://github.com/f1db/f1db, CC BY 4.0).
It is updated after every Grand Prix, so re-run this after a race weekend.
Needs `git` installed. Downloads about 110 MB into data/f1db.

Run:  python src/00_get_data.py
"""
import subprocess

from common import DATA

REPO = "https://github.com/f1db/f1db"
TARGET = DATA / "f1db"


def git(*args):
    subprocess.run(["git", *args], check=True)


def main():
    if (TARGET / ".git").exists():
        git("-C", str(TARGET), "fetch", "--depth", "1", "origin")
        git("-C", str(TARGET), "reset", "--hard", "FETCH_HEAD")
    else:
        DATA.mkdir(exist_ok=True)
        git("clone", "--depth", "1", REPO, str(TARGET))
    git("-C", str(TARGET), "log", "-1", "--format=F1DB version: %cd")


if __name__ == "__main__":
    main()
