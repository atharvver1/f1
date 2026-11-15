"""Run the whole pipeline in order. Use this after every race weekend.

Run:  python run_all.py            (add --skip-download to reuse the data you already have)
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
STEPS = [
    ["00_get_data.py"],
    ["01_build_dataset.py", str(Path(__file__).parent / "data" / "f1db" / "src" / "data")],
    ["02_features.py"],
    ["03_predict.py"],
    ["04_analysis.py"],
]

for script, *args in STEPS:
    if script.startswith("00") and "--skip-download" in sys.argv:
        continue
    print(f"\n=== {script} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / script), *args], check=True)
print("\nDone. Open the app with:  streamlit run app.py")
