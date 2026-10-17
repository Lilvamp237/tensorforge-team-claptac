"""Build the models on a fresh clone or container if they're missing; otherwise do nothing.

    python scripts/bootstrap.py           # train only what's missing
    python scripts/bootstrap.py --force   # rebuild everything

models/ is gitignored, so every new machine needs this once (about a minute; downloads EMSCAD on first run).
If a step fails, the service still starts: it falls back to the red-flag score until models exist.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REQUIRED = ["baseline.joblib", "category.joblib", "novelty.joblib", "drift_reference.json"]
STEPS = ["make_dataset.py", "train_baseline.py", "calibrate_drift.py"]


def missing() -> list[str]:
    return [f for f in REQUIRED if not (MODELS / f).exists()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    todo = missing()
    if not todo and not args.force:
        print("bootstrap: models present, nothing to do")
        return 0
    print(f"bootstrap: building models (missing: {', '.join(todo) or 'none, --force'})")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for step in STEPS:
        print(f"bootstrap: running {step}", flush=True)
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / step)], cwd=ROOT, env=env)
        if result.returncode != 0:
            print(f"bootstrap: {step} failed (exit {result.returncode}); the service will use the fallback score")
            return 0  # never block the service from starting
    print(f"bootstrap: done, still missing: {missing() or 'nothing'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
