from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

MODEL_PATH = MODELS_DIR / "baseline.joblib"
PREDICTION_LOG = LOGS_DIR / "predictions.jsonl"

LABELS = ["legit", "scam"]
CATEGORIES = ["foreign_job", "student_visa", "visa_pr"]

# Scores between these thresholds are "suspicious" and routed to human review.
SCAM_THRESHOLD = 0.7
REVIEW_THRESHOLD = 0.4
