import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
KB_DIR = DATA_DIR / "kb"

MODEL_PATH = MODELS_DIR / "baseline.joblib"
REGISTRY_PATH = DATA_DIR / "processed" / "registry.csv"
PREDICTION_LOG = LOGS_DIR / "predictions.jsonl"
REVIEW_QUEUE = LOGS_DIR / "review_queue.jsonl"
ALERT_LOG = LOGS_DIR / "alerts.jsonl"
REVIEWED_LOG = LOGS_DIR / "reviewed.jsonl"
REPORTS_LOG = LOGS_DIR / "reports.jsonl"
REPUTATION_PATH = LOGS_DIR / "reputation.json"
FEEDBACK_PATH = DATA_DIR / "processed" / "feedback.csv"

MODEL_VERSION = "safepassage-0.1.0"
CATEGORIES = ["local_job", "foreign_job", "student_visa", "visa_pr"]

# P(scam) thresholds: >= HIGH is high risk, >= MEDIUM is medium risk and goes to human review.
HIGH_RISK = 0.7
MEDIUM_RISK = 0.4
NOVELTY_THRESHOLD = 0.6

# Drift response: no verdict until this many recent scans; optional webhook for alerts.
DRIFT_MIN_SCANS = 20
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")

# Limits for fetching user-submitted URLs.
FETCH_TIMEOUT_S = 10
FETCH_MAX_BYTES = 2_000_000
FETCH_MAX_REDIRECTS = 3
FETCH_ALLOWED_TYPES = ("text/html", "text/plain", "application/pdf")
