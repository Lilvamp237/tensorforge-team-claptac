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
# HIGH_RISK 0.65: recall-leaning, inside the team's 0.64-0.7 range (on test: 82% of scams "high", 9% of legit).
# Anything >= 0.4 is flagged either way; HIGH only moves offers between "high" and the review band.
HIGH_RISK = 0.65
MEDIUM_RISK = 0.4
NOVELTY_THRESHOLD = 0.6

# Drift response: no verdict until this many recent scans; optional webhook for alerts.
DRIFT_MIN_SCANS = 20
ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL")

# Official-record trust adjustment (trust.py). Off until demo licence numbers stop clashing with the real registry;
# measured with scripts/evaluate_service.py.
TRUST_ADJUSTMENT = os.environ.get("SAFEPASSAGE_TRUST", "0") == "1"
# Stacking meta-model (stacker.py). Off: it cut missed scams but doubled false alarms on real ads and broke on
# the shifted set (reports/stacker.json).
USE_STACKER = os.environ.get("SAFEPASSAGE_STACKER", "0") == "1"

# Community reports: this many scam reports on one site puts it in the analyst queue (never auto-listed).
REPORT_THRESHOLD = 3

# Limits for fetching user-submitted URLs.
FETCH_TIMEOUT_S = 10
FETCH_MAX_BYTES = 2_000_000
FETCH_MAX_REDIRECTS = 3
FETCH_ALLOWED_TYPES = ("text/html", "text/plain", "application/pdf")
