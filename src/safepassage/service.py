"""The one shared pipeline. Every adapter (REST, MCP, collector, UI) calls these functions."""

import uuid

from .config import MODEL_PATH, NOVELTY_THRESHOLD
from .extract import extract_content
from .features import describe, find_red_flags
from .model import ScamModel, risk_label
from .monitoring import enqueue_review, log_prediction
from .novelty import campaign, novelty
from .response import check_drift, drift_active
from .schema import Risk, ScanResult
from .verify import verify

_model: ScamModel | None = ScamModel() if MODEL_PATH.exists() else None


def model_loaded() -> bool:
    return _model is not None


def scan_text(text: str, source: str = "text") -> ScanResult:
    flags = find_red_flags(text)
    # Until Person A ships a model, fall back to a crude red-flag score so the pipeline runs end to end.
    p = _model.score(text) if _model else min(1.0, 0.2 + 0.15 * len(flags))

    content = extract_content(text)
    nov = novelty(text)
    label = risk_label(p)
    needs_review = label == "medium" or nov.score >= NOVELTY_THRESHOLD
    if drift_active():
        # Safe fallback: while the model is drifting, a human checks anything that isn't clearly low risk.
        needs_review = needs_review or label == "high"
    result = ScanResult(
        scan_id=f"scan_{uuid.uuid4().hex[:8]}",
        source=source,
        risk=Risk(label=label, probability=round(p, 4)),
        needs_review=needs_review,
        content=content,
        indicators=describe(flags),
        verification=verify(text, content),
        novelty=nov,
        campaign=campaign(text),
    )
    record = result.model_dump()
    log_prediction(record)
    if result.needs_review:
        enqueue_review({**record, "text": text[:2000]})
    check_drift()
    return result


def scan_public_url(url: str) -> ScanResult:
    from .fetch import extract_text, safe_fetch

    body, content_type = safe_fetch(url)
    return scan_text(extract_text(body, content_type), source="url")


def scan_file(data: bytes, filename: str) -> ScanResult:
    from .fetch import extract_text

    content_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    # TODO(B, stretch): image OCR
    return scan_text(extract_text(data, content_type), source="file")
