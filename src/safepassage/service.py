"""The one shared pipeline. Every adapter (REST, MCP, collector, UI) calls these functions."""

import uuid
from typing import NamedTuple

from . import reputation
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
_VERSION_FILE = MODEL_PATH.parent / "CURRENT_VERSION"


def _version_stamp() -> int | None:
    return _VERSION_FILE.stat().st_mtime_ns if _VERSION_FILE.exists() else None


_loaded_stamp = _version_stamp()


def model_loaded() -> bool:
    return _model is not None


def reload_model() -> bool:
    """Pick up freshly promoted models without restarting the service: the scam model plus the cached novelty
    index, drift reference and category model."""
    global _model, _loaded_stamp
    from . import drift, extract, novelty as novelty_module

    for cached in (novelty_module._index, drift.load_reference, extract._category_model):
        getattr(cached, "cache_clear", lambda: None)()  # tests may swap these for plain functions
    _loaded_stamp = _version_stamp()
    _model = ScamModel() if MODEL_PATH.exists() else None
    return model_loaded()


def scan_text(text: str, source: str = "text", url: str | None = None, media: str | None = None) -> ScanResult:
    """`media` ("audio" / "image") marks text that was transcribed or OCR'd, so analysts know what they read."""
    if _version_stamp() != _loaded_stamp:  # another process (the scheduler) promoted or rolled back a model
        reload_model()
    flags = find_red_flags(text)
    # Until Person A ships a model, fall back to a crude red-flag score so the pipeline runs end to end.
    p = _model.score(text) if _model else min(1.0, 0.2 + 0.15 * len(flags))
    indicators = describe(flags)

    listing = reputation.status(url) if url else None
    if listing == "blocklisted":
        p = max(p, 0.9)
        indicators.insert(0, "Website is on the SafePassage blocklist (analyst-confirmed reports)")
    elif listing == "allowlisted":
        # Informational only: scammers copy trusted sites, so the model's score is not overridden.
        indicators.insert(0, "Website is on the SafePassage verified list")

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
        url=url,
        risk=Risk(label=label, probability=round(p, 4)),
        needs_review=needs_review,
        content=content,
        indicators=indicators,
        verification=verify(text, content),
        novelty=nov,
        campaign=campaign(text),
    )
    record = result.model_dump()
    log_prediction(record)
    if result.needs_review:
        enqueue_review({**record, "text": text[:2000], **({"media": media} if media else {})})
    check_drift()
    return result


def scan_public_url(url: str) -> ScanResult:
    from .fetch import extract_text, safe_fetch

    body, content_type = safe_fetch(url)
    return scan_text(extract_text(body, content_type), source="url", url=url)


def scan_file(data: bytes, filename: str) -> ScanResult:
    from .fetch import extract_text

    content_type = "application/pdf" if filename.lower().endswith(".pdf") else "text/plain"
    # TODO(B, stretch): image OCR
    return scan_text(extract_text(data, content_type), source="file")


class MediaScan(NamedTuple):
    text: str  # what we heard / read, shown to the user above the result
    result: ScanResult


def scan_audio(data: bytes, filename: str, language: str | None = None) -> MediaScan:
    """Voice note -> transcript -> the same pipeline. The audio itself is never stored."""
    from .media import transcribe

    text = transcribe(data, filename, language)
    return MediaScan(text, scan_text(text, source="file", media="audio"))


def scan_image(data: bytes) -> MediaScan:
    """Screenshot or photo -> OCR text -> the same pipeline. The image itself is never stored."""
    from .media import ocr_image

    text = ocr_image(data)
    return MediaScan(text, scan_text(text, source="file", media="image"))
