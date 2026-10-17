import io
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from safepassage import media
from safepassage.api import app
from safepassage.monitoring import pending_reviews

client = TestClient(app)
FIXTURES = Path(__file__).parent / "fixtures"  # synthetic chat screenshots from make_ocr_fixtures.py
# Two red flags -> fallback score 0.5 -> medium risk -> goes to the review queue.
OCR_TEXT = "Driver jobs in Qatar.\nPay the registration fee today.\nVisa 100% guaranteed."


@pytest.fixture
def fake_tesseract(monkeypatch):
    monkeypatch.setattr(media, "ocr_image", lambda data: OCR_TEXT)


def png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_image_endpoint_returns_ocr_text_and_result(fake_tesseract):
    resp = client.post("/api/v1/scans/image", files={"file": ("chat.png", png(10, 10), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["extracted_text"] == OCR_TEXT and body["source"] == "file"
    assert "Guaranteed visa/job language" in body["indicators"]


def test_ocr_text_is_kept_in_review_queue(fake_tesseract):
    client.post("/api/v1/scans/image", files={"file": ("chat.jpg", png(10, 10), "image/jpeg")})
    (record,) = pending_reviews()
    assert record["media"] == "image" and record["text"] == OCR_TEXT


def test_unsupported_image_type():
    resp = client.post("/api/v1/scans/image", files={"file": ("chat.gif", b"GIF89a", "image/gif")})
    assert resp.status_code == 422


def test_corrupt_image_is_rejected_before_ocr():
    resp = client.post("/api/v1/scans/image", files={"file": ("chat.png", b"not an image", "image/png")})
    assert resp.status_code == 422 and "Could not open image" in resp.json()["detail"]


def test_small_screenshots_are_upscaled_and_greyscaled():
    small, large = media._prepare(png(400, 300)), media._prepare(png(2000, 1000))
    assert small.size == (800, 600) and small.mode == "L"
    assert large.size == (2000, 1000)


requires_tesseract = pytest.mark.skipif(
    not (shutil.which("tesseract") or media._WINDOWS_TESSERACT.exists()), reason="Tesseract not installed")


@requires_tesseract
def test_end_to_end_english_screenshot():
    text = media.ocr_image((FIXTURES / "synthetic_chat_en.png").read_bytes())
    assert "Qatar" in text and "Rs. 150,000" in text and "personal account" in text


@requires_tesseract
def test_end_to_end_sinhala_screenshot():
    if "sin" not in media.ocr_languages():
        pytest.skip("Tesseract Sinhala pack not installed")
    text = media.ocr_image((FIXTURES / "synthetic_chat_si.png").read_bytes())
    assert "පුරප්පාඩු" in text and "150,000" in text
