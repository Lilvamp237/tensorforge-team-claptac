from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from safepassage import api, media
from safepassage.api import app
from safepassage.monitoring import pending_reviews

client = TestClient(app)
CLIP = Path(__file__).parent / "fixtures" / "synthetic_offer_en.wav"  # Windows TTS reading a synthetic offer
# Two red flags -> fallback score 0.5 -> medium risk -> goes to the review queue.
TRANSCRIPT = "Driver jobs in Qatar. Pay the registration fee today. Visa 100% guaranteed."


@pytest.fixture
def fake_whisper(monkeypatch):
    calls = []

    def fake(data, filename, language=None):
        calls.append((filename, language))
        return TRANSCRIPT

    monkeypatch.setattr(media, "transcribe", fake)
    return calls


def test_audio_endpoint_returns_transcript_and_result(fake_whisper):
    resp = client.post("/api/v1/scans/audio?language=ta", files={"file": ("note.ogg", b"fake-audio", "audio/ogg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["extracted_text"] == TRANSCRIPT
    assert body["source"] == "file"
    assert "Guaranteed visa/job language" in body["indicators"]
    assert fake_whisper == [("note.ogg", "ta")]


def test_transcript_is_kept_in_review_queue(fake_whisper):
    client.post("/api/v1/scans/audio", files={"file": ("note.mp3", b"fake-audio", "audio/mpeg")})
    (record,) = pending_reviews()
    assert record["media"] == "audio" and record["text"] == TRANSCRIPT


def test_too_large_upload_is_rejected(fake_whisper, monkeypatch):
    monkeypatch.setattr(api, "MAX_MEDIA_BYTES", 10)
    resp = client.post("/api/v1/scans/audio", files={"file": ("note.wav", b"x" * 11, "audio/wav")})
    assert resp.status_code == 413


def test_unsupported_audio_type():
    resp = client.post("/api/v1/scans/audio", files={"file": ("note.exe", b"x", "application/octet-stream")})
    assert resp.status_code == 422


def test_audio_longer_than_cap_is_rejected(monkeypatch):
    monkeypatch.setattr(media, "MAX_AUDIO_SECONDS", 5)  # the clip is ~20 s; decoding needs PyAV only, not the model
    resp = client.post("/api/v1/scans/audio", files={"file": ("clip.wav", CLIP.read_bytes(), "audio/wav")})
    assert resp.status_code == 422 and "limit" in resp.json()["detail"]


def _whisper_cached() -> bool:
    try:
        from faster_whisper.utils import download_model

        download_model(media.WHISPER_MODEL, local_files_only=True)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _whisper_cached(), reason="Whisper model not downloaded")
def test_end_to_end_transcription():
    text = media.transcribe(CLIP.read_bytes(), CLIP.name, language="en")
    assert "Qatar" in text and "guaranteed" in text.lower()
