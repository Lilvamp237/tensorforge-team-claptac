"""Turn uploaded media into text for the shared scan pipeline. Owner: Person A.

Nothing here is stored: audio and images are processed in memory and only the extracted text reaches
service.scan_text(), where it is logged like any other scan.

Voice: faster-whisper "small" on CPU (int8). Audio decoding uses PyAV (bundled), so no system ffmpeg is needed.
The model (~460 MB) downloads on first use and is loaded once.
"""

import io
import os
from functools import lru_cache

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".ogg")
MAX_AUDIO_SECONDS = 180  # longer clips are rejected so one request can't tie up the API
WHISPER_MODEL = os.environ.get("SAFEPASSAGE_WHISPER_MODEL", "small")
SAMPLE_RATE = 16_000


class MediaError(ValueError):
    """The upload can't be turned into text (wrong type, too long, nothing heard or read)."""


def is_audio(filename: str) -> bool:
    return filename.lower().endswith(AUDIO_EXTENSIONS)


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe(audio_bytes: bytes, filename: str, language: str | None = None) -> str:
    """Speech to text. `language` is an optional ISO hint ("en", "ta", "si"); otherwise it is auto-detected.

    Whisper is reasonable on English and Tamil but weak on Sinhala.
    """
    if not is_audio(filename):
        raise MediaError(f"Unsupported audio type: {filename}. Use {', '.join(AUDIO_EXTENSIONS)}.")
    from faster_whisper.audio import decode_audio

    try:
        samples = decode_audio(io.BytesIO(audio_bytes), sampling_rate=SAMPLE_RATE)
    except Exception as e:  # PyAV raises several error types for corrupt or unsupported files
        raise MediaError(f"Could not decode audio: {e}") from e
    seconds = len(samples) / SAMPLE_RATE
    if seconds > MAX_AUDIO_SECONDS:
        raise MediaError(f"Audio is {seconds:.0f} s long; the limit is {MAX_AUDIO_SECONDS} s.")

    segments, _info = _whisper().transcribe(samples, language=language or None, beam_size=5, vad_filter=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    if not text:
        raise MediaError("No speech was detected in the audio.")
    return text
