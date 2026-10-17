"""Turn uploaded media into text for the shared scan pipeline. Owner: Person A.

Nothing here is stored: audio and images are processed in memory and only the extracted text reaches
service.scan_text(), where it is logged like any other scan.

Voice: faster-whisper "small" on CPU (int8). Audio decoding uses PyAV (bundled), so no system ffmpeg is needed.
The model (~460 MB) downloads on first use and is loaded once.

Images: Tesseract (via pytesseract) with whichever of eng / sin / tam are installed. Windows: the UB-Mannheim
installer with Sinhala and Tamil ticked. Images are cleaned up with Pillow first (greyscale, contrast, 2x upscale for
small screenshots), which matters a lot for chat screenshots.
"""

import io
import os
import shutil
from functools import lru_cache
from pathlib import Path

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


# ---------------------------------------------------------------- OCR
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
OCR_LANGUAGES = ("eng", "sin", "tam")
UPSCALE_BELOW_PX = 1600  # screenshots narrower than this are upscaled 2x before OCR
MAX_IMAGE_PIXELS = 40_000_000  # refuse decompression bombs
_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def is_image(filename: str) -> bool:
    return filename.lower().endswith(IMAGE_EXTENSIONS)


@lru_cache(maxsize=1)
def _tesseract() -> tuple[object, str]:
    """pytesseract configured with the binary and the installed subset of eng+sin+tam."""
    import pytesseract

    cmd = os.environ.get("SAFEPASSAGE_TESSERACT") or shutil.which("tesseract")
    if not cmd and _WINDOWS_TESSERACT.exists():
        cmd = str(_WINDOWS_TESSERACT)
    if not cmd:
        raise MediaError("Image reading is unavailable: Tesseract OCR is not installed on the server.")
    pytesseract.pytesseract.tesseract_cmd = cmd
    installed = set(pytesseract.get_languages())
    langs = [lang for lang in OCR_LANGUAGES if lang in installed] or ["eng"]
    return pytesseract, "+".join(langs)


def ocr_languages() -> str:
    return _tesseract()[1]


def _prepare(image_bytes: bytes):
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.width * img.height > MAX_IMAGE_PIXELS:
            raise MediaError("Image is too large.")
        img = ImageOps.exif_transpose(img)
    except MediaError:
        raise
    except Exception as e:  # Pillow raises several error types for corrupt or unsupported files
        raise MediaError(f"Could not open image: {e}") from e
    img = ImageOps.autocontrast(img.convert("L"))
    if img.width < UPSCALE_BELOW_PX:
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS)
    return img


def ocr_image(image_bytes: bytes) -> str:
    """Image (screenshot, photo of a poster or letter) to text."""
    img = _prepare(image_bytes)
    pytesseract, langs = _tesseract()
    text = pytesseract.image_to_string(img, lang=langs, config="--psm 3")
    text = text.replace("‌", "").replace("​", "")  # Tesseract leaves zero-width joiners in Sinhala
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        raise MediaError("No text was found in the image.")
    return text
