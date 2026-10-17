"""Build the live-demo media: synthetic voice notes and chat screenshots of our own synthetic offers.

Everything is synthetic (Rule 2.4): agencies and licence numbers are the fictional demo ones (9xxx, never a real
SLBFE licence), phone numbers are placeholders, and voices are Windows text-to-speech.
Run on Windows: python demo/make_demo_media.py   -> demo/media/  (expected result: every scam file HIGH, every legit file LOW)
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "media"
sys.path.insert(0, str(HERE.parent / "tests" / "fixtures"))
from make_ocr_fixtures import render  # noqa: E402  (chat-style screenshot renderer, Nirmala UI font)

VOICE = {
    "voice_scam_en.wav": (
        "Hello, this is Ruwan from the overseas jobs office. We have security guard jobs in Russia, two thousand "
        "five hundred dollars a month, no experience needed. To confirm your seat, please deposit the document "
        "verification charge of two hundred thousand rupees to my personal account today, and send your passport "
        "copy on WhatsApp."),
    "voice_legit_en.wav": (
        "Kelani Gateway Agency, S L B F E licence number nine three one six. We are recruiting welders for Qatar, "
        "salary one thousand four hundred riyals. Interviews are at our office on Monday. The only payment is the "
        "S L B F E registration fee, paid at the S L B F E office with an official receipt."),
}

CHATS = {
    "chat_scam_si.png": [
        "හිතවත් අයදුම්කරුවනි,",
        "රුසියාවේ ආරක්ෂක නිලධාරී රැකියා, මාසික වැටුප USD 2,500.",
        "පළපුරුද්ද අවශ්‍ය නැත.",
        "ලේඛන සත්‍යාපන ගාස්තුව රු. 200,000 අදම",
        "පෞද්ගලික ගිණුමට බැර කරන්න.",
        "ගමන් බලපත්‍රයේ පිටපත WhatsApp කරන්න 07X XXX XXXX",
    ],
    "chat_scam_en.png": [
        "Dear candidate,",
        "Customer service jobs in Thailand, USD 1,800.",
        "Free air ticket, our team receives you at the airport.",
        "Pay the processing fee in USDT (TRC20)",
        "to the wallet address below for faster visa processing.",
        "Telegram: @newjobs_lk",
    ],
    "chat_legit_en.png": [
        "Kelani Gateway Agency (SLBFE licence 9316)",
        "Welders for Qatar, salary QAR 1,400.",
        "Interviews at our office on Monday.",
        "Only payment: SLBFE registration fee of Rs. 22,027,",
        "paid at the SLBFE office with a receipt.",
        "Shortlisted candidates are contacted on WhatsApp 07X XXX XXXX.",
    ],
}


def speak(text: str, path: Path) -> None:
    """Windows SAPI text-to-speech to a 16 kHz mono WAV (what Whisper expects)."""
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono);"
        f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak($env:DEMO_TEXT); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, env={**__import__("os").environ,
                                                                               "DEMO_TEXT": text})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, lines in CHATS.items():
        render(lines, OUT / name)
        print(f"wrote {OUT / name}")
    for name, text in VOICE.items():
        speak(text, OUT / name)
        print(f"wrote {OUT / name}")
