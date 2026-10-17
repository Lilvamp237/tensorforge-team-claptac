"""Render synthetic offers (from our own generator's phrasing) as chat-style screenshots for the OCR tests.

No real chats or people: every line is synthetic and phone numbers are placeholders (Rule 2.4).
Needs a font with Sinhala and Tamil glyphs (Windows: Nirmala UI) and Pillow with raqm for correct shaping.
Run: python tests/fixtures/make_ocr_fixtures.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
FONT = Path(r"C:\Windows\Fonts\Nirmala.ttc")

OFFERS = {
    "synthetic_chat_en.png": [
        "URGENT!!! Driver jobs in Qatar,",
        "salary Rs. 600,000 per month.",
        "No interview needed.",
        "Pay the registration fee of Rs. 150,000",
        "today to the personal account below.",
        "Visa 100% guaranteed.",
        "WhatsApp 07X XXX XXXX",
    ],
    "synthetic_chat_si.png": [
        "හදිසියි!!! කටාර් සඳහා රියදුරු පුරප්පාඩු,",
        "මාසික වැටුප රු. 450,000.",
        "ලියාපදිංචි ගාස්තුව රු. 150,000 අදම ගෙවන්න.",
        "100% වීසා සහතිකයි.",
        "WhatsApp 07X XXX XXXX",
    ],
}


def render(lines: list[str], path: Path, size: int = 22) -> None:
    font = ImageFont.truetype(str(FONT), size)
    pad, gap = 24, int(size * 0.7)
    width = max(int(font.getlength(line)) for line in lines) + 2 * pad
    height = len(lines) * (size + gap) + 2 * pad
    img = Image.new("RGB", (width + 40, height + 40), (229, 221, 213))  # chat-app style background
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([20, 20, 20 + width, 20 + height], radius=14, fill=(255, 255, 255))
    for i, line in enumerate(lines):
        draw.text((20 + pad, 20 + pad + i * (size + gap)), line, font=font, fill=(17, 17, 17))
    img.save(path)


if __name__ == "__main__":
    for name, lines in OFFERS.items():
        render(lines, HERE / name)
        print(f"wrote {HERE / name}")
