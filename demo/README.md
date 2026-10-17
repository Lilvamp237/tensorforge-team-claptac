# Demo media

Synthetic files for the live demo. Upload them on the **Check an offer** page (or `POST /api/v1/scans/audio` and `/api/v1/scans/image`). Everything is synthetic (Rule 2.4): the agencies and their 9xxx licence numbers are fictional and never match a real SLBFE licence, phone numbers are placeholders, and the voices are Windows text-to-speech. Regenerate with `python demo/make_demo_media.py` (Windows).

| File | What it is | Expected result |
|---|---|---|
| `media/voice_scam_en.wav` | Voice note: Russia security jobs, "document verification charge" of Rs. 200,000 to a personal account, passport copy on WhatsApp | **HIGH** (~0.99). Flags: personal account, no experience needed, ID documents. "Fee paid into a personal account" is **contradicted** |
| `media/voice_legit_en.wav` | Voice note: licensed-style agency, welders for Qatar, interview at the office, only the SLBFE fee | **LOW** (~0.08). The agency is shown as unverifiable (fictional) |
| `media/chat_scam_en.png` | Chat screenshot: Thailand customer-service job, fee in USDT to a wallet, Telegram | **HIGH** (~0.87). Flags: crypto payment, advance payment |
| `media/chat_scam_si.png` | Sinhala chat screenshot: Russia security jobs, verification fee to a personal account, passport copy | **HIGH** (~0.99), even though Sinhala OCR misreads a few letters |
| `media/chat_legit_en.png` | Poster: agency with licence, interview at the office, SLBFE fee of Rs. 22,027 with a receipt | **LOW** (~0.03). "Fee of Rs. 22,027 as the SLBFE registration fee" is **consistent** |

Suggested order: legit poster (LOW) → Sinhala scam screenshot (HIGH) → scam voice note (HIGH, with the personal-account contradiction). The first voice note downloads the Whisper model (~460 MB), so upload one once before the demo starts.
