"""Rule-based red flags, used as model features and as human-readable explanations. Owner: Person A.

Each flag has patterns in English, Singlish (romanised Sinhala), Sinhala and Tamil script.
Sinhala and Tamil patterns avoid \\b because word boundaries are unreliable around combining vowel signs.
"""

import re

RED_FLAGS = {
    "upfront_fee": [
        r"\b(registration|processing|visa|deposit|advance|placement|reservation)\s+(fee|payment|charge)\b",
        r"\bpay (now|first|before|the deposit)\b",
        r"\b(fee|deposit|advance|salli)\s+(eka|ekak)?\s*(gewanna|gewala|gewwama|dapan|yawanna)\b",
        r"ලියාපදිංචි ගාස්තුව|අත්තිකාරම්|තැන්පතුව|ගාස්තුව.{0,20}ගෙවන්න",
        r"பதிவு கட்டணம்|முன்பணம் செலுத்த|கட்டணம்.{0,20}செலுத்தவும்",
    ],
    "guaranteed": [
        r"(?<!cannot )(?<!not )(?<!no )\bguarantee(d)?\b(?! ekak denna ba)",
        r"100\s*%\s*(visa|success|approval|guarantee)",
        r"සහතිකයි|සහතිකයක් ඇත",
        r"உத்தரவாதம்",
    ],
    "urgency": [
        r"\b(urgent|hurry|limited seats|only \d+ (seats|slots|vacancies)|\d+ slots left|last chance|today only|closes today)\b",
        r"\b(ada (ma|witharai)|ikmanata|hadisi|seats? \d+i? (witharai|thiyenne))\b",
        r"හදිසියි|අද පමණයි|අදම|ස්ථාන \d+ක් පමණයි",
        r"இன்றே|உடனடியாக|இடங்கள் குறைவு|உடனே",
    ],
    "personal_account": [
        r"\b(personal account|ez ?cash|send to (my|this) number|reload)\b",
        r"\bpersonal account ekata\b",
        r"පෞද්ගලික ගිණුම",
        r"தனிப்பட்ட கணக்கு",
    ],
    "no_interview": [
        r"\bno (interview|ielts|experience|qualifications?)( (needed|required))?\b",
        r"\b(interview|ielts|experience) (ekak )?(ona na|nathi|na)\b",
        r"සම්මුඛ පරීක්ෂණයක් අවශ්‍ය නැත|පළපුරුද්ද අවශ්‍ය නැත|IELTS අවශ්‍ය නැත",
        r"நேர்முகத் தேர்வு இல்லை|அனுபவம் தேவையில்லை",
    ],
    "visa_conversion": [
        r"\bvisit visa\b.*\b(work|convert)",
        r"\bconvert(ed)? (it )?to (a )?(work|pr)\b",
        r"\bvisit visa eken\b",
        r"සංචාරක වීසා",
        r"சுற்றுலா விசா",
    ],
    "pr_shortcut": [
        r"\b(direct|fast|guaranteed) pr\b",
        r"skip (the )?points",
        r"without points|points ona na",
        r"සෘජු PR|ලකුණු පද්ධතියකින් තොරව",
    ],
    "no_slbfe": [
        r"no need to register with slbfe",
        r"slbfe register wenna ona na",
        r"SLBFE ලියාපදිංචිය අවශ්‍ය නැත",
    ],
    "contact_offplatform": [r"\b(whatsapp|telegram|viber|imo|tiktok)\b"],
    "task_scam": [
        r"\b(like|rate|review|subscribe)\b.{0,30}\b(videos?|products?|hotels?|channels?)\b",
        r"prepaid task|daily commission|vip level",
        r"\bvideos like karala\b|commission ganna",
        r"වීඩියෝ like|කොමිස්",
    ],
    "id_document_request": [
        r"\b(nic|national id|passport|birth certificate)\b.{0,25}\b(copy|copies|photo|send|upload)\b",
        r"\b(passport|nic)\b.{0,25}\b(copy|whatsapp karanna|yawanna)\b",
        r"ගමන් බලපත්‍ර.{0,30}පිටපත|හැඳුනුම්පතේ පිටපත",
        r"கடவுச்சீட்டு நகல",
    ],
    "crypto_payment": [r"\b(usdt|bitcoin|btc|binance|crypto|trc20|wallet address)\b"],
    "secrecy": [r"\bdo not tell\b|\bdon't tell\b|kiyanna epa|කියන්න එපා"],
}

LABELS = {
    "upfront_fee": "Advance payment requested",
    "guaranteed": "Guaranteed visa/job language",
    "urgency": "Pressure or urgency tactics",
    "personal_account": "Payment to a personal account",
    "no_interview": "No interview or qualifications needed",
    "visa_conversion": "Visit visa promised to convert to work/PR",
    "pr_shortcut": "PR shortcut claims",
    "no_slbfe": "Says SLBFE registration is not needed",
    "contact_offplatform": "Personal messaging contact",
    "task_scam": "Online task-job pattern",
    "id_document_request": "Asks for ID documents",
    "crypto_payment": "Payment in cryptocurrency",
    "secrecy": "Asks you to keep the offer secret",
}

_COMPILED = {name: re.compile("|".join(f"(?:{p})" for p in pats), re.IGNORECASE) for name, pats in RED_FLAGS.items()}


def find_red_flags(text: str) -> list[str]:
    return [name for name, rx in _COMPILED.items() if rx.search(text)]


def describe(flags: list[str]) -> list[str]:
    return [LABELS[f] for f in flags]
