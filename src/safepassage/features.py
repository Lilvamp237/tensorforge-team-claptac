"""Rule-based red flags, used as model features and as human-readable explanations. Owner: Person A."""

import re

RED_FLAGS = {
    "upfront_fee": r"\b(registration|processing|visa|deposit|advance)\s+(fee|payment|charge)\b|pay (now|first|before)",
    "guaranteed": r"\bguarantee(d)?\b|100\s*%\s*(visa|success|approval)",
    "urgency": r"\b(urgent|limited seats|only \d+ (seats|slots|vacancies)|last chance|today only)\b",
    "personal_account": r"\b(personal account|ez ?cash|send to (my|this) number|reload)\b",
    "no_interview": r"\bno (interview|ielts|experience|qualifications?) (needed|required)\b",
    "visa_conversion": r"\bvisit visa\b.*\b(work|convert)|\bconvert(ed)? to (work|pr)\b",
    "pr_shortcut": r"\b(direct|fast|guaranteed) pr\b|skip (the )?points",
    "contact_offplatform": r"\b(whatsapp|telegram|viber)\b",
    "task_scam": r"\b(like|rate|review|subscribe)\b.{0,30}\b(videos?|products?|hotels?|channels?)\b|prepaid task|daily commission",
    "id_document_request": r"\b(nic|national id|passport|birth certificate)\b.{0,25}\b(copy|copies|photo|send|upload)\b",
}

LABELS = {
    "upfront_fee": "Advance payment requested",
    "guaranteed": "Guaranteed visa/job language",
    "urgency": "Pressure or urgency tactics",
    "personal_account": "Payment to a personal account",
    "no_interview": "No interview or qualifications needed",
    "visa_conversion": "Visit visa promised to convert to work/PR",
    "pr_shortcut": "PR shortcut claims",
    "contact_offplatform": "Personal messaging contact",
    "task_scam": "Online task-job pattern",
    "id_document_request": "Asks for ID documents",
}

_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, p in RED_FLAGS.items()}


def find_red_flags(text: str) -> list[str]:
    return [name for name, rx in _COMPILED.items() if rx.search(text)]


def describe(flags: list[str]) -> list[str]:
    return [LABELS[f] for f in flags]
