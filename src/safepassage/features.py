"""Hand-written red-flag rules, used both as model features and as explanations."""

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
}

_COMPILED = {name: re.compile(p, re.IGNORECASE) for name, p in RED_FLAGS.items()}


def find_red_flags(text: str) -> list[str]:
    return [name for name, rx in _COMPILED.items() if rx.search(text)]
