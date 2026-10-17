"""Combine the model score with what official records say about the offer. Owner: Person B.

The text model has never seen many real Sri Lankan legitimate ads, so it over-flags licensed agencies whose ads
use "scammy" words (fees, WhatsApp, deadlines). Verification against the real SLBFE registry and job orders is
stronger evidence than wording, so it adjusts the score:
  - any contradicted official-record check raises the score to at least CONTRADICTED_FLOOR;
  - a verified licensed agency lowers it, unless a red flag that no legitimate agency would show is present.
Impersonation is still caught: a licence quoted under another agency's name is a contradicted check.
"""

from .schema import Verification

CONTRADICTED_FLOOR = 0.75
LICENSED_FACTOR = 0.5        # licensed agency verified, licence consistent
LICENSED_FEE_FACTOR = 0.35   # ...and the fee is within that agency's approved job-order cap
# Red flags a licensed agency should never show; with any of these, verification never lowers the score.
SEVERE_FLAGS = {"personal_account", "id_document_request", "task_scam", "no_slbfe", "crypto_payment", "secrecy"}


def adjust(p: float, verification: Verification, flags: list[str]) -> tuple[float, list[str]]:
    """Return the adjusted scam probability and indicators explaining the adjustment."""
    contradicted = [c for c in verification.claims if c.status == "contradicted"]
    if contradicted:
        return max(p, CONTRADICTED_FLOOR), [f"Official records contradict: {c.claim}" for c in contradicted]

    if verification.agency_status == "matched" and not SEVERE_FLAGS.intersection(flags):
        fee_ok = any(c.status == "consistent" and c.claim.lower().startswith("fee") for c in verification.claims)
        factor = LICENSED_FEE_FACTOR if fee_ok else LICENSED_FACTOR
        note = "Licensed SLBFE agency verified" + (" and fee within its approved job-order cap" if fee_ok else "")
        return p * factor, [note]
    return p, []
