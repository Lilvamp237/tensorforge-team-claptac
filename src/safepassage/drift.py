"""Drift signals over the prediction log. Owner: Person A (metrics), Person B (alerting + UI)."""

from .monitoring import read_log


def drift_report(window: int = 200) -> dict:
    """Compare the latest `window` predictions with the training reference."""
    recent = read_log()[-window:]
    if not recent:
        return {"n": 0, "drift": False, "signals": {}}
    mean_conf = sum(max(r["risk"]["probability"], 1 - r["risk"]["probability"]) for r in recent) / len(recent)
    novelty_share = sum(r["novelty"]["status"] == "unfamiliar_pattern" for r in recent) / len(recent)
    # TODO(A): add PSI / embedding-centroid shift vs. training reference; calibrate thresholds on the shifted set
    signals = {"mean_confidence": round(mean_conf, 3), "novelty_share": round(novelty_share, 3)}
    drift = mean_conf < 0.7 or novelty_share > 0.3
    return {"n": len(recent), "drift": drift, "signals": signals}
