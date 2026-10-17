from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import MODEL_PATH
from .features import find_red_flags
from .model import ScamModel, verdict
from .monitoring import log_prediction

app = FastAPI(title="SafePassage LK", version=__version__)

_model: ScamModel | None = ScamModel() if MODEL_PATH.exists() else None


class Offer(BaseModel):
    text: str = Field(min_length=1)
    destination: str | None = None
    offer_type: str | None = None  # foreign_job | student_visa | visa_pr


class Prediction(BaseModel):
    label: str
    scam_score: float
    red_flags: list[str]
    needs_review: bool


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "version": __version__}


@app.post("/predict", response_model=Prediction)
def predict(offer: Offer):
    if _model is None:
        raise HTTPException(503, "Model not trained yet. Run scripts/train_baseline.py")
    score = _model.score(offer.text)
    label = verdict(score)
    result = Prediction(
        label=label,
        scam_score=round(score, 4),
        red_flags=find_red_flags(offer.text),
        needs_review=label == "suspicious",
    )
    log_prediction({**offer.model_dump(), **result.model_dump()})
    return result
