"""REST adapter. Owner: Person B."""

from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl

from . import __version__, reputation, retrain as retrain_loop
from .drift import drift_report
from .fetch import UnsafeURL
from .monitoring import pending_reviews, read_alerts
from .response import trigger_retrain
from .schema import ScanResult
from .service import model_loaded, reload_model, scan_file, scan_public_url, scan_text

app = FastAPI(title="SafePassage LK", version=__version__)


class TextScanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    url: str | None = None  # where the text came from, if known (enables allow/block list checks)


class URLScanRequest(BaseModel):
    url: HttpUrl


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded(), "version": __version__}


@app.post("/api/v1/scans/text", response_model=ScanResult)
def scan_text_endpoint(req: TextScanRequest):
    return scan_text(req.text, url=req.url)


@app.post("/api/v1/scans/url", response_model=ScanResult)
def scan_url_endpoint(req: URLScanRequest):
    try:
        return scan_public_url(str(req.url))
    except UnsafeURL as e:
        raise HTTPException(400, str(e))


@app.post("/api/v1/scans/file", response_model=ScanResult)
async def scan_file_endpoint(file: UploadFile):
    return scan_file(await file.read(), file.filename or "upload.txt")


@app.get("/api/v1/drift")
def drift():
    return drift_report()


class ReviewDecision(BaseModel):
    decision: Literal["scam", "legit", "block", "allow", "dismiss"]
    reviewer: str = "analyst"


@app.get("/api/v1/reviews")
def list_reviews():
    return pending_reviews()


@app.post("/api/v1/reviews/{review_id}")
def decide_review(review_id: str, body: ReviewDecision):
    try:
        return reputation.apply_review(review_id, body.decision, body.reviewer)
    except KeyError:
        raise HTTPException(404, "No pending review with that id")


class UserReport(BaseModel):
    url: str = Field(min_length=3, max_length=2_000)
    verdict: Literal["scam", "legit"]
    comment: str = Field(default="", max_length=500)


@app.post("/api/v1/reports")
def report_site(body: UserReport):
    return reputation.add_report(body.url, body.verdict, body.comment)


@app.get("/api/v1/reputation")
def reputation_lists():
    return reputation.lists()


@app.post("/api/v1/retrain", status_code=202)
def retrain():
    """Start export -> train -> calibrate -> evaluate -> gate -> promote/reject in the background."""
    return trigger_retrain("api")


@app.get("/api/v1/retrain/{job_id}")
def retrain_job(job_id: str):
    job = retrain_loop.job_status(job_id)
    if job is None:
        raise HTTPException(404, "No retrain job with that id")
    return job


@app.get("/api/v1/models")
def list_models():
    return retrain_loop.status()


@app.post("/api/v1/models/rollback")
def rollback_model():
    result = retrain_loop.rollback()
    if not result["rolled_back"]:
        raise HTTPException(409, result["reason"])
    return result


@app.post("/api/v1/model/reload")
def model_reload():
    return {"model_loaded": reload_model()}


@app.get("/api/v1/alerts")
def list_alerts():
    return read_alerts()
