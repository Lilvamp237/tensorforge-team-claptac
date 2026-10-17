"""REST adapter. Owner: Person B."""

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl

from . import __version__
from .drift import drift_report
from .fetch import UnsafeURL
from .schema import ScanResult
from .service import model_loaded, scan_file, scan_public_url, scan_text

app = FastAPI(title="SafePassage LK", version=__version__)


class TextScanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class URLScanRequest(BaseModel):
    url: HttpUrl


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded(), "version": __version__}


@app.post("/api/v1/scans/text", response_model=ScanResult)
def scan_text_endpoint(req: TextScanRequest):
    return scan_text(req.text)


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
