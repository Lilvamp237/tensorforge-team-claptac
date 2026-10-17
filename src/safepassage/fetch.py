"""Safe fetching of user-submitted URLs (SSRF-hardened). Owner: Person B.

Page content is untrusted data: it is only ever passed on as text to classify, never as instructions.
"""

import io
import ipaddress
import re
import socket
from html import unescape
from urllib.parse import urljoin, urlparse

import requests
from pypdf import PdfReader

from .config import FETCH_ALLOWED_TYPES, FETCH_MAX_BYTES, FETCH_MAX_REDIRECTS, FETCH_TIMEOUT_S


class UnsafeURL(ValueError):
    pass


def _check_host(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURL("Only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        raise UnsafeURL("Host does not resolve")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise UnsafeURL("URL resolves to a non-public address")


def safe_fetch(url: str) -> tuple[bytes, str]:
    """Fetch a public page, re-validating the host on every redirect. Returns (body, content_type)."""
    for _ in range(FETCH_MAX_REDIRECTS + 1):
        _check_host(url)
        resp = requests.get(url, timeout=FETCH_TIMEOUT_S, allow_redirects=False, stream=True,
                            headers={"User-Agent": "SafePassageLK/0.1 (scam-check)"})
        if resp.is_redirect:
            url = urljoin(url, resp.headers["Location"])
            continue
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
        if content_type not in FETCH_ALLOWED_TYPES:
            raise UnsafeURL(f"Unsupported content type: {content_type}")
        body = b""
        for chunk in resp.iter_content(64_000):
            body += chunk
            if len(body) > FETCH_MAX_BYTES:
                raise UnsafeURL("Response too large")
        return body, content_type
    raise UnsafeURL("Too many redirects")


def extract_text(body: bytes, content_type: str) -> str:
    if content_type == "application/pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages)
    text = body.decode("utf-8", errors="replace")
    if content_type == "text/html":
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
        text = unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()
