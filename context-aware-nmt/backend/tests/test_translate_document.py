"""Tests for POST /api/translate/document.

Foundation phase behaviour:
  * Supported file (TXT/PDF/DOCX, small) -> validated / not_implemented job.
  * Validation errors (missing, unknown ext, oversized, empty) -> clear
    envelope.error with 4xx status codes.
  * Authenticated surface, never fabricates translated output bytes.
"""

from __future__ import annotations

import io
import os

from fastapi.testclient import TestClient


_ENDPOINT = "/api/translate/document"
_VALID_FORM = dict(
    source_lang="te",
    target_lang="en",
    pipeline="baseline",
    model="nllb200",
)


def _multipart(filename: str, content: bytes, content_type: str | None = None) -> dict:
    return {
        "file": (filename, io.BytesIO(content), content_type) if content_type is not None
        else (filename, io.BytesIO(content)),
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_txt_file_accepted(client: TestClient, auth_headers: dict) -> None:
    data = _VALID_FORM.copy()
    content = b"Namaste!\nThis is a simple test.\nBye."
    files = _multipart("hello.txt", content)
    response = client.post(_ENDPOINT, data=data, files=files, headers=auth_headers)
    assert response.status_code in (200, 201), response.text
    env = response.json()
    assert env["error"] is None
    job = env["data"]
    assert job["status"] in {"validated", "not_implemented"}
    assert job["source_lang"] == "te"
    assert job["target_lang"] == "en"
    assert job["pipeline"] == "baseline"
    assert job["model_name"] == "nllb200"
    fi = job["file_info"]
    assert fi["extension"] == "txt"
    assert fi["size_bytes"] == len(content)
    assert len(fi["sha256"]) == 64
    assert fi["filename"] == "hello.txt"
    assert job["progress_percent"] == 0


def test_pdf_docx_treated_as_valid_extensions(client: TestClient, auth_headers: dict) -> None:
    for fname in ("doc.pdf", "report.docx"):
        data = _VALID_FORM.copy()
        files = _multipart(fname, b"%PDF-1.4 fake content" if fname.endswith(".pdf") else b"PK\x03\x04 fake docx")
        response = client.post(_ENDPOINT, data=data, files=files, headers=auth_headers)
        assert response.status_code == 200, (fname, response.text)
        job = response.json()["data"]
        assert job["file_info"]["extension"] in {"pdf", "docx"}
        assert job["status"] in {"validated", "not_implemented"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_document_endpoint_requires_auth(client: TestClient) -> None:
    files = _multipart("a.txt", b"hi")
    response = client.post(_ENDPOINT, data=_VALID_FORM, files=files)
    assert response.status_code == 401
    assert response.json().get("error") is not None


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_rejects_missing_file(client: TestClient, auth_headers: dict) -> None:
    # FastAPI raises a 422 validation error when the file Form field is absent.
    response = client.post(_ENDPOINT, data=_VALID_FORM, headers=auth_headers)
    assert response.status_code == 422
    assert response.json().get("error") is not None


def test_rejects_unsupported_file_extension(client: TestClient, auth_headers: dict) -> None:
    files = _multipart("image.png", b"\x89PNG\r\n\x1a\n" + os.urandom(16))
    response = client.post(_ENDPOINT, data=_VALID_FORM, files=files, headers=auth_headers)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "unsupported_file_type"


def test_rejects_file_with_no_extension(client: TestClient, auth_headers: dict) -> None:
    files = _multipart("Makefile", b"all: build\n")
    response = client.post(_ENDPOINT, data=_VALID_FORM, files=files, headers=auth_headers)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "missing_extension"


def test_rejects_empty_zero_byte_file(client: TestClient, auth_headers: dict) -> None:
    files = _multipart("empty.txt", b"")
    response = client.post(_ENDPOINT, data=_VALID_FORM, files=files, headers=auth_headers)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "missing_file"


def test_rejects_oversized_file(client: TestClient, auth_headers: dict) -> None:
    # MAX_UPLOAD_SIZE_MB defaults to 10, so 100 KB isn't enough. We set
    # a smaller effective limit by uploading 10 MB of nonsense. But 10 MB
    # of bytes in memory is wasteful — instead, use 12 MB of repeated
    # chars.
    payload = ("a" * 1024 * 1024 * 12).encode("ascii")
    files = _multipart("huge.txt", payload)
    response = client.post(_ENDPOINT, data=_VALID_FORM, files=files, headers=auth_headers)
    assert response.status_code == 413, response.text
    body = response.json()
    assert body["error"]["code"] == "oversized_file"


def test_rejects_invalid_language_on_form(client: TestClient, auth_headers: dict) -> None:
    data = {**_VALID_FORM, "source_lang": "xyz"}
    files = _multipart("a.txt", b"hi")
    response = client.post(_ENDPOINT, data=data, files=files, headers=auth_headers)
    assert response.status_code == 422


def test_document_submit_adds_history_entry(
    client: TestClient, auth_headers: dict
) -> None:
    files = _multipart("note.txt", b"hello translate document")
    tx = client.post(_ENDPOINT, data=_VALID_FORM, files=files, headers=auth_headers)
    assert tx.status_code == 200
    job_id = tx.json()["data"]["job_id"]
    hist = client.get("/api/history", headers=auth_headers)
    items = hist.json()["data"]["items"]
    assert any(h["job_id"] == job_id for h in items)
