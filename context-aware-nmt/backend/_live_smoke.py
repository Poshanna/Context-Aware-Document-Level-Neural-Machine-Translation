import json
import os
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8765"


def get(path, token=None):
    req = urllib.request.Request(BASE + path)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        r = urllib.request.urlopen(req, timeout=8)
        raw = r.read()
        return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            data = json.loads(body) if body else {}
        except Exception:
            data = {"raw": str(body)[:200]}
        return e.code, data


def post_json(path, data, token=None):
    body = json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        r = urllib.request.urlopen(req, timeout=10)
        raw = r.read()
        return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        b = e.read()
        try:
            d = json.loads(b) if b else {}
        except Exception:
            d = {"raw": str(b)[:200]}
        return e.code, d


def main():
    import jwt  # PyJWT

    SECRET = "uvicorn-smoke-secret-not-for-production-12345"
    token = jwt.encode(
        {"sub": "96f109ab-8765-4321-1234-567890abcdef", "token_type": "access"},
        SECRET,
        algorithm="HS256",
    )

    print("=" * 60)
    print("LIVE UVICORN SMOKE TEST")
    print("=" * 60)

    # 1 & 2 Health
    s1, j1 = get("/health")
    s2, j2 = get("/api/health")
    print(f"[1] GET /health          status={s1} status_field={j1 and j1.get('status')} version={j1 and j1.get('version')}")
    print(f"[2] GET /api/health      status={s2} payload_eq_root={j1 == j2}")

    # 3 Languages
    s3, j3 = get("/api/languages")
    langs = sorted([l["code"] for l in (j3["data"]["languages"] if j3 and j3.get("data") else [])])
    pairs = len(j3["data"]["pairs"]) if j3 and j3.get("data") else 0
    print(f"[3] GET /api/languages   status={s3} langs={langs} pairs={pairs}")

    # 4 /docs
    try:
        r = urllib.request.urlopen(BASE + "/docs", timeout=8)
        ct = r.headers.get("Content-Type", "")
        ok = r.status == 200 and "html" in ct.lower()
        print(f"[4] GET /docs            status={r.status} html={ok}")
    except Exception as e:
        print(f"[4] GET /docs            ERR: {e}")
        ok = False

    # 5 /openapi.json
    s5, j5 = get("/openapi.json")
    paths_count = len(j5.get("paths", {})) if isinstance(j5, dict) else 0
    print(f"[5] GET /openapi.json    status={s5} paths={paths_count}")

    # 6 Valid text translate (with auth)
    s6, j6 = post_json(
        "/api/translate/text",
        {
            "source_language": "en",
            "target_language": "te",
            "text": "Hello world from uvicorn smoke.",
            "translation_mode": "sentence",
            "context_strategy": "none",
            "model": "indictrans2",
        },
        token=token,
    )
    d6 = (j6 or {}).get("data") or {}
    status_val = d6.get("status") if isinstance(d6, dict) else None
    translated = d6.get("translated_text") if isinstance(d6, dict) else "UNKNOWN"
    note = isinstance(d6, dict) and d6.get("note") is not None
    print(f"[6] POST /translate/text (valid auth) status={s6} response_status={status_val} translated_text_is_None={translated is None} has_note={note}")

    # 7 No auth -> 401
    s7, j7 = post_json(
        "/api/translate/text",
        {"source_language": "en", "target_language": "te", "text": "x"},
    )
    e7 = (j7 or {}).get("error") or {}
    err_code = e7.get("code") if isinstance(e7, dict) else None
    print(f"[7] POST /translate/text (no auth)    status={s7} envelope_error_code={err_code}")

    # 8 Empty text -> 400
    s8, j8 = post_json(
        "/api/translate/text",
        {
            "source_language": "en",
            "target_language": "te",
            "text": "",
            "translation_mode": "sentence",
            "context_strategy": "none",
            "model": "indictrans2",
        },
        token=token,
    )
    code8 = None
    err8 = (j8 or {}).get("error")
    if isinstance(err8, dict):
        code8 = err8.get("code")
    else:
        det = (j8 or {}).get("detail")
        if isinstance(det, dict):
            code8 = det.get("code")
    print(f"[8] POST /translate/text (empty text) status={s8} error_code={code8}")

    # 9 History (auth)
    s9, j9 = get("/api/history?page=1&size=20", token=token)
    total = None
    if isinstance(j9, dict) and isinstance(j9.get("data"), dict):
        total = j9["data"].get("total")
    envelope_ok = isinstance(j9, dict) and "data" in j9 and "error" in j9
    print(f"[9] GET /api/history (auth)           status={s9} total={total} envelope_ok={envelope_ok}")

    print()
    print("SMOKE SUMMARY: 1-9 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
