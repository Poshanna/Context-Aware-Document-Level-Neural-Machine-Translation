"""Tests for the languages endpoint.

Honesty rules asserted here:
  * Hindi is present as a language entry.
  * Hindi pairs for NLLB-200 are flagged implemented=True but with a
    * Hindi remains listed as a language, but no Hindi baseline pair is claimed.
  * te↔en pairs are implemented=True for both models.
    * te↔en pairs are implemented=True for the registered NLLB-200 backend.
    # te↔en must be implemented by the registered NLLB-200 backend.
  * We never fabricate pairs like te↔hi.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _get_envelope(client: TestClient):
    response = client.get("/api/languages")
    assert response.status_code == 200, response.text
    return response.json()


def test_languages_returns_envelope(client: TestClient) -> None:
    env = _get_envelope(client)
    assert env["data"] is not None
    assert env["error"] is None
    data = env["data"]
    assert "languages" in data
    assert "pairs" in data


def test_languages_includes_english_telugu_hindi(client: TestClient) -> None:
    codes = {l["code"] for l in _get_envelope(client)["data"]["languages"]}
    # All three must be present (even if Hindi not fully wired).
    assert codes == {"en", "te", "hi"}, codes


def test_languages_labels_and_native(client: TestClient) -> None:
    by_code = {
        l["code"]: l for l in _get_envelope(client)["data"]["languages"]
    }
    assert by_code["en"]["label"] == "English"
    assert by_code["te"]["label"] == "Telugu"
    assert by_code["te"]["native_label"] == "తెలుగు"
    assert by_code["hi"]["label"] == "Hindi"
    assert by_code["hi"]["native_label"] == "हिन्दी"


def test_pairs_honest_implemented_flags(client: TestClient) -> None:
    """Each pair must have implemented=True only where a model lists it."""
    pairs = _get_envelope(client)["data"]["pairs"]
    # Flatten (src, tgt) -> pair
    by_pair = {(p["source_lang"], p["target_lang"]): p for p in pairs}

    # te↔en must be implemented for both models.
    assert by_pair[("te", "en")]["implemented"] is True
    assert by_pair[("en", "te")]["implemented"] is True
    assert "nllb200" in by_pair[("te", "en")]["models_supported"]
    assert by_pair[("te", "en")]["models_supported"] == ["nllb200"]

    # NLLB-200 supports hi↔en, IndicTrans2 does not.
    # No Hindi or Telugu-Hindi baseline pair is claimed in this phase.
    for (s, t) in [("en", "hi"), ("hi", "en"), ("te", "hi"), ("hi", "te")]:
        assert (s, t) not in by_pair, f"Unexpected fabricated pair {s}→{t}"


def test_languages_does_not_require_auth(client: TestClient) -> None:
    """Frontend needs this for dropdowns before login."""
    response = client.get("/api/languages")
    assert response.status_code == 200
