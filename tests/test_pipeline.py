"""Offline guard tests — no Sarvam key or network required.

These do not call Sarvam. They lock the parts a model cannot be trusted with:
retrieval ranking, the not_found gate, citation stripping, empty/invalid
payloads, missing-key and malformed-JSON error mapping.
"""

Run from the server directory:
  python ../tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `server/` importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server"
sys.path.insert(0, str(SERVER))
os.chdir(SERVER)

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import main  # noqa: E402
import retrieve  # noqa: E402
from llm_sarvam import BadModelPayload, MissingApiKey, extract_json  # noqa: E402
from prompt_policy import REFUSAL_TEXT  # noqa: E402

client = TestClient(main.app)
passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  OK  {name}")
    else:
        failed += 1
        print(f" FAIL {name} — {detail}")


# --------------------------------------------------------------------------- #
print("\n[1] Retrieval ranks the right section")
cases = [
    ("How do I reset my password?", "Password Reset"),
    ("Does the product support SMS authentication?", "Two-Factor Authentication"),
    ("Will I get a refund if I cancel?", "Subscription Cancellation"),
]
for q, expect in cases:
    hits, score = retrieve.find_relevant(q)
    titles = [d.title for d, _ in hits]
    check(f"retrieve:{expect}", hits and titles[0] == expect, f"got {titles} score={score}")

print("\n[2] Unsupported / export never clears the gate")
hits, score = retrieve.find_relevant("How do I export my account data?")
check("export→empty hits", hits == [], f"hits={hits} score={score}")

print("\n[3] JSON extraction tolerates fences + preamble")
messy = 'Sure!\n```json\n{"answer":"x","status":"answered","sources":["Password Reset"],"confidence":0.9}\n```\n'
extracted = extract_json(messy)
check("extract_json fence", '"answer":"x"' in extracted.replace(" ", ""))


class Tiny(BaseModel):
    answer: str
    status: str
    sources: list[str]
    confidence: float


check("extract parses", Tiny.model_validate_json(extracted).status == "answered")

print("\n[4] Unsupported question never calls the model")


def boom(_messages):
    raise AssertionError("LLM must not be called when retrieval fails")


main.call_model = boom  # type: ignore[assignment]
r = client.post("/api/chat", json={"message": "How do I export my account data?", "conversation": []})
check("status not_found", r.status_code == 200 and r.json()["status"] == "not_found")
check("sources empty", r.json()["sources"] == [])
check("refusal text", r.json()["answer"] == REFUSAL_TEXT)

print("\n[5] Empty message → 400")
r = client.post("/api/chat", json={"message": "   ", "conversation": []})
check("empty→400", r.status_code == 400)
check("empty error shape", "error" in r.json())

print("\n[6] Extra fields rejected")
r = client.post("/api/chat", json={"message": "hi", "conversation": [], "hack": True})
check("extra forbid", r.status_code == 400)

print("\n[7] Citation cross-check drops unknown sources")


class FakeAnswer:
    answer = "Export from Settings > Data."
    status = "answered"
    sources = ["Data Export"]  # never retrieved
    confidence = 0.99


main.call_model = lambda messages: (FakeAnswer(), {"input_tokens": 10, "output_tokens": 5})  # type: ignore
# Force a retrieval hit by asking a real KB question but returning a fake citation.
r = client.post("/api/chat", json={"message": "How do I reset my password?", "conversation": []})
body = r.json()
check("fake cite→not_found", body.get("status") == "not_found", str(body))

print("\n[8] Valid grounded answer path")


class GoodAnswer:
    answer = "Select Forgot Password on the login page. The link is valid for 30 minutes."
    status = "answered"
    sources = ["Password Reset"]
    confidence = 0.9


main._cache.clear()
main.call_model = lambda messages: (GoodAnswer(), {"input_tokens": 20, "output_tokens": 15})  # type: ignore
r = client.post("/api/chat", json={"message": "How do I reset my password?", "conversation": []})
body = r.json()
check("answered", body["status"] == "answered", str(body))
check("source ok", body["sources"] == ["Password Reset"])
check("meta latency", "latency_ms" in body.get("meta", {}))

print("\n[9] Missing credentials mapped safely")


def missing(_m):
    raise MissingApiKey("no key")


main.call_model = missing  # type: ignore
# Clear cache so we actually hit the model path.
main._cache.clear()
r = client.post(
    "/api/chat",
    json={"message": "Does SMS authentication work?", "conversation": []},
)
check("missing key→503", r.status_code == 503)
check("no secret leak", "SARVAM" not in r.json().get("error", "") and "traceback" not in r.text.lower())

print("\n[10] Malformed model output → 502")


def bad(_m):
    raise BadModelPayload("nope")


main.call_model = bad  # type: ignore
main._cache.clear()
r = client.post("/api/chat", json={"message": "How do I reset my password?", "conversation": []})
check("malformed→502", r.status_code == 502)

print("\n[11] Health endpoint")
r = client.get("/api/health")
check("health ok", r.status_code == 200 and r.json().get("provider") == "sarvam")

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
