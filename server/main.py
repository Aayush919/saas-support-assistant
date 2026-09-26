"""FastAPI entrypoint — SaaS support assistant.

Request path:
  validate → rate-limit → cache → retrieve → relevance gate → Sarvam LLM →
  schema validate → source cross-check → confidence blend → respond

Two independent grounding layers (model alone is not trusted to refuse):
  1. Retrieval gate  — weak match ⇒ not_found, LLM never called
  2. Citation check  — model sources ∩ retrieved titles; empty ⇒ not_found
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict, defaultdict
from threading import Lock
from typing import Literal, Optional

import openai
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

import llm_sarvam as llm
import retrieve
from knowledge_base import titles
from llm_sarvam import BadModelPayload, MissingApiKey
from prompt_policy import REFUSAL_TEXT, SYSTEM_POLICY, clip_history, pack_turn

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
log = logging.getLogger("support.api")

# ---- tunables ------------------------------------------------------------- #
# History is capped so a long chat cannot blow the prompt (or the bill).
# The client may send up to 40 turns; we only forward the last 6, 500 chars each.
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CHARS = 500
# Identical stateless questions reuse this answer for 5 minutes (demo cache).
CACHE_TTL_S = 300
CACHE_MAX = 128
# Sliding window per client IP. In-memory is enough for one demo process.
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_S = int(os.getenv("RATE_LIMIT_WINDOW_S", "60"))

# Shown for upstream failures. The real exception stays in server logs only.
GENERIC_UPSTREAM = "The assistant is temporarily unavailable. Please try again shortly."


# ---- schemas -------------------------------------------------------------- #


class HistoryTurn(BaseModel):
    # extra="forbid" rejects unknown keys (the brief's invalid-payload case).
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """Body of POST /api/chat. Blank messages fail validation with HTTP 400."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=2000)
    # Hard cap on what the client may attach. clip_history trims this further.
    conversation: list[HistoryTurn] = Field(default_factory=list, max_length=40)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty.")
        return value.strip()


class ModelAnswer(BaseModel):
    """Contract we ask Sarvam for — validated before anything reaches the client."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    status: Literal["answered", "not_found"]
    sources: list[str]
    confidence: float


class Meta(BaseModel):
    """Bonus telemetry the chat UI prints under each reply. Not part of the brief contract."""

    latency_ms: int
    cached: bool
    # 0.00 means the question missed every doc, so the LLM was never called.
    retrieval_score: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    # Demo USD from PRICE_*_PER_MTOK. Not Sarvam's INR invoice. See llm_sarvam.estimate_usd.
    estimated_cost_usd: float = 0.0
    rate_limit_remaining: Optional[int] = None


class ChatResponse(BaseModel):
    """What the client renders. `meta` is extra; the brief only requires the four fields above it."""

    answer: str
    status: Literal["answered", "not_found"]
    sources: list[str]
    confidence: float
    meta: Meta


# ---- app + CORS ----------------------------------------------------------- #

app = FastAPI(
    title="SaaS Support Assistant",
    version="1.0.0",
    description="Grounded support API over a fixed product knowledge base (Sarvam).",
)

# React Vite default origin — tighten further in production.
_cors = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors if o.strip()],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---- rate limiter (per-IP sliding window) --------------------------------- #

_rl_lock = Lock()
_rl_hits: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    # Behind a trusted proxy you would read X-Forwarded-For; for local demo, peer.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def check_rate_limit(request: Request) -> tuple[bool, int]:
    """Return (allowed, remaining). Thread-safe in-memory window."""
    ip = _client_ip(request)
    now = time.time()
    with _rl_lock:
        bucket = _rl_hits[ip]
        # Drop timestamps outside the window.
        cutoff = now - RATE_LIMIT_WINDOW_S
        _rl_hits[ip] = [t for t in bucket if t >= cutoff]
        bucket = _rl_hits[ip]
        if len(bucket) >= RATE_LIMIT_REQUESTS:
            return False, 0
        bucket.append(now)
        remaining = RATE_LIMIT_REQUESTS - len(bucket)
        return True, remaining


# ---- response cache (stateless turns only) -------------------------------- #

_cache: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_cache_lock = Lock()


def cache_get(key: str) -> dict | None:
    """LRU read. Expired entries are dropped. Returns a copy so callers cannot mutate the store."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, payload = entry
        if time.time() - ts > CACHE_TTL_S:
            _cache.pop(key, None)
            return None
        _cache.move_to_end(key)
        return dict(payload)


def cache_put(key: str, payload: dict) -> None:
    """Insert and evict the oldest key once CACHE_MAX is exceeded."""
    with _cache_lock:
        _cache[key] = (time.time(), dict(payload))
        _cache.move_to_end(key)
        while len(_cache) > CACHE_MAX:
            _cache.popitem(last=False)


# ---- error envelopes (never leak stack traces / secrets) ------------------ #


def err(status: int, message: str) -> JSONResponse:
    """Every failure the browser sees is `{ "error": "..." }` — no traceback, no API key."""
    return JSONResponse(status_code=status, content={"error": message})


@app.exception_handler(RequestValidationError)
async def on_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Turn Pydantic errors into one readable sentence. Empty message is called out by name."""
    errors = exc.errors()
    if not errors:
        return err(400, "Invalid request payload.")
    first = errors[0]
    msg = str(first.get("msg", "Invalid request payload.")).replace("Value error, ", "")
    field = ".".join(str(p) for p in first.get("loc", ())[1:])
    if "empty" in msg.lower():
        return err(400, "Message cannot be empty.")
    if field:
        return err(400, f"Invalid request payload: '{field}' — {msg}")
    return err(400, f"Invalid request payload: {msg}")


@app.exception_handler(Exception)
async def on_unexpected(_: Request, exc: Exception) -> JSONResponse:
    """Last resort. Log the traceback; the client only sees a generic 500."""
    log.exception("unhandled: %s", exc)
    return err(500, "Internal server error.")


# ---- helpers -------------------------------------------------------------- #


def call_model(messages: list[dict]) -> tuple[ModelAnswer, dict[str, int]]:
    """Seam patched by offline tests — keeps tests free of network/Sarvam."""
    return llm.generate_json(SYSTEM_POLICY, messages, ModelAnswer)


def not_found_core(retrieval_score: float, usage: dict[str, int] | None = None) -> dict:
    """Brief's unsupported shape. confidence 0.15 is fixed, not a model score.

    usage is None when the retrieval gate skipped the LLM (0 tokens).
    It is set when the model ran but its citation failed the cross-check.
    """
    return {
        "answer": REFUSAL_TEXT,
        "status": "not_found",
        "sources": [],
        "confidence": 0.15,
        "retrieval_score": round(retrieval_score, 3),
        "input_tokens": (usage or {}).get("input_tokens", 0),
        "output_tokens": (usage or {}).get("output_tokens", 0),
    }


# ---- routes --------------------------------------------------------------- #


@app.get("/api/health")
def health() -> dict:
    """Sidebar probe. Reports provider and whether a key is set, never the key itself."""
    return {
        "ok": True,
        "provider": "sarvam",
        "model": llm.SARVAM_MODEL,
        "api_key_configured": bool(os.getenv("SARVAM_API_KEY", "").strip()),
        "json_mode": llm.json_mode_label(),
        "kb_sections": titles(),
        "rate_limit": {
            "requests": RATE_LIMIT_REQUESTS,
            "window_seconds": RATE_LIMIT_WINDOW_S,
        },
        "cache_size": len(_cache),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> JSONResponse:
    """Answer from the three docs, or refuse. The model is only called after retrieval passes."""
    started = time.perf_counter()

    allowed, remaining = check_rate_limit(request)
    if not allowed:
        return err(429, "Rate limit exceeded. Please wait a moment and try again.")

    def finish(core: dict, cached: bool) -> JSONResponse:
        # A cache hit did not call Sarvam, so this request's token spend is zero.
        in_tok = 0 if cached else core["input_tokens"]
        out_tok = 0 if cached else core["output_tokens"]
        body = ChatResponse(
            answer=core["answer"],
            status=core["status"],
            sources=core["sources"],
            confidence=core["confidence"],
            meta=Meta(
                latency_ms=int((time.perf_counter() - started) * 1000),
                cached=cached,
                retrieval_score=core["retrieval_score"],
                model=llm.SARVAM_MODEL,
                input_tokens=in_tok,
                output_tokens=out_tok,
                estimated_cost_usd=llm.estimate_usd(in_tok, out_tok),
                rate_limit_remaining=remaining,
            ),
        )
        return JSONResponse(status_code=200, content=body.model_dump())

    # Cache only when there is no conversation — same words can mean different
    # things mid-thread once the user has established context.
    stateless = not payload.conversation
    cache_key = payload.message.casefold()
    if stateless:
        hit = cache_get(cache_key)
        if hit is not None:
            log.info("cache hit | %r", payload.message[:50])
            return finish(hit, cached=True)

    # 1) Retrieve + gate ---------------------------------------------------- #
    hits, best = retrieve.find_relevant(payload.message)
    log.info("retrieve | best=%.3f hits=%s", best, [d.title for d, _ in hits])

    if not hits:
        core = not_found_core(best)
        if stateless:
            cache_put(cache_key, core)
        return finish(core, cached=False)

    # 2) Build messages + call Sarvam --------------------------------------- #
    messages = clip_history(
        payload.conversation,
        max_turns=MAX_HISTORY_TURNS,
        max_chars=MAX_HISTORY_CHARS,
    )
    messages.append(
        {
            "role": "user",
            "content": pack_turn(retrieve.context_xml(hits), payload.message),
        }
    )

    # Map provider failures to safe HTTP errors. Details go to the log, not the client.
    # 503 credentials / upstream, 504 timeout, 502 malformed JSON after the retry.
    try:
        raw, usage = call_model(messages)
    except MissingApiKey:
        log.error("SARVAM_API_KEY missing")
        return err(503, "Assistant is not configured. Missing API credentials.")
    except openai.AuthenticationError:
        log.error("Sarvam authentication failed")
        return err(503, "Assistant is not configured. Invalid API credentials.")
    except openai.RateLimitError:
        log.warning("Sarvam upstream rate limit")
        return err(503, GENERIC_UPSTREAM)
    except openai.APITimeoutError:
        log.warning("Sarvam timeout")
        return err(504, "The assistant timed out. Please try again.")
    except openai.APIConnectionError:
        log.warning("Sarvam connection error")
        return err(503, GENERIC_UPSTREAM)
    except openai.APIStatusError as exc:
        log.error("Sarvam status %s", getattr(exc, "status_code", "?"))
        return err(503, GENERIC_UPSTREAM)
    except BadModelPayload as exc:
        log.error("malformed model output: %s", exc)
        return err(502, "The assistant returned an unreadable response. Please retry.")

    # 3) Citation cross-check (layer 2 grounding) --------------------------- #
    allowed_titles = {doc.title for doc, _ in hits}
    clean_sources = [s for s in raw.sources if s in allowed_titles]
    dropped = set(raw.sources) - set(clean_sources)
    if dropped:
        log.warning("dropped fake citations | %s", sorted(dropped))

    if raw.status == "not_found" or not clean_sources:
        # Model refused, or answered without a valid citation → safe refusal.
        core = not_found_core(best, usage)
        if stateless:
            cache_put(cache_key, core)
        return finish(core, cached=False)

    # Do not trust the model's self-score alone. 35% retrieval evidence, 65% model,
    # capped at 0.99 so the UI never shows a fake 1.00.
    model_c = min(max(raw.confidence, 0.0), 1.0)
    blended = min(0.35 * best + 0.65 * model_c, 0.99)

    core = {
        "answer": raw.answer.strip(),
        "status": "answered",
        "sources": clean_sources,
        "confidence": round(blended, 3),
        "retrieval_score": round(best, 3),
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }
    if stateless:
        cache_put(cache_key, core)
    return finish(core, cached=False)
