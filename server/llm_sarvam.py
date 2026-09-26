"""Sarvam Chat Completions client (OpenAI-compatible transport).

Why Sarvam for this exercise:
- OpenAI-compatible /v1/chat/completions → same SDK patterns, easy provider swap
- Low-latency chat models suitable for a live support demo
- Indian-language strength is a bonus; English grounding still works the same

Auth: Sarvam accepts `Authorization: Bearer <key>` (OpenAI style) and/or
`api-subscription-key`. We use the OpenAI SDK with base_url pointed at Sarvam.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TypeVar

import openai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

log = logging.getLogger("support.sarvam")

T = TypeVar("T", bound=BaseModel)

SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1")
# Conversations variant is tuned for chat; swap to sarvam-105b for heavier reasoning.
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-105b-conversations")
# Maps to HTTP 504 in main.py when Sarvam does not answer in time.
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
# Support answers are short. A low cap keeps cost and rambling down.
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
# Low temperature: we want the docs repeated, not a creative rewrite.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Demo cost chip only. MTOK = price in USD per 1,000,000 tokens.
# 0.30 means $0.30 for one million input tokens; 0.90 is the same unit for output.
# Sarvam's published list for sarvam-105b-conversations is in rupees, not dollars:
# ₹29.28 input / ₹10.98 cached input / ₹73.20 output per 1M tokens
# (https://docs.sarvam.ai/api/getting-started/pricing).
# 0.30 and 0.90 are a rounded USD stand-in near that list. They are not an invoice,
# and cached-input (the cheaper ₹10.98 tier) is not split out here.
PRICE_IN_PER_MTOK = float(os.getenv("PRICE_INPUT_PER_MTOK", "0.30"))
PRICE_OUT_PER_MTOK = float(os.getenv("PRICE_OUTPUT_PER_MTOK", "0.90"))


class MissingApiKey(Exception):
    """SARVAM_API_KEY missing or blank."""


class BadModelPayload(Exception):
    """Model returned text we could not validate into the required schema."""


_client: openai.OpenAI | None = None
_json_mode_ok: bool | None = None


def get_client() -> openai.OpenAI:
    global _client
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise MissingApiKey("SARVAM_API_KEY is not set")
    if _client is None:
        _client = openai.OpenAI(
            base_url=SARVAM_BASE_URL,
            api_key=key,
            timeout=TIMEOUT_S,
            max_retries=1,
            default_headers={"api-subscription-key": key},
        )
    return _client


# Models sometimes wrap JSON in ```json fences or a <think> block. Strip both
# before we look for the object. The schema check still runs after this.
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """Pull the first balanced {...} object out of messy model text."""
    cleaned = _THINK.sub("", text).strip()
    cleaned = _FENCE.sub("", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        raise BadModelPayload("no JSON object found")

    # Walk braces, but ignore { } that sit inside JSON strings.
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    raise BadModelPayload("unterminated JSON object")


def _complete(messages: list[dict], want_json_object: bool):
    """One chat completion; JSON object mode is best-effort."""
    global _json_mode_ok
    kwargs: dict = {
        "model": SARVAM_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    if want_json_object and _json_mode_ok is not False:
        try:
            resp = get_client().chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
            _json_mode_ok = True
            return resp
        except openai.BadRequestError as exc:
            if "response_format" not in str(exc).lower() and "json" not in str(exc).lower():
                raise
            log.warning("Sarvam rejected response_format; falling back to prompt-only JSON")
            _json_mode_ok = False
    return get_client().chat.completions.create(**kwargs)


def generate_json(
    system_prompt: str,
    messages: list[dict],
    schema: type[T],
) -> tuple[T, dict[str, int]]:
    """Call Sarvam, extract JSON, validate with Pydantic. One corrective retry."""
    convo = [{"role": "system", "content": system_prompt}, *messages]
    usage = {"input_tokens": 0, "output_tokens": 0}
    last_err = ""

    for attempt in (1, 2):
        response = _complete(convo, want_json_object=True)
        # Sum both attempts. A retry is a second billable call, so the UI should show it.
        u = getattr(response, "usage", None)
        usage["input_tokens"] += getattr(u, "prompt_tokens", 0) or 0
        usage["output_tokens"] += getattr(u, "completion_tokens", 0) or 0

        content = ""
        if response.choices:
            content = response.choices[0].message.content or ""

        if content:
            try:
                parsed = schema.model_validate_json(extract_json(content))
                return parsed, usage
            except (BadModelPayload, ValidationError, json.JSONDecodeError) as exc:
                last_err = str(exc)

        if attempt == 1:
            log.warning("invalid model JSON, retrying once: %s", last_err[:180])
            convo = [
                *convo,
                {"role": "assistant", "content": (content or "")[:800]},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON for the required schema. "
                        "Reply with a single raw JSON object only — no markdown, no commentary."
                    ),
                },
            ]

    raise BadModelPayload(last_err or "empty model response")


def estimate_usd(input_tokens: int, output_tokens: int) -> float:
    """USD demo estimate: (tokens × price_per_million) / 1_000_000."""
    return round(
        input_tokens * PRICE_IN_PER_MTOK / 1_000_000
        + output_tokens * PRICE_OUT_PER_MTOK / 1_000_000,
        6,
    )


def json_mode_label() -> str:
    """Health-check label: have we confirmed Sarvam accepts response_format yet?"""
    return {None: "untested", True: "active", False: "prompt-only"}[_json_mode_ok]
