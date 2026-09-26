"""Keyword retrieval with a hard relevance gate.

Why not embeddings for three paragraphs? We need a score we can threshold on
before spending an LLM call. Weighted lexical overlap is deterministic, free,
and easy to debug in an interview — swap to hybrid/vector search when the corpus
grows (see README production notes).
"""

from __future__ import annotations

import re
from functools import lru_cache

from knowledge_base import DOCS, DocSection

# Words are [a-z0-9]+. Anything shorter than 2 letters is dropped in _tokens.
# Typos like "rst" / "pswd" do not match "reset" / "password" — that is the
# tradeoff of a literal keyword gate (see _PHRASE_MAP for the aliases we do accept).
_TOKEN = re.compile(r"[a-z0-9]+")

_STOP = frozenset(
    """
    a an the is are am was were be been being do does did doing have has had
    having i you he she it we they me my your our their this that these those
    and or but if in on at to for of from with as by about into over after
    before so such than then there when where which who whom why how what
    can could should would will just like please want need get got
    """.split()
)

# Everyday phrasing → vocabulary that actually appears in the docs / tags.
# Expansions are appended (not substituted) so the original query signal survives.
_PHRASE_MAP: dict[str, str] = {
    "2fa": "two factor authentication",
    "mfa": "two factor authentication",
    "otp": "authenticator code",
    "sms": "sms text",
    "text message": "sms",
    "sign in": "login",
    "log in": "login",
    "pwd": "password",
    "passcode": "password",
    "unsubscribe": "cancel subscription",
    "money back": "refund",
    "stop paying": "cancel subscription",
}

# Weights: title matches matter more than body chatter.
W_TITLE = 3.0
W_TAG = 2.0
W_BODY = 1.0

# Below this, we never call the model — unsupported / injection / off-topic.
RELEVANCE_FLOOR = 0.18


def _stem(word: str) -> str:
    """Lightweight suffix strip — enough for reset/resetting, cancel/cancelling.

    Not a real stemmer. "forgot" and "forget" stay different words, so the
    phrase map and tags have to cover the forms users actually type.
    """
    for suffix in ("ingly", "ing", "edly", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word[-1] == word[-2]:
        word = word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    """Lowercase, drop stop-words and 1-letter tokens, then stem."""
    return [
        _stem(w)
        for w in _TOKEN.findall(text.lower())
        if len(w) > 1 and w not in _STOP
    ]


def _expand_query(query: str) -> str:
    """Append synonym phrases. The original words stay, so a bad expansion cannot erase the query."""
    lowered = query.lower()
    extras = [expansion for phrase, expansion in _PHRASE_MAP.items() if phrase in lowered]
    return f"{lowered} {' '.join(extras)}" if extras else lowered


@lru_cache(maxsize=1)
def _section_weights() -> tuple[dict[str, float], ...]:
    """Build stem → max-weight maps once per process."""
    maps: list[dict[str, float]] = []
    for doc in DOCS:
        weights: dict[str, float] = {}
        for bag, weight in (
            (_tokens(doc.title), W_TITLE),
            (_tokens(" ".join(doc.tags)), W_TAG),
            (_tokens(doc.body), W_BODY),
        ):
            for stem in bag:
                if weights.get(stem, 0.0) < weight:
                    weights[stem] = weight
        maps.append(weights)
    return tuple(maps)


def _overlap_score(query_stems: set[str], vocab: dict[str, float]) -> float:
    """0–1 score. A query whose every word hits a title term scores 1.0."""
    if not query_stems:
        return 0.0
    matched = sum(vocab.get(stem, 0.0) for stem in query_stems)
    # Normalise by the theoretical max (every stem hits a title-weight term).
    ceiling = len(query_stems) * W_TITLE
    return min(matched / ceiling, 1.0)


def find_relevant(
    question: str,
    *,
    top_k: int = 2,
    floor: float = RELEVANCE_FLOOR,
) -> tuple[list[tuple[DocSection, float]], float]:
    """Return (hits, best_score). Empty hits means: skip the LLM."""
    stems = set(_tokens(_expand_query(question)))
    ranked = sorted(
        (
            (doc, _overlap_score(stems, vocab))
            for doc, vocab in zip(DOCS, _section_weights())
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best = ranked[0][1] if ranked else 0.0

    hits: list[tuple[DocSection, float]] = []
    for i, (doc, score) in enumerate(ranked[:top_k]):
        if score < floor:
            continue
        # Drop a weak second hit so we do not blend unrelated sections.
        if i > 0 and score < best * 0.5:
            continue
        hits.append((doc, score))
    return hits, best


def context_xml(hits: list[tuple[DocSection, float]]) -> str:
    """Pack retrieved docs for the prompt. Titles here are the only valid citations."""
    if not hits:
        return "<context>\n(no documentation matched this question)\n</context>"
    parts = [
        f'  <section title="{doc.title}">\n    {doc.body}\n  </section>'
        for doc, _ in hits
    ]
    return "<context>\n" + "\n".join(parts) + "\n</context>"
