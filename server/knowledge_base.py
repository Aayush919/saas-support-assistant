"""Product documentation store.

The assignment supplies a fixed, tiny corpus. We keep it as typed records in
code so retrieval is deterministic and auditable — no PDF pipeline, no database,
and no chance of sneaking in docs the interviewer did not provide.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocSection:
    """One grounded document the assistant is allowed to cite."""

    id: str
    title: str
    body: str
    # Curated terms that users actually type (synonyms live in retrieve.py).
    tags: tuple[str, ...]


# Exact product facts from the brief — do not paraphrase here; the LLM may
# quote menu paths and time limits verbatim.
# These three sections are the entire product truth. Anything else is not_found.
DOCS: tuple[DocSection, ...] = (
    DocSection(
        id="password_reset",
        title="Password Reset",
        body=(
            "Users can reset their password from the login page by selecting "
            "Forgot Password. A reset link remains valid for 30 minutes."
        ),
        tags=(
            "password",
            "reset",
            "forgot",
            "login",
            "signin",
            "credentials",
            "recover",
            "link",
            "expire",
            "minutes",
        ),
    ),
    DocSection(
        id="two_factor",
        title="Two-Factor Authentication",
        body=(
            "Two-factor authentication can be enabled from: "
            "Settings > Security > Two-Factor Authentication. "
            "The application currently supports authenticator applications. "
            "SMS authentication is not supported."
        ),
        tags=(
            "two",
            "factor",
            "authentication",
            "authenticator",
            "2fa",
            "mfa",
            "sms",
            "security",
            "settings",
            "otp",
            "code",
        ),
    ),
    DocSection(
        id="subscription_cancel",
        title="Subscription Cancellation",
        body=(
            "Users can cancel a subscription from: "
            "Settings > Billing > Manage Subscription. "
            "The subscription remains active until the end of the current "
            "billing period. Refunds are reviewed manually by the billing team "
            "and are not guaranteed."
        ),
        tags=(
            "cancel",
            "cancellation",
            "subscription",
            "billing",
            "refund",
            "plan",
            "unsubscribe",
            "manage",
            "charge",
            "renew",
        ),
    ),
)


def titles() -> list[str]:
    return [doc.title for doc in DOCS]


def by_title(title: str) -> DocSection | None:
    for doc in DOCS:
        if doc.title == title:
            return doc
    return None
