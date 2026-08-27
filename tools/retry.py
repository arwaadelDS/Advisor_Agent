"""Retry a model call, but only when a second attempt could answer differently.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, TypeVar

from google.genai.errors import ServerError
from langchain_core.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Three attempts spends at most ~6s of waiting before giving up.
ATTEMPTS = 3
FIRST_WAIT = 2.0
MAX_WAIT = 16.0

# The request, the key or the account is wrong. Sending it again changes nothing.
_PERMANENT = (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
)

# The provider stumbled, or is asking us to slow down. ChatGoogleGenerativeAIError
# is the unmapped-code catch-all, safe here because _PERMANENT is tested first.
_TEMPORARY = (
    ChatGoogleGenerativeAIError,
    ModelAPIError,
    ModelConnectionError,
    ModelTimeoutError,
    ServerError,
    ConnectionError,
    TimeoutError,
)

# 429 wording that means the account is spent rather than the calls too fast.
_SPENT = ("credit", "depleted", "billing", "perday", "daily")


def _letters(text: str) -> str:
    """Lowercase letters only, so "per day", "per_day" and "PerDay" all match."""
    return re.sub(r"[^a-z]", "", text.lower())


def worth_retrying(exc: BaseException) -> bool:
    """True when sending the same call again later could get a different answer."""
    if isinstance(exc, _PERMANENT):
        return False
    if isinstance(exc, ModelRateLimitError):
        return not any(word in _letters(str(exc)) for word in _SPENT)
    return isinstance(exc, _TEMPORARY)


def _log_retry(state: RetryCallState) -> None:
    exc = state.outcome.exception() if state.outcome else None
    logger.warning(
        "model call failed (attempt %s of %s), retrying in %.1fs: %s",
        state.attempt_number,
        ATTEMPTS,
        state.next_action.sleep if state.next_action else 0.0,
        exc,
    )


def call_with_retry(call: Callable[[], T]) -> T:
    """Run ``call``, waiting and trying again while the failure looks temporary.

    The last failure is re-raised as it was, so callers see the real error.
    """
    # Built per call, so the numbers above stay adjustable -- which is how the
    # tests avoid waiting.
    retrying = Retrying(
        stop=stop_after_attempt(ATTEMPTS),
        wait=wait_exponential_jitter(initial=FIRST_WAIT, max=MAX_WAIT),
        retry=retry_if_exception(worth_retrying),
        before_sleep=_log_retry,
        reraise=True,
    )
    return retrying(call)
