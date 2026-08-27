"""Test suite for tools/retry.py

The waiting is tenacity's and is not re-tested. What is tested is the decision
in front of it: which failures earn a second attempt and which come straight
back. The waits are set to zero, so nothing here sleeps.
"""

import pytest
from google.genai.errors import ServerError
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

from tools import retry as rt


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """Read at call time inside call_with_retry, so this removes the sleeps."""
    monkeypatch.setattr(rt, "FIRST_WAIT", 0.0)
    monkeypatch.setattr(rt, "MAX_WAIT", 0.0)


def failing(exc, succeed_on=None):
    """A call that raises ``exc`` until attempt ``succeed_on``, counting calls."""
    calls = []

    def call():
        calls.append(len(calls) + 1)
        if succeed_on is not None and len(calls) >= succeed_on:
            return "answer"
        raise exc

    call.calls = calls
    return call


# ---------------------------------------------------------------------------
# Which failures are worth another attempt
# ---------------------------------------------------------------------------

class TestFailuresThatClearOnTheirOwn:
    def test_a_server_error_is_retried(self):
        assert rt.worth_retrying(ServerError(503, {"message": "overloaded"}))

    def test_a_timeout_is_retried(self):
        assert rt.worth_retrying(ModelTimeoutError("deadline exceeded"))

    def test_a_dropped_connection_is_retried(self):
        assert rt.worth_retrying(ConnectionError("connection reset"))

    def test_an_unrecognised_google_error_is_retried(self):
        # The catch-all the wrapper falls back to when it cannot map the code.
        assert rt.worth_retrying(ChatGoogleGenerativeAIError("something odd"))


class TestFailuresThatNeverClear:
    @pytest.mark.parametrize(
        "exc",
        [
            ModelAuthenticationError("401 API key not valid"),
            ModelPermissionDeniedError("403 The caller does not have permission"),
            ModelNotFoundError("404 model gemini-9.9-flash was not found"),
            ModelInvalidRequestError("400 malformed request"),
        ],
    )
    def test_the_key_the_model_and_the_request_are_not_retried(self, exc):
        assert not rt.worth_retrying(exc)

    def test_an_ordinary_bug_is_not_retried(self):
        # Retrying a TypeError in our own code only hides where it came from.
        assert not rt.worth_retrying(TypeError("expected str"))


class TestTheTwoKindsOf429:
    def test_going_too_fast_is_retried(self):
        exc = ModelRateLimitError(
            "429 RESOURCE_EXHAUSTED. Quota exceeded for quota metric "
            "'GenerateRequestsPerMinutePerProjectPerModel'"
        )
        assert rt.worth_retrying(exc)

    def test_depleted_credits_are_not_retried(self):
        # The message the demo actually hit. Waiting does not mint credits.
        exc = ModelRateLimitError(
            "Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429. "
            "Your prepayment credits are depleted. Please go to AI Studio"
        )
        assert not rt.worth_retrying(exc)

    def test_a_daily_quota_is_not_retried(self):
        # The free tier resets at midnight Pacific, which no demo waits for.
        exc = ModelRateLimitError(
            "429 quota metric 'GenerateRequestsPerDayPerProjectPerModel'"
        )
        assert not rt.worth_retrying(exc)

    @pytest.mark.parametrize("wording", ["per day", "per_day", "PerDay"])
    def test_the_wording_is_matched_however_it_is_spaced(self, wording):
        assert not rt.worth_retrying(ModelRateLimitError(f"429 limit {wording}"))


# ---------------------------------------------------------------------------
# What call_with_retry does with that decision
# ---------------------------------------------------------------------------

class TestRunningTheCall:
    def test_a_call_that_works_is_made_once(self):
        call = failing(None, succeed_on=1)
        assert rt.call_with_retry(call) == "answer"
        assert len(call.calls) == 1

    def test_a_temporary_failure_is_tried_again_and_can_succeed(self):
        call = failing(ModelTimeoutError("slow"), succeed_on=2)
        assert rt.call_with_retry(call) == "answer"
        assert len(call.calls) == 2

    def test_it_gives_up_after_the_configured_attempts(self):
        call = failing(ModelTimeoutError("slow"))
        with pytest.raises(ModelTimeoutError):
            rt.call_with_retry(call)
        assert len(call.calls) == rt.ATTEMPTS

    def test_a_permanent_failure_is_raised_on_the_first_attempt(self):
        call = failing(ModelPermissionDeniedError("403"))
        with pytest.raises(ModelPermissionDeniedError):
            rt.call_with_retry(call)
        assert len(call.calls) == 1

    def test_the_original_error_comes_back_not_a_tenacity_wrapper(self):
        # reraise=True, so the advisor still sees what the provider said.
        call = failing(ModelRateLimitError("429 per minute"))
        with pytest.raises(ModelRateLimitError, match="per minute"):
            rt.call_with_retry(call)

    def test_each_retry_is_logged(self, caplog):
        call = failing(ModelTimeoutError("slow"), succeed_on=3)
        with caplog.at_level("WARNING", logger="tools.retry"):
            rt.call_with_retry(call)
        assert len(caplog.records) == 2
