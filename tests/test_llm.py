"""Test suite for tools/llm.py

The Retrying wrapper is the part worth pinning down: it has to stay transparent
to callers expecting a chat model, while still putting the retry outside every
call. tools/retry.py decides what gets retried; these only check the decision
is reached.
"""

import pytest
from langchain_core.exceptions import ModelPermissionDeniedError, ModelTimeoutError

from tools import llm as llm_module
from tools import retry as rt
from tools.llm import Retrying, text_of


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(rt, "FIRST_WAIT", 0.0)
    monkeypatch.setattr(rt, "MAX_WAIT", 0.0)


@pytest.fixture(autouse=True)
def fresh_llm_cache():
    """get_llm is cached, so a model built under one test's env would outlive it."""
    llm_module.get_llm.cache_clear()
    yield
    llm_module.get_llm.cache_clear()


class FakeModel:
    """Stands in for ChatGoogleGenerativeAI, counting how often it was called."""

    def __init__(self, raises=None, succeed_on=None):
        self.raises = raises
        self.succeed_on = succeed_on
        self.calls = []
        self.structured_with = None
        self.model_name = "fake-model"

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        if self.raises and (
            self.succeed_on is None or len(self.calls) < self.succeed_on
        ):
            raise self.raises
        return "reply"

    def with_structured_output(self, schema):
        self.structured_with = schema
        return FakeModel()


# ---------------------------------------------------------------------------
# The wrapper stays out of the way
# ---------------------------------------------------------------------------

class TestItStillBehavesLikeAModel:
    def test_a_call_that_works_passes_straight_through(self):
        model = FakeModel()
        assert Retrying(model).invoke(["hi"]) == "reply"
        assert model.calls == [["hi"]]

    def test_arguments_reach_the_model_unchanged(self):
        model = FakeModel()
        Retrying(model).invoke(["hi"], config={"tags": ["x"]})
        assert model.calls == [["hi"]]

    def test_anything_else_is_forwarded_to_the_model(self):
        # The supervisor and the agents read attributes off the model directly.
        assert Retrying(FakeModel()).model_name == "fake-model"

    def test_get_llm_hands_back_a_wrapped_model(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")
        assert isinstance(llm_module.get_llm(), Retrying)


class TestStructuredOutputIsWrappedToo:
    def test_the_schema_reaches_the_model(self):
        model = FakeModel()
        Retrying(model).with_structured_output({"schema": 1})
        assert model.structured_with == {"schema": 1}

    def test_what_comes_back_still_retries(self):
        # Without this, the supervisor's routing call and both SQL query
        # builders would quietly lose their retry.
        assert isinstance(Retrying(FakeModel()).with_structured_output(None), Retrying)


# ---------------------------------------------------------------------------
# The retry is actually in the path
# ---------------------------------------------------------------------------

class TestTheRetrySurroundsTheCall:
    def test_a_temporary_failure_is_tried_again(self):
        model = FakeModel(raises=ModelTimeoutError("slow"), succeed_on=2)
        assert Retrying(model).invoke(["hi"]) == "reply"
        assert len(model.calls) == 2

    def test_a_permanent_failure_is_not(self):
        model = FakeModel(raises=ModelPermissionDeniedError("403"))
        with pytest.raises(ModelPermissionDeniedError):
            Retrying(model).invoke(["hi"])
        assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# Reading the reply
# ---------------------------------------------------------------------------

class TestGettingTheProseOut:
    def test_a_plain_string_content(self):
        assert text_of(type("R", (), {"content": "  hello  "})()) == "hello"

    def test_gemini_3_typed_blocks(self):
        response = type(
            "R",
            (),
            {
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "thinking", "signature": "abc"},
                    {"type": "text", "text": "world"},
                ]
            },
        )()
        assert text_of(response) == "hello world"
