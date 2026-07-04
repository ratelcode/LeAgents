import pytest

from leloop.llm import NullLLM, make_llm


def test_no_spec_gives_null_llm():
    llm = make_llm(None)
    assert isinstance(llm, NullLLM)
    assert llm.complete("anything", system="sys") == ""


def test_empty_spec_gives_null_llm():
    assert isinstance(make_llm(""), NullLLM)


def test_spec_without_model_rejected():
    with pytest.raises(ValueError, match="provider:model"):
        make_llm("anthropic")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_llm("foo:bar")
