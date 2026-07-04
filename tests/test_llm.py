import pytest

from leagents.llm import NullLLM, make_llm


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


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        make_llm("gemini:gemini-2.5-flash")


def test_gemini_builds_openai_compatible_client(monkeypatch):
    pytest.importorskip("openai")
    from leagents.llm import OpenAICompatibleLLM

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm = make_llm("gemini:gemini-2.5-flash")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.model == "gemini-2.5-flash"
    assert "generativelanguage.googleapis.com" in str(llm._client.base_url)
