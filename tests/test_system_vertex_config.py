from __future__ import annotations

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.vertex_client import VertexAIClient
from tradingagents.system.config import load_settings


def test_settings_default_to_vertex_gemini(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "vertex")
    monkeypatch.setenv("TRADINGAGENTS_LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_LLM_DEEP_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_LLM_QUICK_MODEL", "gemini-2.5-flash")
    settings = load_settings()
    assert settings.llm.provider == "vertex"
    assert settings.llm.model == "gemini-2.5-flash"
    assert settings.llm.deep_model == "gemini-2.5-flash"
    assert settings.llm.quick_model == "gemini-2.5-flash"


def test_tradingagents_config_includes_vertex_project_region(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "vertex")
    monkeypatch.setenv("TRADINGAGENTS_LLM_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_LLM_DEEP_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_LLM_QUICK_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_VERTEX_PROJECT", "proj-123")
    monkeypatch.setenv("TRADINGAGENTS_VERTEX_REGION", "us-central1")
    settings = load_settings()
    payload = settings.as_tradingagents_config()
    assert payload["llm_provider"] == "vertex"
    assert payload["deep_think_llm"] == "gemini-2.5-flash"
    assert payload["quick_think_llm"] == "gemini-2.5-flash"
    assert payload["vertex_project"] == "proj-123"
    assert payload["vertex_region"] == "us-central1"


def test_factory_vertex_provider_aliases():
    for provider in ("vertex", "vertexai", "google_vertex", "google-vertex"):
        client = create_llm_client(provider=provider, model="gemini-2.5-flash")
        assert isinstance(client, VertexAIClient)
