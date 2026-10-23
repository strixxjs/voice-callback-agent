"""Pytest fixtures for isolating test execution from local .env and credentials."""

from __future__ import annotations

import pytest

import agent.extraction
import agent.worker
import app.config
import tests.test_agent
from app.config import Settings


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Provides isolated test settings without reading local .env, matching CI."""
    clean_settings = Settings(_env_file=None)
    monkeypatch.setattr(app.config, "settings", clean_settings)
    monkeypatch.setattr(agent.worker, "settings", clean_settings)
    monkeypatch.setattr(agent.extraction, "settings", clean_settings)
    monkeypatch.setattr(tests.test_agent, "settings", clean_settings)
    try:
        import scripts.call_me

        monkeypatch.setattr(scripts.call_me, "settings", clean_settings)
    except ImportError:
        pass
    return clean_settings
