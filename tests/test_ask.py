"""Phase 2 /ask endpoint tests.

Uses FastAPI's TestClient with the AskService dependency overridden
to a mock — no Postgres, no live LlamaIndex calls, no real LLMs.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from knowledge_bot.ask_service import AskService, EmptyVectorStoreError
from knowledge_bot.main import app
from knowledge_bot.models import AnswerMetrics, AnswerResponse
from knowledge_bot.routes.ask import get_ask_service


def _make_response() -> AnswerResponse:
    return AnswerResponse(
        answer="Mocked answer.",
        metrics=AnswerMetrics(
            input_tokens=100, output_tokens=50, latency_ms=42, cost_usd=0.000350
        ),
    )


@pytest.fixture
def fake_service() -> MagicMock:
    fake = MagicMock(spec=AskService)
    fake.answer.return_value = _make_response()
    return fake


@pytest.fixture
def client(fake_service):
    app.dependency_overrides[get_ask_service] = lambda: fake_service
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_ask_returns_answer_with_metrics(client, fake_service):
    response = client.post(
        "/ask", json={"question": "What is the minimum ceiling height?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Mocked answer."
    assert body["metrics"]["input_tokens"] == 100
    assert body["metrics"]["output_tokens"] == 50
    assert body["metrics"]["latency_ms"] == 42
    assert body["metrics"]["cost_usd"] == 0.00035
    fake_service.answer.assert_called_once_with("What is the minimum ceiling height?")


def test_ask_missing_question_returns_422(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_empty_store_returns_503(client, fake_service):
    fake_service.answer.side_effect = EmptyVectorStoreError(
        "Vector store returned no chunks."
    )
    response = client.post("/ask", json={"question": "anything"})
    assert response.status_code == 503
    assert "no chunks" in response.json()["detail"].lower()
