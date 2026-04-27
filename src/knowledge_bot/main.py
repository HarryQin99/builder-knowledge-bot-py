"""Phase 2 FastAPI app entrypoint.

The lifespan builds the LlamaIndex query engine once at startup and stashes
the AskService on app.state. If startup fails (e.g., Postgres unreachable),
the service is None and /ask returns 503 — useful for tests that override
the dependency without needing a live Postgres.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from knowledge_bot.ask_service import AskService
from knowledge_bot.query import build_query_engine
from knowledge_bot.routes.ask import router as ask_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine, counter = build_query_engine()
        app.state.ask_service = AskService(engine, counter)
        log.info("AskService initialised.")
    except Exception as e:  # noqa: BLE001
        log.warning("AskService init failed; /ask will return 503. Reason: %s", e)
        app.state.ask_service = None
    yield


app = FastAPI(title="Knowledge Bot", lifespan=lifespan)
app.include_router(ask_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
