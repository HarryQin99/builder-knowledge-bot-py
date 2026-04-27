"""POST /ask FastAPI router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from knowledge_bot.ask_service import AskService, EmptyVectorStoreError
from knowledge_bot.models import AnswerResponse, AskRequest

router = APIRouter()


def get_ask_service(request: Request) -> AskService:
    svc = getattr(request.app.state, "ask_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AskService not initialised. Check Postgres and run `python -m knowledge_bot.ingest`.",
        )
    return svc


@router.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest, service: AskService = Depends(get_ask_service)) -> AnswerResponse:
    try:
        return service.answer(req.question)
    except EmptyVectorStoreError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
