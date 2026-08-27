"""요약 모델 전용 라우터. api_server.py에서 등록된다. """

from fastapi import APIRouter
from pydantic import BaseModel

from inference.summary.model_service import get_summary_service

router = APIRouter(prefix="/summary", tags=["summary"])


class SummarizeRequest(BaseModel):
    text: str
    genre: str = "literary"  # 실제 서비스에선 책 카테고리로 자동 결정 (기존 주석대로)
    num_beams: int = 4


@router.post("/summarize")
def summarize(req: SummarizeRequest):
    service = get_summary_service()
    return service.summarize(req.text, genre=req.genre, num_beams=req.num_beams)