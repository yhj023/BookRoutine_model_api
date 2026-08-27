"""
독서방(종이책) 집중도 모델 전용 라우터. api_server.py가 이걸 통째로 등록한다.

흐름: /focus/session/start (독서 시작) -> /focus/frame (0.5~1초 간격 반복 호출)
      -> /focus/session/end (독서 종료, 메모리 정리)
"""

import numpy as np
import cv2
from fastapi import APIRouter, File, Form, UploadFile

from inference.focus_yolo.model_service import get_focus_service

router = APIRouter(prefix="/focus", tags=["focus"])


@router.post("/session/start")
def start_session(session_id: str = Form(...)):
    get_focus_service().start_session(session_id)
    return {"ok": True}


@router.post("/session/end")
def end_session(session_id: str = Form(...)):
    get_focus_service().end_session(session_id)
    return {"ok": True}


@router.post("/frame")
async def process_frame(session_id: str = Form(...), image: UploadFile = File(...)):
    service = get_focus_service()

    raw = await image.read()
    np_arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "이미지를 읽을 수 없어 — 형식을 확인해줘"}

    return service.process_frame(session_id, img_bgr)