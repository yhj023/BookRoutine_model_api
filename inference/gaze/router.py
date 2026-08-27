"""시선추적 모델 전용 라우터. api_server.py가 이걸 통째로 등록한다."""

import numpy as np
import cv2
from fastapi import APIRouter, File, UploadFile

from inference.gaze.model_service import get_gaze_service

router = APIRouter(prefix="/gaze", tags=["gaze"])


@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    service = get_gaze_service()

    raw = await image.read()
    np_arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "이미지를 읽을 수 없어 — 형식을 확인해줘"}

    result = service.predict(img_bgr)
    if result is None:
        return {"error": "얼굴/눈을 찾지 못했어"}
    return result