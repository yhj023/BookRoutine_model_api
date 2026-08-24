"""
Node.js 백엔드가 HTTP로 호출하는 작은 API 서버.

실행:
    uvicorn inference.api_server:app --host 0.0.0.0 --port 8000

Node.js에서 호출하는 법 (예시, fetch 사용):
    const form = new FormData();
    form.append("image", imageBlob, "frame.jpg");
    const res = await fetch("http://localhost:8000/predict", { method: "POST", body: form });
    const { gaze_x_cm, gaze_y_cm } = await res.json();
"""

import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager

from inference.model_service import get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_service()  # 서버가 켜지는 시점에 모델을 미리 한 번 로드
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    service = get_service()

    raw = await image.read()
    np_arr = np.frombuffer(raw, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "이미지를 읽을 수 없어 — 형식을 확인해줘"}

    result = service.predict(img_bgr)
    if result is None:
        return {"error": "얼굴/눈을 찾지 못했어"}
    return result