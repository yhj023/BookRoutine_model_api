"""
메인 실행 서버

설명: Node.js 백엔드가 HTTP로 호출하는 작은 API 서버.
- 여러 라우터(3개)를 묶는 메인 서버로 작성 

모델: 각 모델 라우터가 자기 모델을 각자 담당하고, 서버가 켜질 때
    lifespan에서 모델을 전부 한 번씩만 로드해둔다. 
    실행 모델 3개(시선추적/객체탐지/요약)

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
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference.gaze.router import router as gaze_router
from inference.gaze.model_service import get_gaze_service
from inference.summary.router import router as summary_router
from inference.summary.model_service import get_summary_service

from inference.focus_yolo.router import router as focus_router
from inference.focus_yolo.model_service import get_focus_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_gaze_service()  # 시선추적 모델 로드
    get_focus_service()   # YOLO 로드
    get_summary_service() # 요약모델 로드
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(gaze_router)
app.include_router(focus_router) 
app.include_router(summary_router)

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