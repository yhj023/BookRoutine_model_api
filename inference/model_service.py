"""
모델(CNN)과 EyeExtractor(MediaPipe)를 "한 번만" 로드해두고,
사진이 들어올 때마다 재사용하는 핵심 추론 모듈.

이 파일 하나가 두 군데에서 재사용된다:
  1) api_server.py — Node.js가 HTTP로 호출하는 실제 백엔드 연동용
  2) analyze_video.py — 녹화 영상을 오프라인으로 분석하는 데모 자료용
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from data.eye_extractor import EyeExtractor
from models.gaze_model import GazeModel

_LABEL_SCALE = 30.0  # train.py에서 쓴 값과 반드시 같아야 함


class GazeService:
    """모델을 한 번만 로드해서 들고 있다가, predict()가 호출될 때마다 재사용."""

    def __init__(self, checkpoint_path: str, model_asset_path: str = "models/assets/face_landmarker.task"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GazeService] 장치: {self.device}")

        self.model = GazeModel(pretrained=False).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[GazeService] 체크포인트 로드 완료: {checkpoint_path} "
              f"(학습 시 검증 오차: {ckpt.get('val_error_cm', '?')}cm)")

        self.extractor = EyeExtractor(model_asset_path=model_asset_path)
        print("[GazeService] MediaPipe 준비 완료 — 모델 로딩 끝, 요청 받을 준비 됐음")

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> dict | None:
        sample = self.extractor.process(image_bgr)
        if sample is None:
            return None

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            t = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = (t - 0.5) / 0.5
            return torch.from_numpy(t).permute(2, 0, 1).unsqueeze(0).to(self.device)

        left = to_tensor(sample.left_eye)
        right = to_tensor(sample.right_eye)
        pred = self.model(left, right)[0].cpu().numpy() * _LABEL_SCALE
        return {"gaze_x_cm": float(pred[0]), "gaze_y_cm": float(pred[1])}


_service: GazeService | None = None


def get_service(checkpoint_path: str = "checkpoints/best_model.pt") -> GazeService:
    """이미 로드돼 있으면 그대로 반환, 처음이면 그때 한 번만 로드."""
    global _service
    if _service is None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"체크포인트가 없어: {checkpoint_path} — train.py로 학습부터 해야 함")
        _service = GazeService(checkpoint_path)
    return _service