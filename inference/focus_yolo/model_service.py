"""
독서방(종이책) 집중도 측정 모델 — YOLOv8n(open_book/hand 감지) + 규칙 기반
시계열 판정 엔진.

원본 로직 출처: 팀원이 만든 app.py(Streamlit 데모)의 FocusEngine·compute_iou를
그대로 재사용했다 — 판정 로직 자체는 이미 검증된 것이라 새로 만들지 않았다.
Streamlit 전용 코드(화면 렌더링, st.cache_resource 등)만 제거했다.

시선추적/요약과 다른 점: 이 모델은 "세션이 흐르며 누적되는 상태"가 필요하다
(겹침 지속시간, 누적 페이지 넘김 횟수 등). 그래서 세션마다 별도의
FocusEngine 인스턴스를 메모리에 들고 있는다.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO

MODEL_PATH = "model_weights/focus_yolo/best.pt"

CLASS_OPEN_BOOK = 0
CLASS_HAND = 1

# 원본 app.py와 동일한 값 — 페이지 넘김으로 인정할 손-책 겹침 지속시간 범위(초)
EVENT_MIN_SEC = 0.6
EVENT_MAX_SEC = 1.5
ALERT_THRESHOLD = 10  # 이 시간(초) 넘게 페이지가 안 넘어가면 집중도 저하 경고


def compute_iou(box1: list, box2: list) -> float:
    """두 바운딩박스의 IoU. 원본 app.py 로직 그대로."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def get_boxes_by_class(results, cls_id: int) -> list:
    boxes = []
    for box in results[0].boxes:
        if int(box.cls[0]) == cls_id:
            boxes.append(list(map(int, box.xyxy[0])))
    return boxes


class FocusEngine:
    """세션 하나의 집중도 상태를 누적 추적. 원본 app.py의 FocusEngine 클래스 그대로."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.overlap_start: Optional[float] = None
        self.page_events: list[float] = []
        self.focus_segments: list[tuple] = []
        self.reading_start: Optional[float] = None
        self.session_start: Optional[float] = None

    def update(self, timestamp: float, book_boxes: list, hand_boxes: list) -> dict:
        if self.session_start is None:
            self.session_start = timestamp
            self.reading_start = timestamp

        max_iou = 0.0
        if book_boxes and hand_boxes:
            for bk in book_boxes:
                for hd in hand_boxes:
                    max_iou = max(max_iou, compute_iou(bk, hd))

        is_overlapping = max_iou > 0
        if is_overlapping:
            if self.overlap_start is None:
                self.overlap_start = timestamp
            status = "OVERLAP"
        else:
            if self.overlap_start is not None:
                duration = timestamp - self.overlap_start
                if EVENT_MIN_SEC <= duration <= EVENT_MAX_SEC:
                    self.page_events.append(timestamp)
                    if self.reading_start is not None:
                        seg_duration = self.overlap_start - self.reading_start
                        if seg_duration > 0:
                            self.focus_segments.append((self.reading_start, self.overlap_start))
                    self.reading_start = timestamp
                self.overlap_start = None
            status = "PURE_READING" if (book_boxes or hand_boxes) else "NO_OBJECT"

        last_event_time = self.page_events[-1] if self.page_events else self.session_start
        interval = timestamp - last_event_time if last_event_time else 0.0

        completed_max = max((end - start for start, end in self.focus_segments), default=0.0)
        current_seg = (timestamp - self.reading_start) if self.reading_start else 0.0
        max_focus_sec = max(completed_max, current_seg)

        return {
            "status": status,
            "iou": round(max_iou, 3),
            "interval": round(interval, 1),
            "page_count": len(self.page_events),
            "max_focus_sec": round(max_focus_sec, 1),
            "alert": interval > ALERT_THRESHOLD,
        }


class FocusYoloService:
    """YOLO 모델은 한 번만 로드, 세션별 FocusEngine은 메모리에서 관리."""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"YOLO 모델 파일이 없어: {model_path}\n"
                f"팀원한테 받은 best.pt를 이 경로에 넣어줘."
            )
        self.model = YOLO(model_path)
        self.engines: dict[str, FocusEngine] = {}
        print(f"[FocusYoloService] YOLO 모델 로드 완료: {model_path} — 요청 받을 준비 됐음")

    def start_session(self, session_id: str) -> None:
        self.engines[session_id] = FocusEngine()

    def end_session(self, session_id: str) -> None:
        self.engines.pop(session_id, None)  # 메모리 누수 방지 — 세션 끝나면 꼭 지워야 함

    def process_frame(self, session_id: str, image_bgr: np.ndarray, timestamp: Optional[float] = None) -> dict:
        if session_id not in self.engines:
            self.engines[session_id] = FocusEngine()  # start_session을 안 불렀어도 방어적으로 생성

        engine = self.engines[session_id]
        results = self.model(image_bgr, verbose=False)
        book_boxes = get_boxes_by_class(results, CLASS_OPEN_BOOK)
        hand_boxes = get_boxes_by_class(results, CLASS_HAND)

        ts = timestamp if timestamp is not None else time.time()
        return engine.update(ts, book_boxes, hand_boxes)


_service: FocusYoloService | None = None


def get_focus_service(model_path: str | None = None) -> FocusYoloService:
    """이미 로드돼 있으면 그대로 반환, 처음이면 그때 한 번만 로드."""
    global _service
    if _service is None:
        _service = FocusYoloService(model_path or MODEL_PATH)
    return _service