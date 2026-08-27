"""
요약 모델(KoBART, [GENRE]/[KW]/[CTX] 포맷으로 학습됨)을 한 번만 로드해두고
재사용하는 모듈.

주의: 이 모델은 학습 때 "[GENRE] ... [KW] ... [CTX] ..." 형태로만 입력을 봤다.
원문 텍스트를 그대로 넣으면 학습 때 한 번도 못 본 형식이라 품질이 떨어진다 —
그래서 여기서도 학습 때와 똑같이 키워드 추출 + 포맷팅을 거친 뒤 모델에 넣는다.
(원본 로직: final_summary.py Cell 14의 summarize_thread 함수)
"""

import os

import torch
from keybert import KeyBERT
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BartForConditionalGeneration

# model_weights/ 통합 폴더 기준 상대경로
MODEL_DIR = "model_weights/summary/kobart_keybert_summary_v1"
KEYWORD_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"  # final_summary.py Cell 6과 동일
MAX_INPUT_LEN = 256  # 학습 때 쓴 값(final_summary.py Cell 9)과 반드시 같아야 함


class SummaryService:
    """KoBART 요약 모델 + 키워드 추출기를 한 번만 로드해서 재사용."""

    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SummaryService] 장치: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = BartForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        print(f"[SummaryService] 요약 모델 로드 완료: {model_dir}")

        # 학습 때 쓴 것과 동일한 키워드 추출기 — 여기가 없으면 학습 때 입력 형식을 재현 못 함
        sbert = SentenceTransformer(KEYWORD_MODEL_NAME)
        self.kw_model = KeyBERT(model=sbert)
        self.okt = Okt()
        print("[SummaryService] 키워드 추출기(KeyBERT+Okt) 준비 완료 — 요청 받을 준비 됐음")

    def _extract_keywords(self, text: str, top_n: int = 3) -> list[str]:
        nouns = list({n for n in self.okt.nouns(text) if len(n) > 1})
        if not nouns:
            return []
        keywords = self.kw_model.extract_keywords(
            text, candidates=nouns, top_n=top_n, use_mmr=True, diversity=0.5
        )
        return [kw for kw, _ in keywords]

    def _safe_decode(self, token_ids: list[int]) -> str:
        vocab_size = len(self.tokenizer)
        clipped = [tid if 0 <= tid < vocab_size else self.tokenizer.pad_token_id for tid in token_ids]
        return self.tokenizer.decode(clipped, skip_special_tokens=True)

    @torch.no_grad()
    def summarize(self, text: str, genre: str = "literary", num_beams: int = 4, max_length: int = 64) -> dict:
        if not text or not text.strip():
            return {"error": "text가 비어있어"}
        if len(text.strip()) < 100:
            # 원본 summaryapp.py와 동일한 정책: 너무 짧으면 요약 없이 원문 그대로
            return {"summary": text.strip(), "keywords": []}

        keywords = self._extract_keywords(text)
        kw_str = ", ".join(keywords)
        # 학습 때(final_summary.py Cell 7)와 반드시 동일한 포맷
        model_input = f"[GENRE] {genre} [KW] {kw_str} [CTX] {text}"

        inputs = self.tokenizer(
            model_input, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs, max_length=max_length, num_beams=num_beams,
            no_repeat_ngram_size=3, early_stopping=True,
        )
        summary = self._safe_decode(output_ids[0].tolist())
        return {"summary": summary, "keywords": keywords}


_service: SummaryService | None = None


def get_summary_service(model_dir: str | None = None) -> SummaryService:
    """이미 로드돼 있으면 그대로 반환, 처음이면 그때 한 번만 로드."""
    global _service
    if _service is None:
        if model_dir is None:
            model_dir = MODEL_DIR  # 프로젝트 루트(uvicorn 실행 위치) 기준 상대경로
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"요약 모델 폴더가 없습니다: {model_dir}\n"
                f"팀원한테 받은 kobart_keybert_summary_v1/ 폴더(model.safetensors 등 5개 파일)를 "
                f"이 경로에 넣어줘."
            )
        _service = SummaryService(model_dir)
    return _service