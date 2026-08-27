# Model Inference API Server

## 모델 프로젝트 실행을 위한 세팅 (필수)
프로젝트 실행 시 해당 명령어만 실행하시면 됩니다. 

1. 기본 세팅 
python -m venv venv
venv\Scripts\activate          (윈도우)
pip install -r requirements.txt 

2. 모델 파일 가져오기 (직접)
- 모델 파일은 팀 공용 구글 드라이브의 models_all 폴더로 올려놓겠습니다. 
- 폴더를 받아와 아래 폴더 구조를 참고하여 위치시켜주세요. 
(프로젝트 최상위 폴더 아래 필요한 폴더를 생성) 

[모델 폴더 구조]
gaze-model/                           ←  최상위
└── model_weights/                    ←  통합 폴더
    ├── gaze/
    │   └── best_model.pt
    ├── summary/
    │   └── kobart_keybert_summary_v1/
    │       ├── model.safetensors
    │       ├── config.json
    │       └── ...
    └── focus_yolo/                   
        └── best.pt

3. Run Server 
```bash
uvicorn inference.api_server:app --host 0.0.0.0 --port 8000
```

성공 시 다음과 같이 출력됩니다.
[GazeService] 장치: cuda
[GazeService] 체크포인트 로드 완료: checkpoints/best_model.pt (학습 시 검증 오차: 4.69cm)
[GazeService] MediaPipe 준비 완료 — 모델 로딩 끝, 요청 받을 준비 됐음


4. 테스트
테스트 방법입니다. 
- http://localhost:8000/predict (postman 모델 api 테스트)
- http://localhost:8000/docs (swagger 테스트 방법)

/gaze/predict는 POST 요청만 받고 다른 형식의 요청 (GET 등을 보내면 에러가 발생할 수 있으므로 스웨거 등에서 테스트 시 첨부 파일을 포함한 POST 요청을 보내주세요. )
테스트 시 직접 인물 이미지 파일(.jpg, .png 등)을 첨부합니다.

- 실행 중 에러 발생 시 반드시 전달해주세요. 



5. 주의 
- 설치 파일 중 konlpy는 Java(JDK)가 깔려있어야 동작하므로 Java 설치가 필요합니다. (이미 깔려 있다면 생략)


----------------------------------------------------
## 1. 환경 설치 - 여기부터는 학습 과정이므로 생략

python -m venv venv
venv\Scripts\activate          (윈도우)
pip install -r requirements.txt

```bash
pip install -r requirements.txt (위 명령어 3줄 실행)
```

## 2. MediaPipe 모델 파일 받기 (최초 1회)

최신 MediaPipe는 예전 `mp.solutions.face_mesh` 방식 대신, 아래처럼
모델 파일을 한 번 받아서 쓰는 **Tasks API**로 바뀌었어:

```bash
mkdir -p models/assets
curl -L -o models/assets/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

> 이 명령은 이 개발 환경(샌드박스)에서는 방화벽 때문에 막혀서 직접 실행/검증은
> 못 했어(403 응답 확인함). 너희 개발 PC에서는 정상적으로 받아질 거야 — 받은 뒤
> `python3 -c "from data.eye_extractor import EyeExtractor; EyeExtractor()"` 로
> 에러 없이 생성되는지만 확인해줘.

## 3. 데이터 전처리

GazeCapture 등에서 받은 원본 이미지 + 정답 좌표(csv)를 준비한 다음:

```bash
python3 prepare_data.py \
  --raw_dir /path/to/raw/images \
  --raw_labels /path/to/raw/labels.csv \
  --out_dir ./processed
```

`raw_labels.csv`는 `image_path,gaze_x_cm,gaze_y_cm` 형식이어야 해. 실행하고 나면
`processed/labels.csv` + `processed/left,right/*.png`가 생기고, 이게 학습 입력이 돼.

## 4. 모델 구조 확인 (학습 전 마지막 점검)

```bash
python3 models/gaze_model.py
```

정상 출력:  `출력 shape: torch.Size([4, 2])`



