# 지능형 도로체계(ITS) - AI 기반 실시간 교통 이상탐지 시스템

## 폴더 구조
```
its_project/
├── data/
│   ├── raw/              # 원본 영상/이미지
│   └── processed/        # 전처리 데이터
├── models/                # 학습된 모델 가중치
├── src/
│   ├── detect.py          # 차량 검출 (YOLO)
│   ├── tracker.py         # 차량 추적 (ByteTrack)
│   ├── anomaly.py         # 이상탐지 로직 (정체/역주행/급정거/과속)
│   ├── ocr.py               # 번호판 인식 (부가 기능)
│   ├── db.py                # 결과 저장/조회 (SQLite)
│   ├── api.py                # FastAPI 서버 (업로드/조회/PC 연동)
│   └── main.py                # 파이프라인 실행 진입점
├── static/
│   └── dashboard.html         # 실시간 시각화 대시보드
├── requirements.txt
└── README.md
```

## 개발 단계 체크리스트
- [ ] 1단계: 데이터 수집 (`data/raw/`)
- [ ] 2단계: 차량 검출 모델 (`src/detect.py`)
- [ ] 3단계: 추적 로직 (`src/tracker.py`)
- [ ] 4단계: 이동 벡터/속도 산출
- [ ] 5단계: 이상탐지 규칙 설계 (`src/anomaly.py`)
- [ ] 6단계: 통계 기반 고도화
- [ ] 7단계: 번호판 인식 연동 (`src/ocr.py`)
- [ ] 8단계: 대시보드/PC 연동 (`src/api.py`, `static/dashboard.html`)
- [ ] 9단계: 테스트/발표자료 정리

## 실행 방법
```bash
python -m venv venv
source venv/bin/activate   # Windows는 venv\Scripts\activate
pip install -r requirements.txt

python src/main.py --video data/raw/sample.mp4
```

## Git 사용 가이드

### 최초 세팅 (로컬에서)
```bash
git init
git add .
git commit -m "chore: 프로젝트 초기 구조 세팅"
```

### GitHub 원격 저장소 연결
1. GitHub에서 새 저장소 생성 (예: `its-anomaly-detection`)
2. 로컬에서 연결:
```bash
git remote add origin https://github.com/<본인아이디>/its-anomaly-detection.git
git branch -M main
git push -u origin main
```

### 커밋 컨벤션 (권장)
| 접두사 | 용도 |
|---|---|
| `feat:` | 새 기능 추가 |
| `fix:` | 버그 수정 |
| `docs:` | 문서 수정 |
| `refactor:` | 리팩토링 |
| `chore:` | 설정/잡일 |

### 단계별 커밋 흐름 예시
```bash
git checkout -b feature/detect
# detect.py 작업
git add src/detect.py
git commit -m "feat: YOLO 차량 검출 모듈 구현"
git checkout main
git merge feature/detect
```

이력서/포트폴리오에는 커밋 로그가 "단계별로 꾸준히 개발했다"는 근거가 되므로, 한 번에 몰아서 커밋하지 말고 기능 단위로 나눠서 커밋하는 걸 추천해요.
