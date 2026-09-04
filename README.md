# 지능형 도로체계(ITS) — AI 기반 실시간 교통 이상탐지 시스템

차량을 검출·추적해서 정체·역주행·급정거·불법정차·과속을 실시간으로 감지하고,
번호판 인식과 급정거 조기 예측까지 하는 개인 프로젝트.

## 핵심 파이프라인 (src/)

| 파일 | 역할 | 단계 |
|---|---|---|
| `detect.py` | YOLOv8로 프레임에서 차량 검출 (승용차/트럭/오토바이/버스) | 2단계 |
| `tracker.py` | IOU 전역 최적 매칭으로 같은 차량에 고유 ID 부여 (프레임 간 추적) | 3단계 |
| `anomaly.py` | 과속·역주행·급정거·불법정차 판정 (규칙 기반, 양방향 도로 자동 캘리브레이션 포함) | 5단계 |
| `ml_anomaly.py` | Isolation Forest 기반 학습형 이상탐지 (정상 패턴에서 벗어난 흐름을 자동 탐지) | 6단계 |
| `ocr.py` | EasyOCR + 정규식으로 한국 번호판(숫자2~3+한글1+숫자4) 인식 | 7단계 |
| `predict.py` | 최근 3프레임 속도로 다음 속도를 예측, 급정거 사전 경고 | 예측 확장 |
| `db.py` | PostgreSQL 스키마 정의 및 저장/조회 함수 전체 | 전체 |
| `api.py` | FastAPI 서버 — 대시보드용 REST API (GET 조회 + POST 저장) | 8단계 |
| `main.py` | 전체 파이프라인 실행 진입점 (영상 → 검출 → 추적 → 이상탐지 → DB) | 4단계 |
| `data_setup.py` | Kaggle 데이터셋 다운로드 |  |

## 실행/학습 스크립트 (최상위)

| 파일 | 역할 |
|---|---|
| `populate_db.py` | `data/raw` 이미지 전체를 검출해서 DB에 채움 |
| `populate_flow_features_demo.py` | Isolation Forest 학습 파이프라인 검증용 가상 데이터 생성 |
| `populate_anomaly_demo.py` | 대시보드 확인용 가상 이상탐지 데이터 생성 |
| `train_isolation_forest.py` | flow_features로 Isolation Forest 학습 → `models/isolation_forest.pkl` |
| `train_speed_predictor.py` | flow_features로 속도 예측 모델 학습 → `models/speed_predictor.pkl` (MAE 5.56km/h) |
| `batch_compare.py` | 여러 영상을 일괄 처리하고 차량수·평균속도·이상유형을 비교표로 출력 |

## 단위테스트 (최상위, 실제 영상 없이 로직 검증용)

| 파일 | 검증 대상 | 결과 |
|---|---|---|
| `test_tracker.py` | 추적 ID가 프레임 간 유지되는지 | PASS |
| `test_anomaly.py` | 4개 이상탐지 규칙 + 오탐 방지 | 7/7 PASS |
| `test_ocr.py` | 합성 번호판 이미지로 OCR 인식 | 부분 성공 (실전 검증용) |
| `test_predict.py` | 속도 예측 로직 (급정거 위험 판정) | PASS |

## DB 구조 (PostgreSQL)

```
tracks (차량 추적 세션 원본)
 ├─ detection_records  (원본 검출 결과: 클래스, 신뢰도, 좌표)
 ├─ flow_features      (프레임별 속도·이동방향 — 학습 데이터)
 ├─ anomaly_records    (이상탐지 결과: 유형, 속도, 번호판)
 ├─ ocr_attempts        (번호판 인식 시도 기록: 성공/실패)
 └─ speed_predictions   (속도 예측 결과: 예측치, 위험여부)
```
모든 테이블이 `tracks.id`를 외래키(FK)로 참조 — 참조 무결성 보장.

## 대시보드 (static/dashboard.html)
- 신호등 색 체계(정상=초록, 위험=빨강)로 실시간 상태 배너 표시
- Chart.js로 차종별/이상유형별/인식률/예측위험률 시각화
- GET(5초 자동 조회) + POST(버튼으로 즉시 데이터 생성) 둘 다 동작

## 실행 방법

```bash
# 환경 설정 (Python 3.11 권장 — 3.14는 일부 패키지 사전빌드 파일 없음)
conda create -n its python=3.11 -y
conda activate its
pip install -r requirements.txt

# DB 초기화
set ITS_DB_PASSWORD=본인비밀번호
python -c "import sys; sys.path.append('src'); from db import init_db; init_db()"

# 파이프라인 실행 (영상)
cd src
python main.py --video ..\data\raw\sample.mp4

# 서버 실행 (최상위 폴더에서)
cd ..
uvicorn src.api:app --reload --port 8000
# 브라우저: http://localhost:8000/static/dashboard.html
```

## 개발 단계 체크리스트
- [x] 1단계: 데이터 수집 (Kaggle, 823장)
- [x] 2단계: 차량 검출 모델 (YOLOv8s)
- [x] 3단계: 추적 로직 (IOU 전역매칭)
- [x] 4단계: 이동 벡터/속도 산출 (5프레임 스무딩)
- [x] 5단계: 이상탐지 규칙 설계 (7/7 단위테스트 PASS)
- [x] 6단계: 학습형 이상탐지 (Isolation Forest)
- [x] 7단계: 번호판 인식 연동 (실전 검증 진행 중)
- [x] 8단계: 대시보드/API (GET+POST, Chart.js)
- [ ] 9단계: 발표자료 정리
- [x] 부가: 간이 속도 예측 모델 (MAE 5.56km/h)

## Git 커밋 컨벤션
| 접두사 | 용도 |
|---|---|
| `feat:` | 새 기능 추가 |
| `fix:` | 버그 수정 |
| `docs:` | 문서 수정 |
| `refactor:` | 리팩토링 |
| `test:` | 테스트 추가 |
| `chore:` | 설정/잡일 |

## 트러블슈팅 기록
개발 중 겪은 문제와 해결 과정은 `docs/ITS_작업로그_*.docx`에 날짜별로 정리되어 있음.
주요 사례: Python 3.14 사전빌드 이슈, DB 세션 DetachedInstanceError, 양방향 도로 역주행 오탐, UTC/KST 시간대 오차 등.
