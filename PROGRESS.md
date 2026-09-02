# ITS 프로젝트 작업 로그

## 09-02 완료한 것

| 항목 | 내용 |
|---|---|
| DB 구조 리팩터링 | `track_id`를 진짜 외래키(FK)로 연결 — `Track` 원본 테이블 신설, `detection_records`/`flow_features`/`anomaly_records`가 이를 참조 |
| 학습형 이상탐지 준비 | Isolation Forest 모듈(`ml_anomaly.py`), 학습 스크립트(`train_isolation_forest.py`) 추가 |
| 3단계: 차량 추적 로직 | IOU 전역 최적 매칭 + 클래스 일치 검증으로 고도화 (ID 스위칭 방지), 합성 데이터로 단위테스트 작성·검증(PASS) |
| 기획 문서 | PRD를 RICE 스코어링·검증계획 포함한 실무형으로 재작성, 공식 통계(도로교통공단)로 배경 데이터 보강 |

---

## 09-01 완료한 것

| 항목 | 내용 |
|---|---|
| 개발 환경 | Python 3.11 가상환경(`its`) 구축, requirements.txt 설치 |
| 1단계: 데이터 수집 | Kaggle `traffic-vehicles-object-detection` 다운로드, `data/raw`로 정리 (이미지 1183장) |
| 2단계: 차량 검출 | YOLOv8(`detect.py`) 동작 확인 — 이미지 1장에서 차량 3대 검출 성공 |
| DB 구축 | SQLite → PostgreSQL(`its` DB) 전환 |
| DB 데이터 적재 | 전체 이미지 검출 실행 → 실제 검출 결과 5,712건 저장 (`populate_db.py`) |
| 번호판 로직 | 한국 번호판 형식(숫자2~3+한글1+숫자4) 정규식 검증 추가 (`ocr.py`) |
| 형상관리 | 로컬 git 저장소 → GitHub 새 저장소(`mussinv6-commits/its-anomaly-detection`) 생성 및 push |

---

## 다음 할 것

| 순서 | 작업 | 파일 |
|---|---|---|
| 1 | 4단계: 이동 벡터/속도 산출 — 실제 영상(video)으로 main.py 파이프라인 테스트 | `src/anomaly.py`, `src/main.py` |
| 2 | populate_db.py/populate_flow_features_demo.py를 새 FK 구조로 재실행, pgAdmin에서 JOIN 확인 | DB |
| 3 | (여유되면) git 커밋 이메일 설정 확인하고 계속 커밋/push 습관화 | — |

---

## 다음에 추가할 것 (백로그)

| 순서 | 작업 | 비고 |
|---|---|---|
| 5단계 | 이상탐지 규칙 설계 (정체·역주행·급정거·불법정차·과속) | 프로젝트 핵심 차별점 |
| 6단계 | 학습형 이상탐지 고도화 (Isolation Forest 실데이터 학습) | 실제 영상 확보 후 진행 가능 |
| 7단계 | 번호판 인식 실사진 테스트 | 지금은 정규식 로직만 완성, 실제 이미지 검증 필요 |
| 8단계 | 대시보드 + PC 연동 (`api.py`, `dashboard.html`) | FastAPI 서버 실행 테스트 |
| 9단계 | 테스트 및 발표자료 정리 | 검출 통계, 정확도 수치 포함 |
| 부가 | requirements.txt 버전 고정 재정비 | 지금은 버전 미고정 상태, 안정화 필요 |
| 부가 | README에 트러블슈팅 과정 요약 추가 | 발표자료/면접 스토리로 활용 가능 |
| 부가 | Alembic 마이그레이션 도구 도입 | DB 구조 변경 이력 관리 |

