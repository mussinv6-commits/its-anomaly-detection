# ITS 프로젝트 작업 로그

## 09-02 완료한 것

| 항목 | 내용 |
|---|---|
| DB 구조 리팩터링 | `track_id`를 진짜 외래키(FK)로 연결 — `Track` 원본 테이블 신설, `detection_records`/`flow_features`/`anomaly_records`가 이를 참조 |
| 학습형 이상탐지 준비 | Isolation Forest 모듈(`ml_anomaly.py`), 학습 스크립트(`train_isolation_forest.py`) 추가 |
| 3단계: 차량 추적 로직 | IOU 전역 최적 매칭 + 클래스 일치 검증으로 고도화 (ID 스위칭 방지), 합성 데이터로 단위테스트 작성·검증(PASS) |
| 4단계: 파이프라인 연결 | `main.py`가 Track 생성 → flow_features 실시간 적재 → 이상 시 anomaly_records 기록까지 새 DB 구조에 맞게 완성 (영상 입력만 있으면 실행 가능) |
| 5단계: 이상탐지 규칙 검증 | 과속·역주행·급정거·불법정차 4개 규칙을 합성 데이터로 단위테스트 — 7/7 케이스 PASS |
| 8단계: 대시보드·API | `api.py`를 새 DB 구조에 맞게 수정, `/detections` `/stats` 엔드포인트 추가, 대시보드에 차종별 통계·검출결과 실시간 반영 확인 |
| 검출 정확도 개선 | YOLOv8n → YOLOv8s로 모델 교체 (신뢰도 낮은 문제 대응) |
| UI 개선 | 대시보드 시간 표시를 `YYYY-MM-DD HH:MM:SS` 형식으로 정리 |
| 기획 문서 | PRD를 RICE 스코어링·검증계획 포함한 실무형으로 재작성, 공식 통계(도로교통공단)로 배경 데이터 보강 |
| 형상관리 | 오늘 변경사항 전체 GitHub push 완료 |

---

## 09-01 완료한 것

| 항목 | 내용 |
|---|---|
| 개발 환경 | Python 3.11 가상환경(`its`) 구축, requirements.txt 설치 |
| 1단계: 데이터 수집 | Kaggle `traffic-vehicles-object-detection` 다운로드, `data/raw`로 정리 (이미지 823장) |
| 2단계: 차량 검출 | YOLOv8(`detect.py`) 동작 확인 — 이미지 1장에서 차량 3대 검출 성공 |
| DB 구축 | SQLite → PostgreSQL(`its` DB) 전환 |
| DB 데이터 적재 | 전체 이미지 검출 실행 → 실제 검출 결과 5,712건 저장 (`populate_db.py`) |
| 번호판 로직 | 한국 번호판 형식(숫자2~3+한글1+숫자4) 정규식 검증 추가 (`ocr.py`) |
| 형상관리 | 로컬 git 저장소 → GitHub 새 저장소(`mussinv6-commits/its-anomaly-detection`) 생성 및 push |

---

## 내일 할 것

| 순서 | 작업 | 파일 |
|---|---|---|
| 1 | 실제 영상(video) 확보 후 `main.py`로 4단계 파이프라인 첫 실행 — 진짜 속도·이상탐지 데이터 생성 | `data/raw/*.mp4` |
| 2 | `populate_flow_features_demo.py` 대신 실제 영상 기반 `flow_features`로 `train_isolation_forest.py` 재학습 | 6단계 |
| 3 | 번호판 인식(`ocr.py`) 실사진 테스트 | 7단계 |

---

## 다음에 추가할 것 (백로그)

| 순서 | 작업 | 비고 |
|---|---|---|
| 6단계 | 학습형 이상탐지 고도화 (Isolation Forest 실데이터 학습) | 실제 영상 확보 후 진행 가능 |
| 7단계 | 번호판 인식 실사진 테스트 | 지금은 정규식 로직만 완성, 실제 이미지 검증 필요 |
| 9단계 | 테스트 및 발표자료 정리 | 검출 통계, 정확도 수치 포함 |
| 부가 | requirements.txt 버전 고정 재정비 | 지금은 버전 미고정 상태, 안정화 필요 |
| 부가 | README에 트러블슈팅 과정 요약 추가 | 발표자료/면접 스토리로 활용 가능 |
| 부가 | Alembic 마이그레이션 도구 도입 | DB 구조 변경 이력 관리 |
| 부가 | 검출 신뢰도 추가 개선 검토 (yolov8m 또는 자체 데이터 파인튜닝) | 지금 yolov8s로 1차 개선함 |

