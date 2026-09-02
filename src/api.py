"""
FastAPI 서버
- PC 웹캠/영상 업로드를 받아 파이프라인 실행
- 이상탐지 결과 조회 API 제공
- 대시보드(static/dashboard.html)에 데이터 제공

실행 위치와 무관하게 항상 동작하도록, 이 파일 자신의 위치를 기준으로
src 폴더와 static 폴더 경로를 계산한다.
"""

import os
import sys
import shutil

# 이 파일(api.py)이 있는 폴더(src)를 import 경로에 추가 → 어디서 실행해도 db.py를 찾을 수 있음
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.append(THIS_DIR)

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

from db import init_db, get_recent_anomalies, get_recent_detections

app = FastAPI(title="ITS 이상탐지 API")

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """PC에서 영상 파일을 업로드하면 저장 후 파이프라인 실행 대상으로 등록한다."""
    save_path = os.path.join(PROJECT_ROOT, "data", "raw", file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # TODO: main.py의 파이프라인을 백그라운드 태스크로 실행하도록 연결
    return {"filename": file.filename, "status": "uploaded"}


@app.get("/records")
def list_records(limit: int = 50):
    """최근 이상탐지 기록을 조회한다. (대시보드에서 호출)"""
    records = get_recent_anomalies(limit)
    return [
        {
            "id": r.id,
            "track_id": r.track_id,
            "flags": r.flags,
            "speed_kmh": r.speed_kmh,
            "plate_number": r.plate_number,
            "source": r.track.source if r.track else None,  # Track 관계를 통해 조회
            "detected_at": r.detected_at.isoformat(),
        }
        for r in records
    ]


@app.get("/detections")
def list_detections(limit: int = 100):
    """최근 원본 검출 결과를 조회한다 (이상 여부와 무관한 전체 검출)."""
    records = get_recent_detections(limit)
    return [
        {
            "id": r.id,
            "image_path": r.image_path,
            "cls": r.cls,
            "conf": r.conf,
            "detected_at": r.detected_at.isoformat(),
        }
        for r in records
    ]


@app.get("/stats")
def stats():
    """차종(cls)별 검출 건수 통계. 대시보드 요약 카드용."""
    records = get_recent_detections(limit=10000)
    counts = {}
    for r in records:
        counts[r.cls] = counts.get(r.cls, 0) + 1
    cls_name = {2: "승용차", 3: "오토바이", 5: "버스", 7: "트럭"}
    return {cls_name.get(cls, f"cls_{cls}"): count for cls, count in counts.items()}


@app.get("/health")
def health():
    return {"status": "ok"}


# 실행: uvicorn src.api:app --reload --port 8000
