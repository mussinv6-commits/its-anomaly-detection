"""
FastAPI 서버
- PC 웹캠/영상 업로드를 받아 파이프라인 실행
- 이상탐지 결과 조회 API 제공
- 대시보드(static/dashboard.html)에 데이터 제공
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import shutil

from db import init_db, get_recent_anomalies

app = FastAPI(title="ITS 이상탐지 API")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    init_db()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """PC에서 영상 파일을 업로드하면 저장 후 파이프라인 실행 대상으로 등록한다."""
    save_path = f"data/raw/{file.filename}"
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
            "source": r.source,
            "detected_at": r.detected_at.isoformat(),
        }
        for r in records
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


# 실행: uvicorn src.api:app --reload --port 8000
