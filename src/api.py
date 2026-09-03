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
import random

# 이 파일(api.py)이 있는 폴더(src)를 import 경로에 추가 → 어디서 실행해도 db.py를 찾을 수 있음
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.append(THIS_DIR)

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles

from db import (
    init_db, get_recent_anomalies, get_recent_detections, get_ocr_stats,
    get_recent_ocr_attempts, create_track, save_anomaly, get_anomaly_type_stats,
    save_speed_prediction, get_prediction_stats, get_recent_predictions,
)

app = FastAPI(title="ITS 이상탐지 API")

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# 업로드된 영상이 지금 처리 중인지 추적 (동시에 여러 개 처리 요청이 겹치는 걸 방지)
processing_status = {"is_processing": False, "current_file": None}


@app.on_event("startup")
def on_startup():
    init_db()


def run_pipeline_in_background(video_path: str):
    """백그라운드에서 실제 검출·추적·이상탐지 파이프라인을 실행한다."""
    processing_status["is_processing"] = True
    processing_status["current_file"] = os.path.basename(video_path)
    try:
        from main import run
        run(video_path)
    finally:
        processing_status["is_processing"] = False
        processing_status["current_file"] = None


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    PC에서 영상 파일을 업로드하면 저장하고, 백그라운드에서 바로 파이프라인을 실행한다.
    업로드 응답은 즉시 오지만, 실제 처리(검출·추적·이상탐지)는 뒤에서 계속 진행되며
    끝나는 대로 결과가 DB에 쌓여 대시보드에 순차적으로 나타난다.
    """
    save_path = os.path.join(PROJECT_ROOT, "data", "raw", file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_pipeline_in_background, save_path)
    return {"filename": file.filename, "status": "uploaded_and_processing"}


@app.get("/processing-status")
def get_processing_status():
    """지금 백그라운드에서 영상 처리 중인지 확인한다. (대시보드에서 진행 표시용)"""
    return processing_status


@app.post("/demo/anomaly")
def create_demo_anomaly():
    """
    데모/테스트용: 가상의 이상탐지 기록 1건을 즉시 DB에 저장한다.
    POST 요청이 실제로 DB에 데이터를 쓴다는 것을 대시보드에서 바로 확인해볼 수 있게 하는 용도.
    """
    flag_options = [["과속"], ["역주행"], ["급정거"], ["불법정차"], ["과속", "급정거"]]
    track_id = create_track(source="demo/manual_trigger")
    flags = random.choice(flag_options)
    speed = round(random.uniform(10, 100), 1)
    save_anomaly(track_id=track_id, flags=flags, speed_kmh=speed)
    return {"status": "created", "track_id": track_id, "flags": flags, "speed_kmh": speed}


@app.post("/demo/prediction")
def create_demo_prediction():
    """
    데모/테스트용: 실제 학습된 speed_predictor.pkl로 가상의 속도 흐름 하나를 예측해서 저장한다.
    학습된 모델이 없으면 안내 메시지를 반환한다.
    """
    from predict import SpeedPredictor

    predictor = SpeedPredictor()
    try:
        predictor.load()
    except FileNotFoundError:
        return {"status": "no_model", "message": "먼저 train_speed_predictor.py로 모델을 학습해주세요."}

    # 데모용 가상 속도 흐름 (실제로는 main.py가 실시간으로 이 값을 넘겨줌)
    recent_speeds = [round(random.uniform(10, 100), 1) for _ in range(3)]
    result = predictor.predict_sudden_drop_risk(recent_speeds)

    track_id = create_track(source="demo/manual_trigger")
    save_speed_prediction(
        track_id=track_id,
        predicted_speed_kmh=result["predicted_speed"],
        risk_flag=result["risk"],
    )
    return {"status": "created", "recent_speeds": recent_speeds, **result}


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


@app.get("/ocr-stats")
def ocr_stats():
    """번호판 인식 성공/실패 건수와 성공률을 반환한다. (대시보드 요약 카드용)"""
    return get_ocr_stats()


@app.get("/anomaly-stats")
def anomaly_stats():
    """이상 유형(과속/역주행/급정거/불법정차)별 건수를 반환한다. (차트용)"""
    return get_anomaly_type_stats()


@app.get("/prediction-stats")
def prediction_stats():
    """예측 시도 중 위험/정상 건수를 반환한다. (차트용)"""
    return get_prediction_stats()


@app.get("/predictions")
def predictions(limit: int = 20):
    """최근 속도 예측 시도 목록을 반환한다."""
    records = get_recent_predictions(limit)
    return [
        {
            "id": r.id,
            "track_id": r.track_id,
            "predicted_speed_kmh": r.predicted_speed_kmh,
            "risk_flag": r.risk_flag,
            "predicted_at": r.predicted_at.isoformat(),
        }
        for r in records
    ]


@app.get("/ocr-attempts")
def ocr_attempts(limit: int = 20):
    """최근 번호판 인식 시도 목록(성공/실패 여부 포함)을 반환한다."""
    records = get_recent_ocr_attempts(limit)
    return [
        {
            "id": r.id,
            "raw_text": r.raw_text,
            "parsed_plate": r.parsed_plate,
            "success": r.success,
            "attempted_at": r.attempted_at.isoformat(),
        }
        for r in records
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


# 실행: uvicorn src.api:app --reload --port 8000
