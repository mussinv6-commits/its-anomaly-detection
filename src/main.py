"""
전체 파이프라인 실행 (영상 단위)
사용법: python src/main.py --video data/raw/sample.mp4

이 스크립트가 하는 일:
1. 영상에서 프레임마다 차량을 검출·추적한다.
2. 각 차량(트랙)이 화면에 처음 나타나면 DB에 Track을 하나 생성한다.
3. 매 프레임마다 그 트랙의 이동 특징(속도, 방향)을 flow_features에 저장한다
   → 이 데이터가 나중에 Isolation Forest 학습 재료가 된다.
4. 규칙 기반으로 이상상황이 감지되면 anomaly_records에도 별도로 기록한다.
"""

import argparse
import cv2

from detect import VehicleDetector
from tracker import SimpleTracker
from anomaly import AnomalyDetector
from db import init_db, create_track, save_flow_feature, save_anomaly


def run(video_path: str, meters_per_pixel: float = 0.05):
    detector = VehicleDetector()
    tracker = SimpleTracker()
    anomaly_detector = AnomalyDetector()
    init_db()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_speeds = {}
    frame_idx = 0

    # 로컬 tracker의 임시 track_id -> DB의 진짜 tracks.id 매핑
    # (같은 차량이 여러 프레임에 걸쳐 나타나도 DB Track은 하나만 생성되도록)
    local_to_db_track = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)

        for track in tracks:
            local_id = track["track_id"]

            # 이 차량을 처음 보는 경우, DB에 Track을 새로 생성
            if local_id not in local_to_db_track:
                local_to_db_track[local_id] = create_track(source=video_path)
            db_track_id = local_to_db_track[local_id]

            result = anomaly_detector.evaluate(
                track, fps, meters_per_pixel, prev_speeds.get(local_id, 0.0)
            )
            prev_speeds[local_id] = result["speed_kmh"]

            bbox = track["bbox"]
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # 이상 여부와 무관하게 모든 흐름 데이터를 학습용으로 저장
            save_flow_feature(
                track_id=db_track_id,
                speed_kmh=result["speed_kmh"],
                dx=result["dx"],
                dy=result["dy"],
                bbox_area=bbox_area,
                frame_idx=frame_idx,
            )

            if result["flags"]:
                print(f"[frame {frame_idx}] track {local_id}: {result['flags']} ({result['speed_kmh']:.1f}km/h)")
                save_anomaly(
                    track_id=db_track_id,
                    flags=result["flags"],
                    speed_kmh=result["speed_kmh"],
                )

        frame_idx += 1

    cap.release()
    print(f"\n완료: 총 {frame_idx}프레임 처리, {len(local_to_db_track)}대 차량 추적, DB에 저장됨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="입력 영상 경로")
    args = parser.parse_args()
    run(args.video)

