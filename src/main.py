"""
전체 파이프라인 실행 (영상 단위)
사용법: python src/main.py --video data/raw/sample.mp4
"""

import argparse
import cv2

from detect import VehicleDetector
from tracker import SimpleTracker
from anomaly import AnomalyDetector
from db import init_db, save_anomaly


def run(video_path: str, meters_per_pixel: float = 0.05):
    detector = VehicleDetector()
    tracker = SimpleTracker()
    anomaly_detector = AnomalyDetector()
    init_db()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_speeds = {}
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)

        for track in tracks:
            result = anomaly_detector.evaluate(
                track, fps, meters_per_pixel, prev_speeds.get(track["track_id"], 0.0)
            )
            prev_speeds[track["track_id"]] = result["speed_kmh"]

            if result["flags"]:
                print(f"[frame {frame_idx}] track {result['track_id']}: {result['flags']} ({result['speed_kmh']:.1f}km/h)")
                save_anomaly(
                    track_id=result["track_id"],
                    flags=result["flags"],
                    speed_kmh=result["speed_kmh"],
                    source=video_path,
                )

        frame_idx += 1

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="입력 영상 경로")
    args = parser.parse_args()
    run(args.video)
