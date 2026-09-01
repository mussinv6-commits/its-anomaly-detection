"""
data/raw/images의 모든 이미지에 대해 차량 검출을 실행하고,
결과를 PostgreSQL의 detection_records 테이블에 저장한다.

사용법:
    set ITS_DB_PASSWORD=본인비밀번호
    python populate_db.py [--limit 100]
"""

import argparse
import sys
from pathlib import Path

sys.path.append("src")

import cv2
from detect import VehicleDetector
from db import init_db, save_detection, get_or_create_track


def main(limit: int | None):
    init_db()
    detector = VehicleDetector()

    image_paths = sorted(Path("data/raw/images").glob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]

    total_images = len(image_paths)
    total_detections = 0

    for i, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[{i}/{total_images}] 읽기 실패: {image_path}")
            continue

        track_id = get_or_create_track(source=str(image_path))

        results = detector.detect(frame)
        for r in results:
            save_detection(
                image_path=str(image_path),
                cls=r["cls"],
                conf=r["conf"],
                bbox=r["bbox"],
                track_id=track_id,
            )
        total_detections += len(results)

        if i % 20 == 0 or i == total_images:
            print(f"[{i}/{total_images}] 진행 중... (누적 검출: {total_detections}건)")

    print(f"\n완료: 이미지 {total_images}장 처리, 총 검출 {total_detections}건을 DB에 저장했습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="처리할 이미지 수 제한 (테스트용)")
    args = parser.parse_args()
    main(args.limit)
