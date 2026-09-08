"""
우리가 가진 실제 영상(1~5.mp4)에서 번호판이 확실하게 검출된 프레임을 뽑아
번호판 검출 모델의 학습 데이터에 추가한다 (준지도학습/self-training 방식).

동작 방식:
1. 각 영상을 N프레임 간격으로 훑으며 차량을 검출한다.
2. 검출된 차량 crop마다, 지금 학습된 번호판 검출 모델(plate_detector.pt)을 돌려본다.
3. confidence가 기준치(기본 0.5) 이상으로 확실하게 번호판을 찾은 경우에만
   그 차량 crop 이미지 + 번호판 위치(YOLO 라벨)를 저장한다.
4. 기존 data/plate_yolo/ 학습 데이터셋에 이어붙인다 (train/val 비율 유지).

주의: 이건 사람이 직접 확인한 라벨이 아니라 "모델이 확신한" 라벨이다(pseudo-label).
확신도 기준을 높게(0.5) 잡아서 품질을 최대한 보장하지만, 완벽하지 않을 수 있다.

사용법:
    python extract_plate_frames_from_videos.py --videos 1.mp4 2.mp4 3.mp4 4.mp4 5.mp4
    python extract_plate_frames_from_videos.py --videos 5.mp4 --conf 0.6 --frame-step 3
"""

import argparse
import os
import random
from pathlib import Path

import cv2

import sys
sys.path.append("src")

from detect import VehicleDetector
from plate_detect import PlateDetector

OUTPUT_DIR = Path("data/plate_yolo")
VAL_RATIO = 0.15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True, help="번호판 프레임을 추출할 영상들")
    parser.add_argument("--conf", type=float, default=0.5, help="이 확신도 이상인 검출만 학습 데이터로 채택")
    parser.add_argument("--frame-step", type=int, default=5, help="N프레임마다 한 번씩 검사 (전부 다 보면 너무 느리고 중복이 많음)")
    args = parser.parse_args()

    detector = VehicleDetector()
    plate_detector = PlateDetector(conf_threshold=args.conf)

    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for video_path in args.videos:
        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem
        frame_idx = 0
        saved_for_this_video = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % args.frame_step == 0:
                detections = detector.detect(frame)
                for det_idx, det in enumerate(detections):
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    x1, y1 = max(0, x1), max(0, y1)
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0:
                        continue

                    plate_box = plate_detector.detect(vehicle_crop)
                    if plate_box is None:
                        continue

                    # YOLO 포맷으로 변환 (vehicle_crop 기준 정규화 좌표)
                    ch, cw = vehicle_crop.shape[:2]
                    px1, py1, px2, py2 = plate_box
                    xc = ((px1 + px2) / 2) / cw
                    yc = ((py1 + py2) / 2) / ch
                    bw = (px2 - px1) / cw
                    bh = (py2 - py1) / ch

                    split = "val" if random.random() < VAL_RATIO else "train"
                    filename = f"{video_name}_f{frame_idx}_d{det_idx}"
                    img_path = OUTPUT_DIR / "images" / split / f"{filename}.jpg"
                    label_path = OUTPUT_DIR / "labels" / split / f"{filename}.txt"

                    cv2.imwrite(str(img_path), vehicle_crop)
                    with open(label_path, "w") as f:
                        f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

                    saved_for_this_video += 1
                    total_saved += 1

            frame_idx += 1

        cap.release()
        print(f"{video_path}: {saved_for_this_video}건 추가 (총 {frame_idx}프레임 중 {frame_idx // args.frame_step}프레임 검사)")

    print(f"\n총 {total_saved}건의 새 학습 데이터가 {OUTPUT_DIR}에 추가되었습니다.")
    print("다음: python train_plate_detector.py --epochs 100  (새 데이터 포함해서 처음부터 재학습)")


if __name__ == "__main__":
    main()
