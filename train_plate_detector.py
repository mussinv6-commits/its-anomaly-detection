"""
build_plate_dataset.py로 만든 번호판 전용 데이터셋으로 YOLOv8n을 파인튜닝한다.

사용법:
    python train_plate_detector.py                      (처음부터 새로 학습, 50 epoch)
    python train_plate_detector.py --epochs 100          (처음부터 새로 학습, 100 epoch)
    python train_plate_detector.py --resume --epochs 100 (중단된 지점에서 이어서 학습, 총 100 epoch까지)

--resume을 쓰면 runs/plate_train/exp/weights/last.pt(마지막 중단 시점 가중치)에서
이어서 학습한다. Ctrl+C로 중단했던 학습을 처음부터 다시 돌리지 않아도 된다.
주의: --resume 시 --epochs는 "총 목표 epoch"를 의미한다 (예: 21epoch까지 하고 중단했는데
--epochs 100이면, 나머지 79epoch만 더 진행됨).
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data/plate_yolo/data.yaml")
OUTPUT_MODEL = Path("models/plate_detector.pt")
# Ultralytics가 실행 위치에 따라 runs/detect/runs/... 처럼 detect 폴더를 한 번 더 만드는 경우가 있어
# 두 경로 다 확인해서 실제로 존재하는 쪽을 쓴다.
LAST_CHECKPOINT_CANDIDATES = [
    Path("runs/plate_train/exp/weights/last.pt"),
    Path("runs/detect/runs/plate_train/exp/weights/last.pt"),
]


def find_last_checkpoint():
    for path in LAST_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="목표 총 epoch 수")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--resume", action="store_true",
        help="중단된 학습을 이어서 진행 (runs/plate_train/exp/weights/last.pt에서 이어감)",
    )
    args = parser.parse_args()

    if args.resume:
        last_checkpoint = find_last_checkpoint()
        if last_checkpoint is None:
            print("이어할 체크포인트를 찾을 수 없습니다. 확인한 경로들:")
            for p in LAST_CHECKPOINT_CANDIDATES:
                print(f"  - {p} (존재: {p.exists()})")
            print("--resume 없이 새로 시작하거나, 경로를 확인해주세요.")
            return
        print(f"이전 학습을 이어서 진행합니다: {last_checkpoint} (목표 총 {args.epochs} epoch)")
        model = YOLO(str(last_checkpoint))
        results = model.train(resume=True, epochs=args.epochs)
    else:
        if not DATA_YAML.exists():
            print(f"{DATA_YAML}를 찾을 수 없습니다. 먼저 다음을 실행하세요:")
            print("  1. python inspect_plate_classes.py")
            print("  2. python build_plate_dataset.py --plate-class-id N")
            return

        OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
        print(f"YOLOv8n을 번호판 검출용으로 처음부터 파인튜닝합니다 (epochs={args.epochs})...")
        model = YOLO("yolov8n.pt")  # COCO 사전학습 모델에서 시작 (전이학습)
        results = model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            project="runs/plate_train",
            name="exp",
            exist_ok=True,
        )

    # 학습된 best.pt를 models/plate_detector.pt로 복사
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_path.exists():
        shutil.copy2(best_path, OUTPUT_MODEL)
        print(f"\n학습 완료: {OUTPUT_MODEL} 저장됨")
        print("이제 src/plate_detect.py에서 이 모델을 사용할 수 있습니다.")
    else:
        print(f"경고: {best_path}를 찾지 못했습니다. runs/plate_train/exp/weights/ 를 직접 확인하세요.")


if __name__ == "__main__":
    main()
