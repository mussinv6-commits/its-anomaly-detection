"""
build_plate_dataset.py로 만든 번호판 전용 데이터셋으로 YOLOv8n을 파인튜닝한다.

사용법:
    python train_plate_detector.py
    python train_plate_detector.py --epochs 100   (기본 50)
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data/plate_yolo/data.yaml")
OUTPUT_MODEL = Path("models/plate_detector.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print(f"{DATA_YAML}를 찾을 수 없습니다. 먼저 다음을 실행하세요:")
        print("  1. python inspect_plate_classes.py")
        print("  2. python build_plate_dataset.py --plate-class-id N")
        return

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    print(f"YOLOv8n을 번호판 검출용으로 파인튜닝합니다 (epochs={args.epochs})...")
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
