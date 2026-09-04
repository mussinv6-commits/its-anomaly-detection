"""
inspect_plate_classes.py로 확인한 "번호판" 클래스 번호를 받아서,
그 클래스만 골라내 단일 클래스(번호판 전용) YOLO 학습 데이터셋을 만든다.

사용법:
    python build_plate_dataset.py --plate-class-id 4

    (blur_number_plate처럼 번호판 클래스가 2개 이상이면 콤마로 여러 개 지정 가능)
    python build_plate_dataset.py --plate-class-id 4,5
"""

import argparse
import random
import shutil
from pathlib import Path

LABELS_DIR = Path("data/raw/labels")
IMAGES_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("data/plate_yolo")
VAL_RATIO = 0.15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plate-class-id", required=True,
        help="inspect_plate_classes.py로 확인한 번호판 클래스 번호 (콤마로 여러 개 가능, 예: 4 또는 4,5)",
    )
    args = parser.parse_args()
    plate_ids = {int(x) for x in args.plate_class_id.split(",")}

    if not LABELS_DIR.exists():
        print(f"라벨 폴더를 찾을 수 없습니다: {LABELS_DIR}")
        return

    # 출력 폴더 구조 준비
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    label_files = list(LABELS_DIR.glob("*.txt"))
    random.seed(42)
    random.shuffle(label_files)
    val_count = int(len(label_files) * VAL_RATIO)

    stats = {"train": {"images": 0, "boxes": 0}, "val": {"images": 0, "boxes": 0}}

    for i, label_path in enumerate(label_files):
        image_path = IMAGES_DIR / (label_path.stem + ".jpg")
        if not image_path.exists():
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        # 번호판 클래스만 골라서, 단일 클래스(0번)로 다시 라벨링
        plate_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if int(parts[0]) in plate_ids:
                plate_lines.append("0 " + " ".join(parts[1:5]))

        if not plate_lines:
            continue  # 이 이미지엔 번호판이 없으면 학습 데이터에서 제외

        split = "val" if i < val_count else "train"
        shutil.copy2(image_path, OUTPUT_DIR / "images" / split / image_path.name)
        with open(OUTPUT_DIR / "labels" / split / label_path.name, "w") as f:
            f.write("\n".join(plate_lines) + "\n")

        stats[split]["images"] += 1
        stats[split]["boxes"] += len(plate_lines)

    # YOLO 학습에 필요한 data.yaml 생성
    yaml_content = f"""\
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val
names:
  0: plate
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"train: 이미지 {stats['train']['images']}장, 번호판 박스 {stats['train']['boxes']}개")
    print(f"val:   이미지 {stats['val']['images']}장, 번호판 박스 {stats['val']['boxes']}개")
    print(f"\n데이터셋 구성 완료: {OUTPUT_DIR}/")
    print("다음: python train_plate_detector.py")


if __name__ == "__main__":
    main()
