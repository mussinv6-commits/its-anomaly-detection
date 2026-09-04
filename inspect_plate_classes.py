"""
data/raw/labels의 YOLO 라벨 파일에는 클래스가 번호(0,1,2...)로만 적혀있고
이름이 없다. 어떤 번호가 "번호판(number_plate)"인지 직접 확인하기 위해,
클래스 번호별로 샘플 이미지를 몇 장씩 잘라서 저장한다.

사용법: python inspect_plate_classes.py
실행 후 data/class_review/class_0/, class_1/... 폴더를 열어서
번호판이 찍힌 폴더의 번호를 확인하면 된다.
"""

import os
from pathlib import Path
import cv2

LABELS_DIR = Path("data/raw/labels")
IMAGES_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("data/class_review")
SAMPLES_PER_CLASS = 5


def main():
    if not LABELS_DIR.exists():
        print(f"라벨 폴더를 찾을 수 없습니다: {LABELS_DIR}")
        print("data_setup.py 또는 organize_data.py로 Kaggle 데이터를 먼저 받아주세요.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_per_class = {}

    label_files = list(LABELS_DIR.glob("*.txt"))
    print(f"라벨 파일 {len(label_files)}개에서 클래스별 샘플을 수집합니다...")

    for label_path in label_files:
        image_path = IMAGES_DIR / (label_path.stem + ".jpg")
        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])

            if saved_per_class.get(cls_id, 0) >= SAMPLES_PER_CLASS:
                continue

            # YOLO 포맷: class_id x_center y_center width height (모두 0~1 정규화)
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            class_dir = OUTPUT_DIR / f"class_{cls_id}"
            class_dir.mkdir(exist_ok=True)
            idx = saved_per_class.get(cls_id, 0)
            cv2.imwrite(str(class_dir / f"sample_{idx}.jpg"), crop)
            saved_per_class[cls_id] = idx + 1

        if all(v >= SAMPLES_PER_CLASS for v in saved_per_class.values()) and len(saved_per_class) >= 7:
            break  # 클래스 7개(이 데이터셋 기준) 다 채웠으면 충분

    print(f"\n완료. 발견된 클래스 번호: {sorted(saved_per_class.keys())}")
    print(f"각 클래스별 샘플이 {OUTPUT_DIR}/class_N/ 폴더에 저장되었습니다.")
    print("폴더를 하나씩 열어서 번호판이 찍힌 클래스 번호를 확인한 뒤,")
    print("build_plate_dataset.py --plate-class-id N 으로 다음 단계를 진행하세요.")


if __name__ == "__main__":
    main()
