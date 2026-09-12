"""
AI Hub "자동차 차종/연식/번호판 인식용 영상" 데이터셋(TS_원본이미지 + TL_라벨json)을
번호판 검출 모델(YOLO) 학습에 바로 쓸 수 있는 형식으로 변환한다.

원본 구조:
    TS_.../파일명.jpg   (원본 이미지, 3840x2160)
    TL_.../파일명.json  (라벨, license_plate[].bbox = [x, y, width, height])

변환 후:
    data/plate_yolo/images/train(or val)/파일명.jpg
    data/plate_yolo/labels/train(or val)/파일명.txt  (YOLO 포맷: class x_center y_center width height, 0~1 정규화)

사용법:
    python convert_aihub_dataset.py \\
        --ts-dir "data/aihub_raw/TS_차량번호판인식_교차로_[cr01]비산사거리_03번" \\
        --tl-dir "data/aihub_raw/TL_차량번호판인식_교차로_[cr01]비산사거리_03번"
"""

import argparse
import json
import random
import shutil
from pathlib import Path

OUTPUT_DIR = Path("data/plate_yolo")
VAL_RATIO = 0.15


def find_json_for_image(image_path: Path, tl_dir: Path) -> Path | None:
    """이미지 파일명과 같은 이름의 json을 TL 폴더(하위 폴더 포함)에서 찾는다."""
    json_name = image_path.stem + ".json"
    matches = list(tl_dir.rglob(json_name))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts-dir", required=True, help="원본 이미지(TS_) 폴더 경로")
    parser.add_argument("--tl-dir", required=True, help="라벨(TL_) 폴더 경로")
    args = parser.parse_args()

    ts_dir = Path(args.ts_dir)
    tl_dir = Path(args.tl_dir)

    if not ts_dir.exists():
        print(f"TS 폴더를 찾을 수 없습니다: {ts_dir}")
        return
    if not tl_dir.exists():
        print(f"TL 폴더를 찾을 수 없습니다: {tl_dir}")
        return

    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_paths = list(ts_dir.rglob("*.jpg")) + list(ts_dir.rglob("*.jpeg"))
    print(f"원본 이미지 {len(image_paths)}장 발견. 라벨 매칭 및 변환을 시작합니다...")

    random.seed(42)
    random.shuffle(image_paths)
    val_count = int(len(image_paths) * VAL_RATIO)

    converted, skipped_no_label, skipped_no_plate = 0, 0, 0

    for i, image_path in enumerate(image_paths):
        json_path = find_json_for_image(image_path, tl_dir)
        if json_path is None:
            skipped_no_label += 1
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 이미지 실제 해상도 (raw_data_info에 있음, 없으면 기본값 3840x2160 가정)
        resolution = data.get("Raw_Data_Info", {}).get("resolution", "3840, 2160")
        img_w, img_h = [int(v.strip()) for v in resolution.split(",")]

        annotations = data.get("Learning_Data_Info", {}).get("annotations", [])
        yolo_lines = []
        for ann in annotations:
            for plate in ann.get("license_plate", []):
                x, y, w, h = plate["bbox"]
                # YOLO 포맷으로 변환: 중심좌표+크기, 0~1 정규화
                xc = (x + w / 2) / img_w
                yc = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        if not yolo_lines:
            skipped_no_plate += 1
            continue

        split = "val" if i < val_count else "train"
        # 파일명이 서로 다른 하위 폴더(다른 시간대/영상)에서 겹칠 수 있어서,
        # 그냥 원본 파일명만 쓰면 뒤에 처리된 파일이 앞의 파일을 덮어써 데이터가 유실된다.
        # 순번(i)을 접두어로 붙여 항상 고유한 파일명이 되도록 한다.
        unique_stem = f"{i:05d}_{image_path.stem}"
        dest_img = OUTPUT_DIR / "images" / split / f"{unique_stem}{image_path.suffix}"
        dest_label = OUTPUT_DIR / "labels" / split / f"{unique_stem}.txt"

        shutil.copy2(image_path, dest_img)
        with open(dest_label, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        converted += 1
        if converted % 500 == 0:
            print(f"  {converted}장 변환 완료...")

    print(f"\n변환 완료: {converted}장")
    print(f"라벨 없어서 스킵: {skipped_no_label}장")
    print(f"번호판 없어서 스킵: {skipped_no_plate}장")
    print(f"\n결과: {OUTPUT_DIR}/images/{{train,val}}, {OUTPUT_DIR}/labels/{{train,val}}")
    print("기존 data/plate_yolo 데이터와 합쳐졌습니다 (같은 폴더에 이어붙임).")
    print("다음: python train_plate_detector.py --epochs 100")


if __name__ == "__main__":
    main()
