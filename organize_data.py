"""
kagglehub로 받은 데이터를 data/raw/images, data/raw/labels로 정리
사용법: python organize_data.py
"""

import shutil
from pathlib import Path

# 다운로드 로그에서 확인한 실제 경로로 교체하세요
src_path = Path(
    r"C:\Users\mbc\.cache\kagglehub\datasets\saumyapatel\traffic-vehicles-object-detection\versions\1"
)

raw_dir = Path("data/raw")
images_dir = raw_dir / "images"
labels_dir = raw_dir / "labels"
images_dir.mkdir(parents=True, exist_ok=True)
labels_dir.mkdir(parents=True, exist_ok=True)

image_exts = {".jpg", ".jpeg", ".png"}
copied_images, copied_labels = 0, 0

for file in src_path.rglob("*"):
    if not file.is_file():
        continue
    if file.suffix.lower() in image_exts:
        shutil.copy2(file, images_dir / file.name)
        copied_images += 1
    elif file.suffix.lower() == ".txt":
        shutil.copy2(file, labels_dir / file.name)
        copied_labels += 1

print(f"이미지 {copied_images}개, 라벨 {copied_labels}개 정리 완료")
