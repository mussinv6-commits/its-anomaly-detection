"""
Kaggle 데이터셋 다운로드 및 프로젝트 구조로 정리
사용법: python src/data_setup.py

주의: kagglehub는 최초 실행 시 Kaggle 로그인(브라우저 인증 또는 kaggle.json)이 필요합니다.
"""

import shutil
from pathlib import Path

import kagglehub

RAW_DIR = Path("data/raw")
IMAGES_DIR = RAW_DIR / "images"
LABELS_DIR = RAW_DIR / "labels"


def download_dataset() -> Path:
    path = kagglehub.dataset_download("saumyapatel/traffic-vehicles-object-detection")
    print(f"다운로드 완료: {path}")
    return Path(path)


def organize_into_project(src_path: Path):
    """kagglehub 캐시 폴더 구조를 data/raw/images, data/raw/labels로 정리한다."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 데이터셋 내부 폴더명은 버전에 따라 다를 수 있어 재귀적으로 탐색
    image_exts = {".jpg", ".jpeg", ".png"}
    label_ext = ".txt"

    copied_images, copied_labels = 0, 0
    for file in src_path.rglob("*"):
        if not file.is_file():
            continue
        if file.suffix.lower() in image_exts:
            shutil.copy2(file, IMAGES_DIR / file.name)
            copied_images += 1
        elif file.suffix.lower() == label_ext:
            shutil.copy2(file, LABELS_DIR / file.name)
            copied_labels += 1

    print(f"이미지 {copied_images}개, 라벨 {copied_labels}개를 data/raw/ 아래로 정리했습니다.")


if __name__ == "__main__":
    dataset_path = download_dataset()
    organize_into_project(dataset_path)
