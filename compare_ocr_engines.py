"""
debug_output/에 이미 저장된 번호판 crop 이미지들로, EasyOCR과 PaddleOCR의
인식 결과를 나란히 비교한다. 영상을 다시 돌릴 필요 없이 바로 비교 가능하다.

사용법:
    pip install paddlepaddle paddleocr --break-system-packages   (최초 1회, 시간 걸림)
    python compare_ocr_engines.py
    python compare_ocr_engines.py --limit 50   (이미지가 너무 많으면 개수 제한)
"""

import argparse
import sys
from pathlib import Path

sys.path.append("src")

from ocr import PlateOCR

DEBUG_DIR = Path("debug_output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100, help="비교할 이미지 최대 개수 (너무 많으면 오래 걸림)")
    args = parser.parse_args()

    if not DEBUG_DIR.exists():
        print(f"{DEBUG_DIR}가 없습니다. main.py --save-ocr-debug로 먼저 이미지를 쌓아주세요.")
        return

    # plate_frame*.jpg(번호판 crop만 있는 것)를 우선 사용, 없으면 vehicle_frame도 포함
    images = sorted(DEBUG_DIR.glob("plate_frame*.jpg"))
    if not images:
        print("plate_frame*.jpg가 없어 vehicle_frame*.jpg로 대체합니다 (정확도가 더 낮게 나올 수 있음).")
        images = sorted(DEBUG_DIR.glob("vehicle_frame*.jpg"))
    images = images[: args.limit]

    if not images:
        print(f"{DEBUG_DIR}에 비교할 이미지가 없습니다.")
        return

    print(f"이미지 {len(images)}장으로 EasyOCR vs PaddleOCR 비교를 시작합니다...")

    easy_ocr = PlateOCR()
    try:
        from ocr_paddle import PlateOCRPaddle
        paddle_ocr = PlateOCRPaddle()
    except ImportError:
        print("\npaddleocr이 설치되어 있지 않습니다. 먼저 설치해주세요:")
        print("  pip install paddlepaddle paddleocr --break-system-packages")
        return

    easy_success, paddle_success = 0, 0

    for img_path in images:
        easy_result = easy_ocr.read_debug(str(img_path))
        paddle_result = paddle_ocr.read_debug(str(img_path))

        easy_ok = easy_result["parsed"] is not None
        paddle_ok = paddle_result["parsed"] is not None
        easy_success += easy_ok
        paddle_success += paddle_ok

        # 둘 중 하나라도 성공했거나, 둘 다 뭔가 읽긴 한 경우만 출력 (완전 무음 케이스는 스킵해서 로그 절약)
        if easy_ok or paddle_ok:
            print(f"\n[{img_path.name}]")
            print(f"  EasyOCR : 원문={easy_result['raw_text']!r} → 파싱={easy_result['parsed']}")
            print(f"  PaddleOCR: 원문={paddle_result['raw_text']!r} → 파싱={paddle_result['parsed']}")

    total = len(images)
    print(f"\n{'='*60}")
    print(f"총 {total}장 중:")
    print(f"  EasyOCR   성공: {easy_success}장 ({easy_success/total*100:.1f}%)")
    print(f"  PaddleOCR 성공: {paddle_success}장 ({paddle_success/total*100:.1f}%)")
    print(f"{'='*60}")

    if paddle_success > easy_success:
        print("\nPaddleOCR이 더 나은 결과를 보였습니다. ocr.py를 PaddleOCR 기반으로 교체하는 것을 고려해보세요.")
    elif easy_success > paddle_success:
        print("\nEasyOCR이 더 나은 결과를 보였습니다. 엔진 문제가 아니라 이미지 자체의 해상도/품질 한계일 가능성이 높습니다.")
    else:
        print("\n두 엔진의 성능이 비슷합니다. 엔진 교체보다 이미지 품질(해상도/각도) 개선이 우선일 수 있습니다.")


if __name__ == "__main__":
    main()
