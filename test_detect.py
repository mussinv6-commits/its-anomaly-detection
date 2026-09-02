"""
1장짜리 이미지로 차량 검출이 되는지 확인하는 테스트 스크립트
사용법: python test_detect.py
"""

import sys
from pathlib import Path

sys.path.append("src")

from detect import VehicleDetector
import cv2

sample_image = list(Path("data/raw/images").glob("*.jpg"))[0]
print("테스트 이미지:", sample_image)

detector = VehicleDetector()
frame = cv2.imread(str(sample_image))
results = detector.detect(frame)

print(f"검출된 차량 수: {len(results)}")
for r in results:
    print(r)
