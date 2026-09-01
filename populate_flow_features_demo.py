"""
[데모/테스트 전용] Isolation Forest 학습 파이프라인이 정상 동작하는지 검증하기 위해,
detection_records의 bbox 크기를 기반으로 임시 flow_features를 생성한다.

주의: 이건 실제 속도 데이터가 아니다. 정지 이미지에는 "이동"이 없어서
      실제 속도를 계산할 수 없기 때문에, 정상 분포를 흉내낸 가상의 수치를 채운 것이다.
      실제 데이터는 main.py로 영상(video)을 처리해야 쌓인다 (src/anomaly.py 참고).

사용법: python populate_flow_features_demo.py
"""

import sys
sys.path.append("src")

import random
from db import init_db, get_recent_detections, save_flow_feature

random.seed(42)


def main():
    init_db()
    detections = get_recent_detections(limit=500)

    if not detections:
        print("detection_records에 데이터가 없습니다. populate_db.py를 먼저 실행하세요.")
        return

    saved = 0
    for i, d in enumerate(detections):
        bbox_area = (d.x2 - d.x1) * (d.y2 - d.y1)

        # 정상 주행을 가정한 가상의 속도(대부분 30~70km/h 근처에 분포)
        speed_kmh = max(0, random.gauss(50, 10))
        dx = random.gauss(5, 2)
        dy = random.gauss(0, 1)

        # d.track_id가 실제 존재하는 트랙을 가리키므로 그대로 재사용 (FK 무결성 유지)
        save_flow_feature(
            track_id=d.track_id,
            speed_kmh=speed_kmh,
            dx=dx,
            dy=dy,
            bbox_area=bbox_area,
            frame_idx=0,
        )
        saved += 1

    print(f"[데모 데이터] flow_features {saved}건 생성 완료 (실제 속도 데이터 아님, 파이프라인 검증용)")


if __name__ == "__main__":
    main()
