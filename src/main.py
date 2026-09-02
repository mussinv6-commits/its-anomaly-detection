"""
전체 파이프라인 실행 (영상 단위)
사용법: python src/main.py --video data/raw/sample.mp4

이 스크립트가 하는 일:
1. 영상에서 프레임마다 차량을 검출·추적한다.
2. 각 차량(트랙)이 화면에 처음 나타나면 DB에 Track을 하나 생성한다.
3. 매 프레임마다 그 트랙의 이동 특징(속도, 방향)을 flow_features에 저장한다
   → 이 데이터가 나중에 Isolation Forest 학습 재료가 된다.
4. 규칙 기반으로 이상상황이 감지되면 anomaly_records에도 별도로 기록한다.
"""

import argparse
import cv2

from detect import VehicleDetector
from tracker import SimpleTracker
from anomaly import AnomalyDetector, AnomalyConfig
from db import init_db, create_track, save_flow_feature, save_anomaly


def calibrate_lane_directions(video_path: str, detector, sample_frames: int = 90):
    """
    영상 앞부분(sample_frames만큼)을 먼저 훑어서, 정상 주행 방향(들)을 자동으로 찾는다.

    편도 도로면 방향 1개, 양방향 도로면 서로 반대인 방향 2개를 찾아
    "정상"으로 등록한다. 이렇게 하지 않으면(정상 방향을 1개로 고정하면)
    양방향 도로에서 한쪽 차선 차량이 전부 역주행으로 오탐된다.
    """
    tracker = SimpleTracker()
    cap = cv2.VideoCapture(video_path)
    vectors = []  # 캘리브레이션 구간에서 수집한 (dx, dy) 이동 벡터들
    frame_idx = 0

    while frame_idx < sample_frames:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)
        for track in tracks:
            if len(track["history"]) >= 2:
                p1 = track["history"][-2]["bbox"]
                p2 = track["history"][-1]["bbox"]
                c1 = ((p1[0] + p1[2]) / 2, (p1[1] + p1[3]) / 2)
                c2 = ((p2[0] + p2[2]) / 2, (p2[1] + p2[3]) / 2)
                dx, dy = c2[0] - c1[0], c2[1] - c1[1]
                mag = (dx ** 2 + dy ** 2) ** 0.5
                if mag > 1.0:  # 너무 미세한(노이즈성) 움직임은 방향 추정에서 제외
                    vectors.append((dx, dy))
        frame_idx += 1

    cap.release()

    if not vectors:
        print("경고: 초반 구간에서 뚜렷한 이동을 찾지 못해 기본값(1,0)을 사용합니다.")
        return [(1.0, 0.0)]

    # 전체 평균 방향을 기준으로, 벡터들을 "평균과 같은 편"과 "반대 편" 두 그룹으로 나눈다.
    avg_dx = sum(v[0] for v in vectors) / len(vectors)
    avg_dy = sum(v[1] for v in vectors) / len(vectors)

    group_a, group_b = [], []
    for dx, dy in vectors:
        dot = dx * avg_dx + dy * avg_dy
        (group_a if dot >= 0 else group_b).append((dx, dy))

    def normalize(group):
        gx = sum(v[0] for v in group) / len(group)
        gy = sum(v[1] for v in group) / len(group)
        mag = (gx ** 2 + gy ** 2) ** 0.5
        return (gx / mag, gy / mag) if mag > 1e-6 else None

    lane_directions = []
    dir_a = normalize(group_a)
    if dir_a:
        lane_directions.append(dir_a)

    # group_b(반대 방향 그룹)가 충분히 크면(전체의 15% 이상) 양방향 도로로 보고 같이 등록
    if len(group_b) >= max(3, len(vectors) * 0.15):
        dir_b = normalize(group_b)
        if dir_b:
            lane_directions.append(dir_b)

    print(f"자동 캘리브레이션 완료: 정상 주행 방향 {len(lane_directions)}개 등록 "
          f"({'편도' if len(lane_directions) == 1 else '양방향'} 도로로 판단) "
          f"{[tuple(round(v, 2) for v in d) for d in lane_directions]}")
    return lane_directions


def run(video_path: str, meters_per_pixel: float = 0.05, speed_limit: float = 60.0):
    detector = VehicleDetector()
    tracker = SimpleTracker()
    init_db()

    # 1차 통과: 정상 주행 방향(들)을 먼저 파악해 캘리브레이션 (편도/양방향 자동 판단)
    lane_directions = calibrate_lane_directions(video_path, detector)
    config = AnomalyConfig(speed_limit=speed_limit)
    anomaly_detector = AnomalyDetector(config=config, lane_directions=lane_directions)

    # 2차 통과: 실제 검출·추적·이상탐지 수행
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev_speeds = {}
    frame_idx = 0

    # 로컬 tracker의 임시 track_id -> DB의 진짜 tracks.id 매핑
    # (같은 차량이 여러 프레임에 걸쳐 나타나도 DB Track은 하나만 생성되도록)
    local_to_db_track = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx)

        for track in tracks:
            local_id = track["track_id"]

            # 이 차량을 처음 보는 경우, DB에 Track을 새로 생성
            if local_id not in local_to_db_track:
                local_to_db_track[local_id] = create_track(source=video_path)
            db_track_id = local_to_db_track[local_id]

            result = anomaly_detector.evaluate(
                track, fps, meters_per_pixel, prev_speeds.get(local_id, 0.0)
            )
            prev_speeds[local_id] = result["instant_speed_kmh"]  # 급정거 판단은 순간속도 기준으로 이어감
            bbox = track["bbox"]
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # 이상 여부와 무관하게 모든 흐름 데이터를 학습용으로 저장
            save_flow_feature(
                track_id=db_track_id,
                speed_kmh=result["speed_kmh"],
                dx=result["dx"],
                dy=result["dy"],
                bbox_area=bbox_area,
                frame_idx=frame_idx,
            )

            if result["flags"]:
                print(f"[frame {frame_idx}] track {local_id}: {result['flags']} ({result['speed_kmh']:.1f}km/h)")
                save_anomaly(
                    track_id=db_track_id,
                    flags=result["flags"],
                    speed_kmh=result["speed_kmh"],
                )

        frame_idx += 1

    cap.release()
    print(f"\n완료: 총 {frame_idx}프레임 처리, {len(local_to_db_track)}대 차량 추적, DB에 저장됨")

    if prev_speeds:
        avg_speed = sum(prev_speeds.values()) / len(prev_speeds)
        print(f"평균 속도: {avg_speed:.1f}km/h")
        if avg_speed > 100:
            print("참고: 평균 속도가 비정상적으로 높게 나왔다면, meters_per_pixel 값이 "
                  "이 카메라의 실제 축척과 안 맞을 수 있습니다. run() 호출 시 이 값을 조정해보세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="입력 영상 경로")
    parser.add_argument(
        "--speed-limit", type=float, default=60.0,
        help="과속 판단 기준(km/h). 이상탐지가 전혀 안 잡힐 때 낮춰서(예: 10) 로직 자체가 "
             "작동하는지 검증하는 용도로도 쓸 수 있음",
    )
    args = parser.parse_args()
    run(args.video, speed_limit=args.speed_limit)

