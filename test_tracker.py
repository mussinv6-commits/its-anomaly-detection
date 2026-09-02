"""
tracker.py가 올바르게 동작하는지, 실제 영상 없이 가상의 움직이는 박스로 검증한다.

시나리오:
- 차량 A: 왼쪽에서 오른쪽으로 이동 (매 프레임 x가 10씩 증가)
- 차량 B: 다른 위치에서 별도로 이동
- 5프레임 동안 같은 track_id가 유지되는지 확인

사용법: python test_tracker.py
"""

import sys
sys.path.append("src")

from tracker import SimpleTracker


def make_box(x, y, w=50, h=30):
    return [x, y, x + w, y + h]


def main():
    tracker = SimpleTracker(iou_threshold=0.3, max_missed=5)

    # 두 대의 차량이 5프레임 동안 각자 이동하는 가상 시나리오
    frames = []
    for frame_idx in range(5):
        car_a = {"bbox": make_box(10 + frame_idx * 10, 100), "conf": 0.9, "cls": 2}  # 승용차
        car_b = {"bbox": make_box(300 - frame_idx * 8, 200), "conf": 0.85, "cls": 7}  # 트럭
        frames.append([car_a, car_b])

    track_id_history = {}  # 프레임별로 어떤 차량이 어떤 track_id를 받았는지 기록

    for frame_idx, detections in enumerate(frames):
        results = tracker.update(detections, frame_idx)
        print(f"[frame {frame_idx}] ", end="")
        for r in results:
            print(f"track_id={r['track_id']} bbox={r['bbox']}", end="  |  ")
        print()

        for i, r in enumerate(results):
            track_id_history.setdefault(i, []).append(r["track_id"])

    print("\n검증 결과:")
    all_stable = True
    for vehicle_idx, ids in track_id_history.items():
        stable = len(set(ids)) == 1
        all_stable = all_stable and stable
        status = "OK (ID 유지됨)" if stable else "FAIL (ID가 중간에 바뀜)"
        print(f"- 차량 {vehicle_idx}: track_id 이력 = {ids} → {status}")

    print("\n전체 결과:", "PASS ✅" if all_stable else "FAIL ❌")


if __name__ == "__main__":
    main()
