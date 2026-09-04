"""
여러 영상을 한 번에 처리하고, 영상별 결과를 표로 비교한다.

사용법:
    set ITS_DB_PASSWORD=본인비밀번호
    python batch_compare.py --videos data/raw/1.mp4 data/raw/2.mp4 data/raw/3.mp4

    또는 폴더 안의 모든 mp4를 자동으로 찾아서 처리:
    python batch_compare.py --folder data/raw

    이미 처리한 영상은 다시 돌리지 않고 기존 결과만 보고 싶으면:
    python batch_compare.py --summary-only
"""

import argparse
import sys
from pathlib import Path

sys.path.append("src")

from main import run
from db import get_video_summary, get_all_video_sources


def print_comparison_table(summaries: list):
    print("\n[이벤트 건수 — 한 차량이 여러 프레임에서 반복 감지되면 다 더해짐]")
    print("=" * 78)
    print(f"{'영상':<28} {'차량수':>6} {'평균속도':>10} {'과속':>5} {'역주행':>6} {'급정거':>6} {'불법정차':>7}")
    print("-" * 78)
    for s in summaries:
        name = Path(s["source"]).name
        counts = s["anomaly_counts"]
        print(
            f"{name:<28} {s['vehicle_count']:>6} {s['avg_speed_kmh']:>8.1f}km/h "
            f"{counts.get('과속', 0):>5} {counts.get('역주행', 0):>6} "
            f"{counts.get('급정거', 0):>6} {counts.get('불법정차', 0):>7}"
        )
    print("=" * 78)

    print("\n[고유 차량 수 — 같은 차량은 한 번만 카운트, 실제 대수 파악용]")
    print("=" * 78)
    print(f"{'영상':<28} {'차량수':>6} {'과속':>5} {'역주행':>6} {'급정거':>6} {'불법정차':>7}")
    print("-" * 78)
    for s in summaries:
        name = Path(s["source"]).name
        vcounts = s.get("anomaly_vehicle_counts", {})
        print(
            f"{name:<28} {s['vehicle_count']:>6} "
            f"{vcounts.get('과속', 0):>5} {vcounts.get('역주행', 0):>6} "
            f"{vcounts.get('급정거', 0):>6} {vcounts.get('불법정차', 0):>7}"
        )
    print("=" * 78)
    print("(두 표의 숫자 차이가 크면, 소수의 차량이 반복적으로 이상탐지에 걸린 것 —")
    print(" 추적이 불안정하거나 meters_per_pixel 보정이 필요할 수 있음)")

    if len(summaries) > 1:
        speeds = [s["avg_speed_kmh"] for s in summaries if s["vehicle_count"] > 0]
        if speeds:
            print(f"\n영상 간 평균속도 편차: 최소 {min(speeds):.1f}km/h ~ 최대 {max(speeds):.1f}km/h")
            print("(편차가 크면 카메라 각도/거리에 따라 meters_per_pixel 값을 영상별로 조정할 필요가 있음)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", help="처리할 영상 파일 경로들 (여러 개 가능)")
    parser.add_argument("--folder", help="이 폴더 안의 모든 .mp4 파일을 자동으로 처리")
    parser.add_argument("--summary-only", action="store_true", help="새로 처리하지 않고 기존 결과만 비교")
    parser.add_argument("--speed-limit", type=float, default=60.0)
    args = parser.parse_args()

    if args.summary_only:
        sources = get_all_video_sources()
        print(f"기존에 처리된 영상/이미지 소스 {len(sources)}개 발견")
    else:
        video_paths = []
        if args.folder:
            video_paths += [str(p) for p in Path(args.folder).glob("*.mp4")]
        if args.videos:
            video_paths += args.videos

        if not video_paths:
            print("처리할 영상이 없습니다. --videos 또는 --folder를 지정하세요.")
            return

        print(f"총 {len(video_paths)}개 영상을 순서대로 처리합니다.\n")
        for i, video_path in enumerate(video_paths, start=1):
            print(f"[{i}/{len(video_paths)}] {video_path} 처리 중...")
            run(video_path, speed_limit=args.speed_limit)
            print()

        sources = video_paths

    summaries = [get_video_summary(src) for src in sources]
    print_comparison_table(summaries)


if __name__ == "__main__":
    main()
