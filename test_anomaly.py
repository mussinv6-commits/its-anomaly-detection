"""
anomaly.py의 4가지 이상탐지 규칙(과속·역주행·급정거·불법정차)이
각각 의도한 대로 동작하는지, 실제 영상 없이 가상의 움직임 데이터로 검증한다.

시나리오별로 "이게 걸려야 정상"인 케이스와 "안 걸려야 정상"인 케이스를 함께 확인한다.

사용법: python test_anomaly.py
"""

import sys
sys.path.append("src")

from anomaly import AnomalyDetector, AnomalyConfig


FPS = 30.0
METERS_PER_PIXEL = 0.05


def make_history(positions):
    """[(x, y), (x, y), ...] 형태의 좌표 리스트를 track history 포맷으로 변환한다."""
    return [{"frame": i, "bbox": [x, y, x + 40, y + 20]} for i, (x, y) in enumerate(positions)]


def run_case(name, history, prev_speed, expected_flag, should_trigger, detector=None):
    detector = detector or AnomalyDetector()
    track = {"track_id": 0, "history": history}
    result = detector.evaluate(track, FPS, METERS_PER_PIXEL, prev_speed)

    triggered = expected_flag in result["flags"]
    ok = triggered == should_trigger
    status = "PASS" if ok else "FAIL"
    expect_str = "감지되어야 함" if should_trigger else "감지되지 않아야 함"

    print(f"[{status}] {name}")
    print(f"       기대: '{expected_flag}' {expect_str} / 실제 flags={result['flags']} (speed={result['speed_kmh']:.1f}km/h)")
    return ok


def main():
    results = []

    # 1. 과속: 매 프레임 18픽셀씩 이동 → 현실적인 과속 범위(60~150km/h 사이)
    fast_positions = [(i * 18, 100) for i in range(10)]
    results.append(run_case(
        "과속 - 빠르게 이동하는 차량",
        make_history(fast_positions), prev_speed=0.0,
        expected_flag="과속", should_trigger=True,
    ))

    # 1-2. 비현실적 속도: 매 프레임 100픽셀씩 이동 → 250km/h대, 추적 오류로 간주되어 과속 아님
    implausible_positions = [(i * 100, 100) for i in range(10)]
    results.append(run_case(
        "비현실적 속도 - 200km/h대 (추적 오류로 간주, 과속 오탐 없어야 함)",
        make_history(implausible_positions), prev_speed=0.0,
        expected_flag="과속", should_trigger=False,
    ))

    # 1-1. 정상 속도: 매 프레임 3픽셀씩만 이동 → 과속 아님
    normal_positions = [(i * 3, 100) for i in range(10)]
    results.append(run_case(
        "정상 속도 - 천천히 이동하는 차량 (과속 오탐 없어야 함)",
        make_history(normal_positions), prev_speed=0.0,
        expected_flag="과속", should_trigger=False,
    ))

    # 2. 역주행: 정상 방향(1,0)과 반대로 이동 (x가 감소)
    wrong_way_positions = [(500 - i * 20, 100) for i in range(10)]
    results.append(run_case(
        "역주행 - 정상 방향과 반대로 이동",
        make_history(wrong_way_positions), prev_speed=0.0,
        expected_flag="역주행", should_trigger=True,
    ))

    # 2-1. 정상 주행: 정방향 이동 → 역주행 아님
    correct_way_positions = [(i * 20, 100) for i in range(10)]
    results.append(run_case(
        "정상 주행 - 정방향 이동 (역주행 오탐 없어야 함)",
        make_history(correct_way_positions), prev_speed=0.0,
        expected_flag="역주행", should_trigger=False,
    ))

    # 3. 급정거: 이전 속도가 높았는데 마지막 두 프레임 사이 이동량이 급감
    sudden_stop_positions = [(i * 30, 100) for i in range(9)] + [(240 + 1, 100)]  # 마지막만 거의 안 움직임
    results.append(run_case(
        "급정거 - 이전 대비 속도가 급격히 감소",
        make_history(sudden_stop_positions), prev_speed=80.0,  # 이전 속도가 높았다고 가정
        expected_flag="급정거", should_trigger=True,
    ))

    # 4. 불법정차: 10초(300프레임, 30fps 기준) 이상 거의 안 움직임
    stopped_positions = [(100, 100)] * 305
    results.append(run_case(
        "불법정차 - 10초 이상 정지",
        make_history(stopped_positions), prev_speed=0.0,
        expected_flag="불법정차", should_trigger=True,
    ))

    # 4-1. 정상 정차: 5초만 정지 (기준 미달, 불법정차 아님)
    short_stop_positions = [(100, 100)] * 150
    results.append(run_case(
        "짧은 정차 - 5초만 정지 (불법정차 오탐 없어야 함)",
        make_history(short_stop_positions), prev_speed=0.0,
        expected_flag="불법정차", should_trigger=False,
    ))

    print(f"\n총 {len(results)}개 케이스 중 {sum(results)}개 통과")
    print("전체 결과:", "PASS ✅" if all(results) else "FAIL ❌")


if __name__ == "__main__":
    main()
