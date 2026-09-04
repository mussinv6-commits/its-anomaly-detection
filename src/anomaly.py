"""
이상탐지 모듈 (프로젝트 핵심)
- 차량 추적 이력(history)을 바탕으로 정체/역주행/급정거/불법정차/과속을 판별한다.
- 1차: 규칙 기반 임계값 판단
- 2차(선택): 정상 흐름 데이터의 평균/표준편차로 이상치(outlier) 판단
"""

from dataclasses import dataclass


@dataclass
class AnomalyConfig:
    speed_limit: float = 60.0          # km/h, 과속 기준
    max_plausible_speed: float = 150.0  # km/h, 이보다 빠르면 실제 속도가 아니라 추적 오류(ID 스위칭 등)로 간주
    stop_seconds: float = 10.0          # 불법 정차 판단 기준(초)
    sudden_decel_ratio: float = 0.8     # 급정거 판단: 속도가 이전 대비 이 비율 이상 감소 (70%→80%로 재강화)
    min_speed_for_decel: float = 25.0   # km/h, 이 속도 이상으로 달리던 차량만 급정거 판단 대상 (20→25로 상향)
                                          # (저속 구간의 미세한 흔들림까지 "급정거"로 잡히는 걸 방지)
    congestion_ratio: float = 0.4       # 정체 판단: 평균 통과량 대비 이 비율 이하로 감소


class AnomalyDetector:
    def __init__(self, config: AnomalyConfig = None, lane_directions=(1, 0)):
        self.config = config or AnomalyConfig()
        # 정상 주행 방향(들). 단일 벡터 (x,y) 하나를 줄 수도 있고,
        # 양방향 도로 대응을 위해 [(x1,y1), (x2,y2), ...] 리스트를 줄 수도 있다.
        if isinstance(lane_directions[0], (int, float)):
            self.lane_directions = [lane_directions]
        else:
            self.lane_directions = list(lane_directions)

    def compute_velocity(self, history: list, fps: float, meters_per_pixel: float, window: int = 5):
        """
        history의 최근 여러 프레임(window)을 평균 내서 속도(km/h)와 이동 벡터를 계산한다.
        딱 2프레임만 보면 검출 박스가 살짝만 흔들려도 속도가 크게 튀기 때문에
        (예: 한 프레임에 76km/h, 다음 프레임에 140km/h) 여러 프레임을 평균해서 노이즈를 줄인다.
        """
        if len(history) < 2:
            return 0.0, (0.0, 0.0)

        n = min(window, len(history) - 1)
        p_start, p_end = history[-n - 1]["bbox"], history[-1]["bbox"]
        c1 = ((p_start[0] + p_start[2]) / 2, (p_start[1] + p_start[3]) / 2)
        c2 = ((p_end[0] + p_end[2]) / 2, (p_end[1] + p_end[3]) / 2)

        dx, dy = (c2[0] - c1[0]) / n, (c2[1] - c1[1]) / n  # 프레임당 평균 이동량
        dist_px = (dx ** 2 + dy ** 2) ** 0.5
        dist_m = dist_px * meters_per_pixel
        dt = 1 / fps
        speed_kmh = (dist_m / dt) * 3.6 if dt > 0 else 0.0

        return speed_kmh, (dx, dy)

    def is_wrong_way(self, movement_vector: tuple) -> bool:
        """
        이동 벡터가 등록된 정상 주행 방향들(양방향 도로면 2개) 중 어느 것과도
        비슷하지 않고, 확실히 반대 방향(120도 이상 벌어짐)일 때만 역주행으로 판단한다.
        여러 정상 방향 중 "가장 가까운" 방향을 기준으로 비교한다.
        """
        dx, dy = movement_vector
        mag_move = (dx ** 2 + dy ** 2) ** 0.5
        if mag_move < 1e-6:
            return False  # 정지 상태에서는 방향 판단 자체가 무의미

        best_cos_sim = -1.0  # 등록된 방향들 중 가장 비슷한(코사인 유사도가 가장 큰) 것을 채택
        for lx, ly in self.lane_directions:
            mag_lane = (lx ** 2 + ly ** 2) ** 0.5
            if mag_lane < 1e-6:
                continue
            cos_sim = (dx * lx + dy * ly) / (mag_move * mag_lane)
            best_cos_sim = max(best_cos_sim, cos_sim)

        return best_cos_sim < -0.3

    def is_speeding(self, speed_kmh: float) -> bool:
        """
        과속 기준(speed_limit)을 넘되, 물리적으로 말이 안 되는 속도(max_plausible_speed 초과)는
        실제 과속이 아니라 추적 오류(ID 스위칭, bbox 튐)로 보고 제외한다.
        예: 200km/h는 일반 도로에서 나올 수 없는 값 — 십중팔구 계산 오류.
        """
        return self.config.speed_limit < speed_kmh <= self.config.max_plausible_speed

    def is_sudden_deceleration(self, prev_speed: float, curr_speed: float) -> bool:
        """
        원래 어느 정도 속도(min_speed_for_decel 이상)로 달리던 차량이
        급격히(sudden_decel_ratio 이상) 느려질 때만 급정거로 판단한다.
        저속 구간(예: 5km/h → 2km/h)의 미세한 흔들림은 비율로는 60% 감소처럼 보여도
        실제로는 위험한 급정거가 아니므로 최소 속도 기준으로 걸러낸다.
        """
        if prev_speed < self.config.min_speed_for_decel:
            return False
        return (prev_speed - curr_speed) / prev_speed >= self.config.sudden_decel_ratio

    def is_illegally_stopped(self, history: list, fps: float, threshold_px: float = 3.0) -> bool:
        """최근 N초간 위치 변화가 거의 없으면 불법 정차로 판단한다."""
        n_frames = int(self.config.stop_seconds * fps)
        if len(history) < n_frames:
            return False
        recent = history[-n_frames:]
        xs = [(h["bbox"][0] + h["bbox"][2]) / 2 for h in recent]
        ys = [(h["bbox"][1] + h["bbox"][3]) / 2 for h in recent]
        movement = max(xs) - min(xs) + max(ys) - min(ys)
        return movement < threshold_px

    def evaluate(self, track: dict, fps: float, meters_per_pixel: float, prev_instant_speed: float = 0.0):
        """
        트랙 하나를 받아 이상탐지 결과 dict를 반환한다.

        속도는 두 가지를 따로 계산한다:
        - speed_kmh (스무딩): 여러 프레임 평균 — 리포팅, 과속·역주행 판단용 (노이즈에 안정적)
        - instant_speed_kmh (순간): 마지막 2프레임만 — 급정거 판단용
          (급정거는 "방금 급격히 느려졌는가"가 핵심이라, 평균을 쓰면 오히려 둔감해짐)
        """
        speed, vector = self.compute_velocity(track["history"], fps, meters_per_pixel, window=5)
        instant_speed, _ = self.compute_velocity(track["history"], fps, meters_per_pixel, window=1)
        flags = []

        if self.is_speeding(speed):
            flags.append("과속")
        if self.is_wrong_way(vector):
            flags.append("역주행")
        if self.is_sudden_deceleration(prev_instant_speed, instant_speed):
            flags.append("급정거")
        if self.is_illegally_stopped(track["history"], fps):
            flags.append("불법정차")

        return {
            "track_id": track["track_id"],
            "speed_kmh": speed,
            "instant_speed_kmh": instant_speed,
            "dx": vector[0],
            "dy": vector[1],
            "flags": flags,
        }


# TODO: 6단계에서 정상 주행 속도 분포를 수집해 평균/표준편차 기반 outlier 판단 추가
