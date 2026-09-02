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
    stop_seconds: float = 10.0          # 불법 정차 판단 기준(초)
    sudden_decel_ratio: float = 0.6     # 급정거 판단: 속도가 이전 대비 이 비율 이상 감소
    congestion_ratio: float = 0.4       # 정체 판단: 평균 통과량 대비 이 비율 이하로 감소


class AnomalyDetector:
    def __init__(self, config: AnomalyConfig = None, lane_direction: tuple = (1, 0)):
        self.config = config or AnomalyConfig()
        self.lane_direction = lane_direction  # 정상 주행 방향 벡터 (x, y)

    def compute_velocity(self, history: list, fps: float, meters_per_pixel: float):
        """history의 마지막 두 지점으로 속도(km/h)와 이동 벡터를 계산한다."""
        if len(history) < 2:
            return 0.0, (0.0, 0.0)

        p1, p2 = history[-2]["bbox"], history[-1]["bbox"]
        c1 = ((p1[0] + p1[2]) / 2, (p1[1] + p1[3]) / 2)
        c2 = ((p2[0] + p2[2]) / 2, (p2[1] + p2[3]) / 2)

        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        dist_px = (dx ** 2 + dy ** 2) ** 0.5
        dist_m = dist_px * meters_per_pixel
        dt = 1 / fps
        speed_kmh = (dist_m / dt) * 3.6 if dt > 0 else 0.0

        return speed_kmh, (dx, dy)

    def is_wrong_way(self, movement_vector: tuple) -> bool:
        """이동 벡터가 정상 주행 방향과 반대인지 내적으로 판단한다."""
        dx, dy = movement_vector
        lx, ly = self.lane_direction
        dot = dx * lx + dy * ly
        return dot < 0

    def is_speeding(self, speed_kmh: float) -> bool:
        return speed_kmh > self.config.speed_limit

    def is_sudden_deceleration(self, prev_speed: float, curr_speed: float) -> bool:
        if prev_speed <= 0:
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

    def evaluate(self, track: dict, fps: float, meters_per_pixel: float, prev_speed: float = 0.0):
        """트랙 하나를 받아 이상탐지 결과 dict를 반환한다."""
        speed, vector = self.compute_velocity(track["history"], fps, meters_per_pixel)
        flags = []

        if self.is_speeding(speed):
            flags.append("과속")
        if self.is_wrong_way(vector):
            flags.append("역주행")
        if self.is_sudden_deceleration(prev_speed, speed):
            flags.append("급정거")
        if self.is_illegally_stopped(track["history"], fps):
            flags.append("불법정차")

        return {
            "track_id": track["track_id"],
            "speed_kmh": speed,
            "dx": vector[0],
            "dy": vector[1],
            "flags": flags,
        }


# TODO: 6단계에서 정상 주행 속도 분포를 수집해 평균/표준편차 기반 outlier 판단 추가
