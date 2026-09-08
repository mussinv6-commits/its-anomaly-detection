"""
원근법 보정(호모그래피) 모듈.

기존 방식의 한계: meters_per_pixel 하나의 값으로 화면 전체를 환산하면,
카메라에 가까운 차선과 먼 차선의 "1픽셀=실제 몇 미터"가 다르다는 걸 반영 못 한다.
그래서 먼 차선의 차량 속도가 비현실적으로 높게 계산되는 문제가 생긴다.

호모그래피는 "카메라가 찍은 화면(2D)"과 "실제 지면(2D, 위에서 내려다본 평면)" 사이의
변환 행렬을 계산해서, 화면 어느 위치에 있든 정확한 실제 거리를 구할 수 있게 해준다.

사용 흐름:
1. calibrate_homography.py로 기준점 4개를 입력해서 호모그래피 행렬을 계산 -> 저장
2. main.py --homography 옵션으로 그 행렬을 불러와서 속도 계산에 사용
"""

import numpy as np
import cv2


class PerspectiveCalibrator:
    def __init__(self, homography_matrix: np.ndarray):
        self.H = homography_matrix

    @classmethod
    def from_points(cls, image_points, real_world_points):
        """
        image_points: 화면 속 기준점 4개 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] (픽셀)
        real_world_points: 그에 대응하는 실제 지면 좌표 4개 [(x1,y1), ...] (미터, 임의 원점 기준)
        두 좌표 리스트는 순서가 서로 대응해야 한다 (image_points[0]이 real_world_points[0]에 대응).
        """
        img_pts = np.array(image_points, dtype=np.float32)
        real_pts = np.array(real_world_points, dtype=np.float32)
        H, _ = cv2.findHomography(img_pts, real_pts)
        return cls(H)

    def pixel_to_world(self, point):
        """화면 픽셀 좌표(x,y)를 실제 지면 좌표(미터)로 변환한다."""
        px = np.array([[point]], dtype=np.float32)
        world = cv2.perspectiveTransform(px, self.H)
        return float(world[0][0][0]), float(world[0][0][1])

    def distance_meters(self, point_a, point_b) -> float:
        """화면 속 두 픽셀 좌표 사이의 실제 거리(미터)를 계산한다."""
        wa = self.pixel_to_world(point_a)
        wb = self.pixel_to_world(point_b)
        return ((wa[0] - wb[0]) ** 2 + (wa[1] - wb[1]) ** 2) ** 0.5

    def save(self, path: str):
        np.save(path, self.H)

    @classmethod
    def load(cls, path: str):
        H = np.load(path)
        return cls(H)


class DepthScaleCalibrator:
    """
    화면 속 가로 위치(차선 간격)는 추측하지 않고, 세로 위치(y, 카메라로부터의 깊이)에
    따라서만 "1픽셀 = 몇 미터"를 보간하는 더 단순하고 보수적인 보정 방식.

    PerspectiveCalibrator(호모그래피)는 화면 속 사각형 기준점 4개가 다 정확해야 하는데,
    그중 가로 방향(차선 간 거리, 두 기준점 사이의 깊이)은 실측 없이는 추측에 의존하게 되어
    오차가 커질 수 있다(실제로 도로 폭이 위치에 따라 8m/13m로 모순되게 계산되는 문제를 겪음).

    이 클래스는 실제로 측정 검증된 지점(예: 표준 규격 차량의 폭)만 사용해서,
    "화면 y좌표 -> meters_per_pixel"의 관계를 두 지점 사이에서 선형 보간한다.
    측정 범위를 벗어난 지점은 가장 가까운 측정값으로 고정(clamp)해서 극단적인 외삽을 방지한다.
    """

    def __init__(self, reference_points: list):
        """reference_points: [(image_y1, meters_per_pixel1), (image_y2, meters_per_pixel2), ...]
        y가 작을수록(화면 위쪽, 카메라에서 멈) 보통 meters_per_pixel이 커진다."""
        self.points = sorted(reference_points, key=lambda p: p[0])

    def scale_at(self, y: float) -> float:
        ys = [p[0] for p in self.points]
        scales = [p[1] for p in self.points]

        if y <= ys[0]:
            return scales[0]
        if y >= ys[-1]:
            return scales[-1]

        for i in range(len(ys) - 1):
            if ys[i] <= y <= ys[i + 1]:
                ratio = (y - ys[i]) / (ys[i + 1] - ys[i])
                return scales[i] + ratio * (scales[i + 1] - scales[i])
        return scales[-1]

    def distance_meters(self, point_a, point_b) -> float:
        """두 픽셀 좌표 사이의 실제 거리. 두 점의 평균 y에서의 스케일을 사용한다."""
        avg_y = (point_a[1] + point_b[1]) / 2
        scale = self.scale_at(avg_y)
        dist_px = ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5
        return dist_px * scale
