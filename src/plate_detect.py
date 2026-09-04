"""
번호판 위치 검출 모듈 (학습형)

detect.py(VehicleDetector)가 "차량"을 찾는 것처럼, 이 모듈은 "번호판"만 찾는다.
train_plate_detector.py로 학습한 전용 모델(models/plate_detector.pt)을 사용한다.

기존 방식(main.py의 "차량 하단 40% 추정")보다 훨씬 정확하게 번호판 위치를 찾아서,
그 잘라낸 영역만 ocr.py에 넘기면 인식률이 크게 개선될 것으로 기대한다.
"""

from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path: str = "models/plate_detector.pt", conf_threshold: float = 0.1):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold  # 기본값을 0.3→0.1로 낮춤 (아무것도 못 찾는 문제 진단을 위해 임시로 완화)

    def detect(self, vehicle_crop):
        """
        차량 크롭 이미지 안에서 번호판 bbox를 찾는다.
        반환: [x1, y1, x2, y2] (차량 크롭 기준 좌표) 또는 못 찾으면 None
        여러 개 검출되면 신뢰도가 가장 높은 것 하나만 반환한다.
        """
        results = self.model.predict(source=vehicle_crop, conf=self.conf_threshold, verbose=False)
        best_box, best_conf = None, 0.0

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box.xyxy[0].tolist()

        return best_box

    def crop_plate(self, vehicle_crop, plate_box):
        x1, y1, x2, y2 = [int(v) for v in plate_box]
        return vehicle_crop[y1:y2, x1:x2]
