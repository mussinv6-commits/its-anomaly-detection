"""
차량 검출 모듈
- YOLOv8로 영상 프레임에서 차량 bounding box를 검출한다.
"""

from ultralytics import YOLO


class VehicleDetector:
    def __init__(self, model_path: str = "yolov8s.pt", conf_threshold: float = 0.4):
        # yolov8n(nano) → yolov8s(small)로 교체: 속도는 조금 느려지지만 정확도가 눈에 띄게 개선됨
        # 초기엔 COCO 사전학습 모델(car, truck, bus 클래스 포함)로 시작 가능
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.vehicle_classes = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO 기준)

    def detect(self, frame):
        """
        단일 프레임에서 차량 bounding box 리스트를 반환한다.
        반환 형식: [{"bbox": [x1, y1, x2, y2], "conf": float, "cls": int}, ...]
        """
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.vehicle_classes:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append({"bbox": [x1, y1, x2, y2], "conf": conf, "cls": cls_id})
        return detections


if __name__ == "__main__":
    import cv2

    detector = VehicleDetector()
    frame = cv2.imread("data/raw/sample.jpg")
    print(detector.detect(frame))
