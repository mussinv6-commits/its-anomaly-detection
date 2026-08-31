"""
차량 추적 모듈
- 프레임 간 검출 결과를 이어붙여 차량마다 고유 ID를 부여한다.
- 초기 버전은 IOU 기반 단순 매칭으로 시작하고, 추후 ByteTrack으로 고도화한다.
"""


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        self.tracks = {}  # track_id -> {"bbox": [...], "missed": int, "history": [...]}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def _iou(self, box_a, box_b):
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0

    def update(self, detections, frame_idx: int):
        """
        detections: [{"bbox": [...], "conf": ...}, ...]
        반환: [{"track_id": int, "bbox": [...], "history": [...]}, ...]
        """
        matched_ids = set()
        results = []

        for det in detections:
            best_id, best_iou = None, self.iou_threshold
            for track_id, track in self.tracks.items():
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_id, best_iou = track_id, iou

            if best_id is not None:
                track = self.tracks[best_id]
                track["bbox"] = det["bbox"]
                track["missed"] = 0
                track["history"].append({"frame": frame_idx, "bbox": det["bbox"]})
                matched_ids.add(best_id)
                results.append({"track_id": best_id, "bbox": det["bbox"], "history": track["history"]})
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "bbox": det["bbox"],
                    "missed": 0,
                    "history": [{"frame": frame_idx, "bbox": det["bbox"]}],
                }
                matched_ids.add(new_id)
                results.append({"track_id": new_id, "bbox": det["bbox"], "history": self.tracks[new_id]["history"]})

        # 매칭 안 된 트랙은 missed 카운트 증가, 초과 시 삭제
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_ids:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missed:
                    del self.tracks[track_id]

        return results


# TODO: 정확도가 부족하면 ByteTrack(bytetracker 패키지)으로 교체
