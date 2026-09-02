"""
차량 추적 모듈
- 프레임 간 검출 결과를 이어붙여 차량마다 고유 ID를 부여한다.
- IOU 기반 전역 최적 매칭(greedy) + 클래스 일치 검증으로 ID 스위칭을 줄인다.
"""


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5):
        self.tracks = {}  # track_id -> {"bbox": [...], "cls": int, "missed": int, "history": [...]}
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
        detections: [{"bbox": [...], "conf": ..., "cls": ...}, ...]
        반환: [{"track_id": int, "bbox": [...], "history": [...]}, ...]

        매칭 방식: 모든 (검출, 트랙) 쌍의 IOU를 계산해 점수 높은 순으로 정렬한 뒤,
        이미 매칭된 검출/트랙은 건너뛰며 그리디하게 배정한다.
        같은 클래스끼리만 매칭해서 "차가 갑자기 버스로 바뀌는" 오매칭을 방지한다.
        """
        # 1. 모든 (검출 idx, 트랙 id) 쌍의 IOU 점수 계산 (같은 클래스만)
        candidates = []
        for det_idx, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                if det.get("cls") is not None and track.get("cls") is not None:
                    if det["cls"] != track["cls"]:
                        continue
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > self.iou_threshold:
                    candidates.append((iou, det_idx, track_id))

        # 2. IOU 높은 순으로 정렬 후 그리디 배정 (전역 최적에 가깝게)
        candidates.sort(key=lambda c: c[0], reverse=True)

        matched_det_idx = set()
        matched_track_id = set()
        assignment = {}  # det_idx -> track_id

        for iou, det_idx, track_id in candidates:
            if det_idx in matched_det_idx or track_id in matched_track_id:
                continue
            assignment[det_idx] = track_id
            matched_det_idx.add(det_idx)
            matched_track_id.add(track_id)

        # 3. 배정 결과 반영
        results = []
        for det_idx, det in enumerate(detections):
            if det_idx in assignment:
                track_id = assignment[det_idx]
                track = self.tracks[track_id]
                track["bbox"] = det["bbox"]
                track["cls"] = det.get("cls", track.get("cls"))
                track["missed"] = 0
                track["history"].append({"frame": frame_idx, "bbox": det["bbox"]})
                results.append({"track_id": track_id, "bbox": det["bbox"], "history": track["history"]})
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    "bbox": det["bbox"],
                    "cls": det.get("cls"),
                    "missed": 0,
                    "history": [{"frame": frame_idx, "bbox": det["bbox"]}],
                }
                results.append({"track_id": new_id, "bbox": det["bbox"], "history": self.tracks[new_id]["history"]})

        # 4. 매칭 안 된 트랙은 missed 카운트 증가, 초과 시 삭제
        matched_ids = {r["track_id"] for r in results}
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_ids:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missed:
                    del self.tracks[track_id]

        return results


# TODO: 정확도가 여전히 부족하면 ByteTrack(bytetracker 패키지)으로 교체

