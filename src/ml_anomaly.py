"""
학습형 이상탐지 모듈 (Isolation Forest)

규칙 기반(anomaly.py)이 "임계값을 넘었는가"로 판단하는 반면,
이 모듈은 "정상 흐름 데이터 전체와 비교했을 때 얼마나 동떨어져 있는가"로 판단한다.

핵심 아이디어:
- 이상상황은 워낙 드물어서 "이상 사례"를 학습 데이터로 모으기 어렵다.
- 대신 "정상 흐름" 데이터는 넘쳐나므로, 정상 데이터만으로 모델을 학습시키고
  그 정상 패턴에서 멀리 떨어진 데이터를 이상치로 판단한다 (비지도 학습).

사용 흐름:
1. train_isolation_forest.py 로 학습 → models/isolation_forest.pkl 생성
2. 이 모듈의 MLAnomalyDetector가 그 모델을 불러와 실시간 판정에 사용
"""

import joblib
import numpy as np


class MLAnomalyDetector:
    def __init__(self, model_path: str = "models/isolation_forest.pkl"):
        self.model_path = model_path
        self.model = None

    def load(self):
        self.model = joblib.load(self.model_path)
        return self

    def predict(self, speed_kmh: float, dx: float, dy: float, bbox_area: float) -> dict:
        """
        하나의 트랙 특징을 받아 이상치 여부를 판정한다.
        반환: {"is_anomaly": bool, "score": float}
        score가 낮을수록(음수로 클수록) 정상 패턴에서 더 동떨어진 것.
        """
        if self.model is None:
            self.load()

        features = np.array([[speed_kmh, dx, dy, bbox_area]])
        prediction = self.model.predict(features)[0]   # 1=정상, -1=이상치
        score = self.model.score_samples(features)[0]

        return {"is_anomaly": prediction == -1, "score": float(score)}


# TODO: 파일럿 단계에서 관제 담당자의 실제 확인 결과를 피드백으로 반영해
#       주기적으로 재학습하는 파이프라인 구축 (본 문서 로드맵 V1.2 참고)
