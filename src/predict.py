"""
간이 속도 예측 모듈

지금까지의 이상탐지(anomaly.py)는 "지금 이 순간이 이상한가"를 판단하는 반면,
이 모듈은 "최근 흐름을 보고 다음 순간에 무슨 일이 생길지"를 예측한다.

핵심 아이디어:
- 차량의 속도는 완전히 무작위가 아니라 관성이 있다 (급발진/급정거가 아닌 이상 부드럽게 변함)
- 그래서 최근 몇 프레임의 속도 패턴을 보면 다음 속도를 어느 정도 예측할 수 있다
- 예측치가 실제와 크게 어긋나기 시작하면(급감 등) "곧 급정거할 가능성"으로 조기 경고 가능

사용 흐름:
1. train_speed_predictor.py 로 학습 → models/speed_predictor.pkl 생성
2. 이 모듈의 SpeedPredictor가 그 모델을 불러와 실시간 예측에 사용
"""

import joblib
import numpy as np


class SpeedPredictor:
    LAG = 3  # 몇 프레임 전까지의 속도를 보고 예측할지

    def __init__(self, model_path: str = "models/speed_predictor.pkl"):
        self.model_path = model_path
        self.model = None

    def load(self):
        self.model = joblib.load(self.model_path)
        return self

    def predict_next(self, recent_speeds: list) -> float | None:
        """
        최근 속도 리스트(오래된 순서, 최소 LAG개)를 받아 다음 속도를 예측한다.
        데이터가 부족하면 None을 반환한다.
        """
        if len(recent_speeds) < self.LAG:
            return None
        if self.model is None:
            self.load()

        features = np.array([recent_speeds[-self.LAG:]])
        prediction = self.model.predict(features)[0]
        return float(prediction)

    def predict_sudden_drop_risk(self, recent_speeds: list, drop_ratio: float = 0.5) -> dict:
        """
        다음 속도를 예측해서, 지금보다 얼마나 급격히 떨어질 것으로 보이는지 평가한다.
        drop_ratio 이상 떨어질 것으로 예측되면 "급정거 위험"으로 조기 경고한다.
        """
        predicted = self.predict_next(recent_speeds)
        if predicted is None:
            return {"predicted_speed": None, "risk": False}

        current = recent_speeds[-1]
        if current <= 0:
            return {"predicted_speed": predicted, "risk": False}

        drop = (current - predicted) / current
        return {"predicted_speed": round(predicted, 1), "risk": drop >= drop_ratio}
