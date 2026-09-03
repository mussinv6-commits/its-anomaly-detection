"""
flow_features에 쌓인 시계열 데이터로 "다음 속도 예측" 회귀 모델을 학습한다.

사용법:
    set ITS_DB_PASSWORD=본인비밀번호
    python train_speed_predictor.py
"""

import sys
sys.path.append("src")

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

from db import SessionLocal, FlowFeature, Track

LAG = 3  # predict.py의 SpeedPredictor.LAG와 반드시 맞춰야 함


def build_lag_dataset():
    """
    트랙별로 시간 순서대로 정렬한 뒤, (t-3, t-2, t-1) -> t 형태의 학습쌍을 만든다.
    서로 다른 차량(트랙)의 속도를 섞어서 앞뒤로 잇지 않도록 트랙 단위로 끊어서 처리한다.
    """
    session = SessionLocal()
    tracks = session.query(Track).all()

    X, y = [], []
    for track in tracks:
        flows = (
            session.query(FlowFeature)
            .filter_by(track_id=track.id)
            .order_by(FlowFeature.frame_idx)
            .all()
        )
        speeds = [f.speed_kmh for f in flows]

        for i in range(LAG, len(speeds)):
            X.append(speeds[i - LAG:i])
            y.append(speeds[i])

    session.close()
    return np.array(X), np.array(y)


def main():
    print("flow_features에서 학습 데이터를 구성하는 중...")
    X, y = build_lag_dataset()

    if len(X) < 50:
        print(f"학습 데이터가 너무 적습니다 (현재 {len(X)}건). 최소 50건 이상 필요합니다.")
        print("main.py로 실제 영상을 몇 개 더 처리해서 flow_features를 채워주세요.")
        return

    print(f"학습 데이터 {len(X)}건 구성 완료 (입력: 최근 {LAG}프레임 속도 → 다음 속도)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"검증 결과: 평균 절대 오차(MAE) = {mae:.2f} km/h")
    print("(예: MAE가 5면, 평균적으로 실제 속도와 ±5km/h 정도 차이로 예측한다는 뜻)")

    joblib.dump(model, "models/speed_predictor.pkl")
    print("학습 완료: models/speed_predictor.pkl 저장됨")


if __name__ == "__main__":
    main()
