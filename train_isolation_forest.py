"""
flow_features 테이블에 쌓인 데이터로 Isolation Forest 모델을 학습한다.

사용법:
    set ITS_DB_PASSWORD=본인비밀번호
    python train_isolation_forest.py

전제:
    flow_features 테이블에 데이터가 있어야 한다.
    (video 파이프라인을 돌리면 main.py가 자동으로 쌓지만,
     정지 이미지만 있는 지금 단계에서는 populate_flow_features_demo.py로
     검출 결과에서 임시 특징을 만들어 테스트할 수 있다.)
"""

import sys
sys.path.append("src")

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

from db import get_all_flow_features


def main():
    records = get_all_flow_features()
    if len(records) < 20:
        print(f"학습 데이터가 너무 적습니다 (현재 {len(records)}건). 최소 20건 이상 필요합니다.")
        print("먼저 populate_flow_features_demo.py 를 실행해 데이터를 채워주세요.")
        return

    X = np.array([[r.speed_kmh, r.dx, r.dy, r.bbox_area] for r in records])
    print(f"학습 데이터 {len(X)}건으로 Isolation Forest 학습을 시작합니다.")

    # contamination: 전체 데이터 중 이상치로 볼 비율의 사전 추정치.
    # 도로 상황에서는 이상 케이스가 드물다고 가정해 5%로 설정.
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)

    joblib.dump(model, "models/isolation_forest.pkl")
    print("학습 완료: models/isolation_forest.pkl 저장됨")

    # 학습 데이터 자체에 대한 이상치 비율 확인 (참고용)
    preds = model.predict(X)
    anomaly_ratio = (preds == -1).mean()
    print(f"학습 데이터 중 이상치로 판정된 비율: {anomaly_ratio:.1%}")


if __name__ == "__main__":
    main()
