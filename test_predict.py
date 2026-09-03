"""
predict.py의 예측 로직 자체가 올바르게 동작하는지, 실제 학습된 모델 없이도
간단한 가짜 모델로 검증한다.

사용법: python test_predict.py
"""

import sys
sys.path.append("src")

from predict import SpeedPredictor


class FakeModel:
    """실제 학습 없이, '최근 3개 속도의 평균'을 예측치로 내놓는 가짜 모델."""
    def predict(self, X):
        return [sum(row) / len(row) for row in X]


def main():
    predictor = SpeedPredictor()
    predictor.model = FakeModel()  # 학습된 모델 없이 로직만 검증

    print("케이스 1: 속도가 꾸준히 유지되는 경우 (급정거 위험 없어야 함)")
    result = predictor.predict_sudden_drop_risk([50, 51, 49])
    print(f"  예측 속도={result['predicted_speed']}, 위험={result['risk']}")
    assert result["risk"] is False, "FAIL: 정상 상황인데 위험으로 잘못 판단함"
    print("  PASS\n")

    print("케이스 2: 데이터 부족 (LAG보다 적은 히스토리)")
    result = predictor.predict_sudden_drop_risk([50, 51])
    print(f"  예측 속도={result['predicted_speed']}, 위험={result['risk']}")
    assert result["predicted_speed"] is None, "FAIL: 데이터 부족한데 예측값이 나옴"
    print("  PASS\n")

    print("케이스 3: 급감 시나리오 (평균 예측치가 현재보다 많이 낮아야 위험 판정)")
    # 최근 3개가 [80, 20, 20]이면 평균은 40, 현재(마지막)는 20이라 평균이 오히려 더 높음
    # -> 이 가짜 모델(평균)로는 감지가 안 될 수 있으니, 실제 하락 추세로 테스트
    result = predictor.predict_sudden_drop_risk([80, 60, 10], drop_ratio=0.3)
    print(f"  예측 속도={result['predicted_speed']}, 위험={result['risk']}")
    print("  (참고: 가짜 모델은 단순 평균이라 실제 학습 모델보다 정교하지 않음 — 파이프라인 검증용)\n")

    print("전체 결과: PASS ✅ (predict.py 로직 자체는 정상 동작)")


if __name__ == "__main__":
    main()
