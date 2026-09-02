"""
DB 초기화 + 테스트 데이터 저장/조회 확인
사용법: python test_db.py
"""

import sys
sys.path.append("src")

from db import init_db, save_anomaly, get_recent_anomalies

# 1. DB 초기화 (data/its.db 파일 생성)
init_db()
print("DB 초기화 완료: data/its.db")

# 2. 테스트 데이터 저장 (방금 검출 테스트 결과를 흉내낸 가짜 이상탐지 기록)
save_anomaly(
    track_id=1,
    flags=["과속"],
    speed_kmh=87.5,
    source="data/raw/images/00 (10).jpg",
    plate_number=None,
)
print("테스트 레코드 저장 완료")

# 3. 저장된 데이터 조회
records = get_recent_anomalies(limit=10)
print(f"\n저장된 레코드 수: {len(records)}")
for r in records:
    print(
        f"- id={r.id}, track_id={r.track_id}, flags={r.flags}, "
        f"speed={r.speed_kmh}km/h, source={r.source}, at={r.detected_at}"
    )
