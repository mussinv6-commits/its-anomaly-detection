"""
이상탐지 결과 저장/조회 모듈 (SQLite + SQLAlchemy)
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()
engine = create_engine("sqlite:///data/its.db")
SessionLocal = sessionmaker(bind=engine)


class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, nullable=False)
    flags = Column(String)          # "과속,역주행" 형태로 저장
    speed_kmh = Column(Float)
    plate_number = Column(String, nullable=True)
    source = Column(String)         # 영상/카메라 소스 식별자
    detected_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def save_anomaly(track_id: int, flags: list, speed_kmh: float, source: str, plate_number: str = None):
    session = SessionLocal()
    record = AnomalyRecord(
        track_id=track_id,
        flags=",".join(flags),
        speed_kmh=speed_kmh,
        plate_number=plate_number,
        source=source,
    )
    session.add(record)
    session.commit()
    session.close()


def get_recent_anomalies(limit: int = 50):
    session = SessionLocal()
    records = (
        session.query(AnomalyRecord)
        .order_by(AnomalyRecord.detected_at.desc())
        .limit(limit)
        .all()
    )
    session.close()
    return records


if __name__ == "__main__":
    init_db()
    print("DB 초기화 완료: data/its.db")
