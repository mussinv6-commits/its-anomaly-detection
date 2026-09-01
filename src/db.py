"""
이상탐지 결과 저장/조회 모듈 (PostgreSQL + SQLAlchemy)

접속 정보는 환경변수로 관리합니다 (비밀번호를 코드에 직접 넣지 않기 위함).
필요시 실행 전 아래처럼 설정하세요 (Windows cmd 기준):
    set ITS_DB_PASSWORD=본인비밀번호
    set ITS_DB_NAME=its          (선택, 기본값 its)
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

DB_USER = os.environ.get("ITS_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("ITS_DB_PASSWORD", "")
DB_HOST = os.environ.get("ITS_DB_HOST", "localhost")
DB_PORT = os.environ.get("ITS_DB_PORT", "5432")
DB_NAME = os.environ.get("ITS_DB_NAME", "its")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
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


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_path = Column(String, nullable=False)
    cls = Column(Integer)           # COCO 클래스 번호 (2=car, 5=bus, 7=truck 등)
    conf = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    detected_at = Column(DateTime, default=datetime.utcnow)


class FlowFeature(Base):
    """
    학습형 이상탐지(Isolation Forest) 학습용 특징 테이블.
    이상 여부와 무관하게 모든 트랙의 프레임별 이동 특징을 저장한다.
    """
    __tablename__ = "flow_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, nullable=False)
    speed_kmh = Column(Float)
    dx = Column(Float)              # x축 이동량 (픽셀)
    dy = Column(Float)              # y축 이동량 (픽셀)
    bbox_area = Column(Float)       # 검출 박스 크기 (차량 크기/거리 대리 지표)
    source = Column(String)
    frame_idx = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


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


def save_detection(image_path: str, cls: int, conf: float, bbox: list):
    session = SessionLocal()
    record = DetectionRecord(
        image_path=image_path,
        cls=cls,
        conf=conf,
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
    )
    session.add(record)
    session.commit()
    session.close()


def get_recent_detections(limit: int = 50):
    session = SessionLocal()
    records = (
        session.query(DetectionRecord)
        .order_by(DetectionRecord.detected_at.desc())
        .limit(limit)
        .all()
    )
    session.close()
    return records


def save_flow_feature(track_id: int, speed_kmh: float, dx: float, dy: float, bbox_area: float, source: str, frame_idx: int):
    session = SessionLocal()
    record = FlowFeature(
        track_id=track_id,
        speed_kmh=speed_kmh,
        dx=dx,
        dy=dy,
        bbox_area=bbox_area,
        source=source,
        frame_idx=frame_idx,
    )
    session.add(record)
    session.commit()
    session.close()


def get_all_flow_features():
    """학습용: 저장된 모든 flow_features를 반환한다."""
    session = SessionLocal()
    records = session.query(FlowFeature).all()
    session.close()
    return records


if __name__ == "__main__":
    init_db()
    print(f"DB 초기화 완료: {DB_HOST}:{DB_PORT}/{DB_NAME} (테이블: anomaly_records, detection_records, flow_features)")
