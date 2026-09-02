"""
이상탐지 결과 저장/조회 모듈 (PostgreSQL + SQLAlchemy)

접속 정보는 환경변수로 관리합니다 (비밀번호를 코드에 직접 넣지 않기 위함).
필요시 실행 전 아래처럼 설정하세요 (Windows cmd 기준):
    set ITS_DB_PASSWORD=본인비밀번호
    set ITS_DB_NAME=its          (선택, 기본값 its)
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, joinedload

DB_USER = os.environ.get("ITS_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("ITS_DB_PASSWORD", "")
DB_HOST = os.environ.get("ITS_DB_HOST", "localhost")
DB_PORT = os.environ.get("ITS_DB_PORT", "5432")
DB_NAME = os.environ.get("ITS_DB_NAME", "its")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Track(Base):
    """
    차량 추적 세션의 '원본' 테이블. 다른 모든 테이블은 이 테이블의 id를
    외래키(FK)로 참조한다 — track_id가 실제로 존재하는 추적인지 DB가 보장한다.
    """
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)     # 영상/카메라 소스 식별자
    started_at = Column(DateTime, default=datetime.now)

    detections = relationship("DetectionRecord", back_populates="track", cascade="all, delete-orphan")
    flow_features = relationship("FlowFeature", back_populates="track", cascade="all, delete-orphan")
    anomalies = relationship("AnomalyRecord", back_populates="track", cascade="all, delete-orphan")


class AnomalyRecord(Base):
    __tablename__ = "anomaly_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    flags = Column(String)          # "과속,역주행" 형태로 저장
    speed_kmh = Column(Float)
    plate_number = Column(String, nullable=True)
    detected_at = Column(DateTime, default=datetime.now)

    track = relationship("Track", back_populates="anomalies")


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True)  # 프레임 단독 검출은 트랙 없을 수도 있음
    image_path = Column(String, nullable=False)
    cls = Column(Integer)           # COCO 클래스 번호 (2=car, 5=bus, 7=truck 등)
    conf = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    detected_at = Column(DateTime, default=datetime.now)

    track = relationship("Track", back_populates="detections")


class FlowFeature(Base):
    """
    학습형 이상탐지(Isolation Forest) 학습용 특징 테이블.
    이상 여부와 무관하게 모든 트랙의 프레임별 이동 특징을 저장한다.
    """
    __tablename__ = "flow_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=False)
    speed_kmh = Column(Float)
    dx = Column(Float)              # x축 이동량 (픽셀)
    dy = Column(Float)              # y축 이동량 (픽셀)
    bbox_area = Column(Float)       # 검출 박스 크기 (차량 크기/거리 대리 지표)
    frame_idx = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)

    track = relationship("Track", back_populates="flow_features")


class OcrAttempt(Base):
    """
    번호판 OCR 시도마다 성공/실패를 기록한다 (인식률 통계용).
    성공 여부와 무관하게 시도할 때마다 1건씩 쌓여서, "몇 번 시도해서 몇 번 성공했는지"를 계산할 수 있다.
    """
    __tablename__ = "ocr_attempts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), nullable=True)
    raw_text = Column(String, nullable=True)       # EasyOCR이 실제로 읽은 원문
    parsed_plate = Column(String, nullable=True)   # 정규식 필터를 통과한 최종 번호판 (실패 시 NULL)
    success = Column(String, nullable=False)       # "success" / "fail" (Boolean 대신 문자열로, 조회 편의성 위함)
    attempted_at = Column(DateTime, default=datetime.now)


def init_db():
    Base.metadata.create_all(engine)


def get_or_create_track(source: str) -> int:
    """source(영상/이미지 경로)에 대한 Track을 찾거나 새로 만들고 id를 반환한다.
    정지 이미지 처리(populate_db.py)처럼 소스당 트랙 1개면 충분한 경우에 쓴다."""
    session = SessionLocal()
    track = session.query(Track).filter_by(source=source).first()
    if track is None:
        track = Track(source=source)
        session.add(track)
        session.commit()
        session.refresh(track)
    track_id = track.id
    session.close()
    return track_id


def create_track(source: str) -> int:
    """항상 새 Track 행을 만들고 id를 반환한다.
    영상(main.py) 처리처럼 같은 소스 안에 차량마다 별도 트랙이 필요한 경우에 쓴다."""
    session = SessionLocal()
    track = Track(source=source)
    session.add(track)
    session.commit()
    session.refresh(track)
    track_id = track.id
    session.close()
    return track_id


def save_anomaly(track_id: int, flags: list, speed_kmh: float, plate_number: str = None):
    session = SessionLocal()
    record = AnomalyRecord(
        track_id=track_id,
        flags=",".join(flags),
        speed_kmh=speed_kmh,
        plate_number=plate_number,
    )
    session.add(record)
    session.commit()
    session.close()


def get_recent_anomalies(limit: int = 50):
    """
    최근 이상탐지 기록을 조회한다. track 관계도 세션이 열려있는 동안 미리 함께 가져와서
    (joinedload), 세션이 닫힌 뒤에 r.track.source에 접근해도 에러가 나지 않게 한다.
    """
    session = SessionLocal()
    records = (
        session.query(AnomalyRecord)
        .options(joinedload(AnomalyRecord.track))
        .order_by(AnomalyRecord.detected_at.desc())
        .limit(limit)
        .all()
    )
    session.close()
    return records


def save_detection(image_path: str, cls: int, conf: float, bbox: list, track_id: int = None):
    session = SessionLocal()
    record = DetectionRecord(
        track_id=track_id,
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


def save_flow_feature(track_id: int, speed_kmh: float, dx: float, dy: float, bbox_area: float, frame_idx: int):
    session = SessionLocal()
    record = FlowFeature(
        track_id=track_id,
        speed_kmh=speed_kmh,
        dx=dx,
        dy=dy,
        bbox_area=bbox_area,
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


def save_ocr_attempt(raw_text: str, parsed_plate: str, track_id: int = None):
    """번호판 OCR 시도 결과를 기록한다. parsed_plate가 None이면 실패로 기록된다."""
    session = SessionLocal()
    record = OcrAttempt(
        track_id=track_id,
        raw_text=raw_text,
        parsed_plate=parsed_plate,
        success="success" if parsed_plate else "fail",
    )
    session.add(record)
    session.commit()
    session.close()


def get_ocr_stats():
    """번호판 인식 성공/실패 건수와 성공률을 계산해서 반환한다."""
    session = SessionLocal()
    total = session.query(OcrAttempt).count()
    success = session.query(OcrAttempt).filter_by(success="success").count()
    session.close()
    fail = total - success
    rate = (success / total * 100) if total > 0 else 0.0
    return {"total": total, "success": success, "fail": fail, "success_rate": round(rate, 1)}


def get_recent_ocr_attempts(limit: int = 50):
    session = SessionLocal()
    records = (
        session.query(OcrAttempt)
        .order_by(OcrAttempt.attempted_at.desc())
        .limit(limit)
        .all()
    )
    session.close()
    return records


if __name__ == "__main__":
    init_db()
    print(f"DB 초기화 완료: {DB_HOST}:{DB_PORT}/{DB_NAME} (테이블: tracks, anomaly_records, detection_records, flow_features, ocr_attempts)")
