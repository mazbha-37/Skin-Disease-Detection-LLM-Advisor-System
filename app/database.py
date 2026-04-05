from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///skin_advisor.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class AnalysisLog(Base):
    __tablename__ = "analysis_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(String)
    disease = Column(String)
    confidence = Column(Float)
    llm_used = Column(Boolean)
    processing_time_ms = Column(Float)


def init_db():
    Base.metadata.create_all(bind=engine)


def log_analysis(
    disease: str,
    confidence: float,
    llm_used: bool,
    processing_time_ms: float,
):
    db = SessionLocal()
    try:
        entry = AnalysisLog(
            created_at=datetime.now(timezone.utc).isoformat(),
            disease=disease,
            confidence=confidence,
            llm_used=llm_used,
            processing_time_ms=processing_time_ms,
        )
        db.add(entry)
        db.commit()
    finally:
        db.close()
