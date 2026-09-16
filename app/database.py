from datetime import datetime
from pathlib import Path
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class SignalRecord(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ticker: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    confidence: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[str] = mapped_column(Text)


class OrderRecord(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ticker: Mapped[str] = mapped_column(String(16))
    side: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))
    mode: Mapped[str] = mapped_column(String(16), default="paper")


def create_database(url: str = "sqlite:///brokeria.db"):
    if url.startswith("sqlite:///"):
        Path(url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)
