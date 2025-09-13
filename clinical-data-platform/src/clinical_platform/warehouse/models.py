from __future__ import annotations

from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Subject(Base):
    __tablename__ = "dim_subject"
    subject_sk: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject_id: Mapped[str] = mapped_column(String)
    arm: Mapped[str | None] = mapped_column(String, nullable=True)
    sex: Mapped[str | None] = mapped_column(String, nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)


def make_engine(db_path: str) -> any:
    # SQLAlchemy support for DuckDB is limited; use duckdb directly typically.
    url = f"duckdb:///{db_path}"
    return create_engine(url)

