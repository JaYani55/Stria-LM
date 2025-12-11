from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, Float, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    embedding_model: Mapped[str] = mapped_column(String)
    vector_dimension: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    qa_pairs: Mapped[List["QAPair"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    scraped_content: Mapped[List["ScrapedContent"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    prompt_files: Mapped[List["PromptFile"]] = relationship(back_populates="project", cascade="all, delete-orphan")

class QAPair(Base):
    __tablename__ = "qa_pairs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    prompt_text: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    embedding: Mapped[Vector] = mapped_column(Vector)  # Generic Vector to support different dimensions

    project: Mapped["Project"] = relationship(back_populates="qa_pairs")

class ScrapedContent(Base):
    __tablename__ = "scraped_content"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    url: Mapped[str] = mapped_column(String)
    title: Mapped[Optional[str]] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped["Project"] = relationship(back_populates="scraped_content")

class PromptFile(Base):
    __tablename__ = "prompt_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    prompt_data: Mapped[str] = mapped_column(Text) # JSON string
    business_context: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    project: Mapped["Project"] = relationship(back_populates="prompt_files")
