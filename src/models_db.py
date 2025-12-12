from datetime import datetime
from typing import Optional, List
import uuid
from sqlalchemy import String, Integer, Float, Text, ForeignKey, DateTime, func, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB, UUID
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
    actors: Mapped[List["Actor"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    personas: Mapped[List["Persona"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    chat_sessions: Mapped[List["ChatSession"]] = relationship(back_populates="project", cascade="all, delete-orphan")

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


# =============================================================================
# ACTOR - LLM persona with full generation parameters
# =============================================================================

class Actor(Base):
    """
    LLM Actor configuration with full generation parameters.
    Used for AI assistants/chatbots with specific personas and model settings.
    """
    __tablename__ = "actors"

    actor_id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    actor_name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Prompt messages as JSONB: [{key, role, content}, ...]
    prompt_messages: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    # Model configuration
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Generation parameters
    temperature: Mapped[Optional[float]] = mapped_column(Float)
    max_tokens: Mapped[Optional[int]] = mapped_column(Integer)
    top_p: Mapped[Optional[float]] = mapped_column(Float)
    top_k: Mapped[Optional[int]] = mapped_column(Integer)
    repetition_penalty: Mapped[Optional[float]] = mapped_column(Float)
    
    # Additional generation parameters as JSONB
    other_generation_parameters: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    project: Mapped["Project"] = relationship(back_populates="actors")
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="actor", 
        foreign_keys="ChatSession.actor_id"
    )


# =============================================================================
# PERSONA - Simplified profile for human users or AI impersonation fallback
# =============================================================================

class Persona(Base):
    """
    Simplified persona for human users or AI impersonation.
    Contains basic identity info without LLM generation parameters.
    """
    __tablename__ = "personas"

    persona_id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    persona_name: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(String(255))
    description: Mapped[Optional[str]] = mapped_column(Text)
    avatar_url: Mapped[Optional[str]] = mapped_column(String(512))
    
    # Whether this is an AI-impersonated persona or real human
    is_ai: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Optional fallback actor for AI impersonation
    fallback_actor_id: Mapped[Optional[int]] = mapped_column(ForeignKey("actors.actor_id"))
    
    # Additional metadata as JSONB (named extra_data to avoid SQLAlchemy reserved name)
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    project: Mapped["Project"] = relationship(back_populates="personas")
    fallback_actor: Mapped[Optional["Actor"]] = relationship(foreign_keys=[fallback_actor_id])
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="persona",
        foreign_keys="ChatSession.persona_id"
    )


# =============================================================================
# CHAT SESSION - Conversation thread between persona and actor
# =============================================================================

class ChatSession(Base):
    """
    A chat session linking a persona (user) with an actor (AI) in a project.
    Contains all messages and token usage metadata.
    """
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    persona_id: Mapped[str] = mapped_column(ForeignKey("personas.persona_id"))
    actor_id: Mapped[int] = mapped_column(ForeignKey("actors.actor_id"))
    
    title: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Token usage tracking
    total_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    
    # Session status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    project: Mapped["Project"] = relationship(back_populates="chat_sessions")
    persona: Mapped["Persona"] = relationship(back_populates="chat_sessions", foreign_keys=[persona_id])
    actor: Mapped["Actor"] = relationship(back_populates="chat_sessions", foreign_keys=[actor_id])
    messages: Mapped[List["ChatMessage"]] = relationship(
        back_populates="session", 
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at"
    )


# =============================================================================
# CHAT MESSAGE - Individual message in a chat session
# =============================================================================

class ChatMessage(Base):
    """
    Individual message within a chat session.
    Tracks token counts for context length management.
    """
    __tablename__ = "chat_messages"

    message_id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.session_id"))
    
    # Message role: "user", "assistant", "system", "context"
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Token counts for this message
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Context metadata (QA pairs used, similarity scores, etc.)
    context_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    # Generation metadata (model used, parameters, etc.)
    generation_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["ChatSession"] = relationship(back_populates="messages")
