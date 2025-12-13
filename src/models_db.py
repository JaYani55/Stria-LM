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
    scripts: Mapped[List["ProjectScript"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    schema_versions: Mapped[List["SchemaVersion"]] = relationship(back_populates="project", cascade="all, delete-orphan")

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


# =============================================================================
# PROJECT SCRIPTS - Per-project custom scripts registry
# =============================================================================

class ProjectScript(Base):
    """
    Registry of custom scripts for a project.
    Scripts can be scrapers, data-manipulation, ai-scripts, or migrations.
    """
    __tablename__ = "project_scripts"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    script_name: Mapped[str] = mapped_column(String(255), nullable=False)
    script_type: Mapped[str] = mapped_column(String(50), nullable=False)  # scraper, data-manipulation, ai-script, migration
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Additional metadata as JSONB (dependencies, schedule, etc.)
    script_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    project: Mapped["Project"] = relationship(back_populates="scripts")
    execution_logs: Mapped[List["ScriptExecutionLog"]] = relationship(
        back_populates="script", 
        cascade="all, delete-orphan"
    )


class ScriptExecutionLog(Base):
    """
    Execution log for project scripts.
    Tracks script runs with status, output, and timing.
    """
    __tablename__ = "script_execution_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    script_id: Mapped[int] = mapped_column(ForeignKey("project_scripts.id"))
    status: Mapped[str] = mapped_column(String(50), nullable=False)  # running, success, failed, cancelled
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer)
    stdout: Mapped[Optional[str]] = mapped_column(Text)
    stderr: Mapped[Optional[str]] = mapped_column(Text)
    
    # Additional execution metadata
    execution_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)

    script: Mapped["ProjectScript"] = relationship(back_populates="execution_logs")


class SchemaVersion(Base):
    """
    Tracks applied migration scripts for version control.
    """
    __tablename__ = "schema_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    script_name: Mapped[str] = mapped_column(String(255), nullable=False)
    applied_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    checksum: Mapped[Optional[str]] = mapped_column(String(64))  # SHA-256 hash of script content

    project: Mapped["Project"] = relationship(back_populates="schema_versions")
