from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ProjectCreate(BaseModel):
    project_name: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class AddData(BaseModel):
    prompt: str
    response: str
    weight: float = 1.0

class ChatRequest(BaseModel):
    prompt: str
    top_k: int = 3

class ChatResponseItem(BaseModel):
    id: int
    response_text: str
    original_prompt: str
    similarity_score: float
    weight: Optional[float] = None
    weighted_similarity: Optional[float] = None

class AutoGenerateRequest(BaseModel):
    project_name: str
    url: str
    max_pages: int = 10
    business_context: Optional[str] = None
    default_weight: float = 1.0

class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    domain: str

class PromptFileData(BaseModel):
    prompts: List[str]


# =============================================================================
# Agent Models
# =============================================================================

class AgentStepResponse(BaseModel):
    """Response model for an agent workflow step."""
    name: str
    description: str
    status: str
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AgentResultResponse(BaseModel):
    """Response model for agent execution results."""
    success: bool
    message: str
    steps: List[AgentStepResponse]
    data: Dict[str, Any] = {}


class BusinessWebsiteScraperRequest(BaseModel):
    """Request model for the Business Website Scraper agent."""
    project_name: str = Field(..., description="Target project name")
    url: str = Field(..., description="Website URL to scrape")
    max_pages: int = Field(10, ge=1, le=100, description="Maximum pages to scrape")
    business_context: Optional[str] = Field(None, description="Optional context about the business")
    default_weight: float = Field(1.0, ge=0.1, le=10.0, description="Default weight for Q&A pairs")


# =============================================================================
# Actor Models
# =============================================================================

class PromptMessage(BaseModel):
    """A single prompt message for an actor's system prompts."""
    key: Optional[str] = Field(None, description="Identifier for this prompt message")
    role: str = Field(..., description="Message role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class ActorCreate(BaseModel):
    """Request model for creating an actor."""
    actor_name: str = Field(..., description="Unique name for this actor")
    description: Optional[str] = Field("", description="Description of the actor's purpose")
    prompt_messages: List[PromptMessage] = Field(
        default_factory=list, 
        description="List of system/context messages for the actor"
    )
    model_name: str = Field("gpt-4", description="LLM model identifier")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2048, ge=1, le=128000, description="Maximum response tokens")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(None, ge=1, description="Top-k sampling parameter")
    repetition_penalty: Optional[float] = Field(None, ge=0.0, description="Repetition penalty")
    other_generation_parameters: Optional[Dict[str, Any]] = Field(
        None, description="Additional generation parameters"
    )


class ActorUpdate(BaseModel):
    """Request model for updating an actor."""
    actor_name: Optional[str] = None
    description: Optional[str] = None
    prompt_messages: Optional[List[PromptMessage]] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=128000)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)
    repetition_penalty: Optional[float] = Field(None, ge=0.0)
    other_generation_parameters: Optional[Dict[str, Any]] = None


class ActorResponse(BaseModel):
    """Response model for an actor."""
    actor_id: str
    actor_name: str
    description: Optional[str] = None
    prompt_messages: List[Dict[str, Any]] = []
    model_name: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    other_generation_parameters: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# =============================================================================
# Persona Models
# =============================================================================

class PersonaCreate(BaseModel):
    """Request model for creating a persona."""
    persona_name: str = Field(..., description="Unique identifier for this persona")
    display_name: str = Field(..., description="Display name shown in chat")
    is_ai: bool = Field(False, description="Whether this persona represents an AI")
    fallback_actor_id: Optional[str] = Field(None, description="Optional actor ID for AI personas")


class PersonaResponse(BaseModel):
    """Response model for a persona."""
    persona_id: str
    persona_name: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    avatar_url: Optional[str] = None
    is_ai: bool = False
    fallback_actor_id: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


# =============================================================================
# Chat Session Models
# =============================================================================

class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""
    actor_id: str = Field(..., description="UUID of the actor for this session")
    persona_id: str = Field(..., description="UUID of the persona (user)")
    session_name: Optional[str] = Field(None, description="Optional name for the session")


class ChatSessionResponse(BaseModel):
    """Response model for a chat session."""
    session_id: str
    actor_id: str
    persona_id: str
    title: Optional[str] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# =============================================================================
# Chat Message Models
# =============================================================================

class LLMChatRequest(BaseModel):
    """Request model for sending a chat message to the LLM."""
    message: str = Field(..., description="User message to send")
    use_context: bool = Field(True, description="Whether to retrieve QA context")
    context_top_k: int = Field(5, ge=1, le=20, description="Number of context items to retrieve")
    max_context_tokens: int = Field(2000, ge=100, le=16000, description="Max tokens for context")
    max_history_tokens: int = Field(4000, ge=100, le=32000, description="Max tokens for chat history")


class LLMChatResponse(BaseModel):
    """Response model for LLM chat."""
    message: str = Field(..., description="Assistant's response message")
    session_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    context_used: int = Field(..., description="Number of QA pairs used as context")
    model: str


class ChatMessageResponse(BaseModel):
    """Response model for a chat message."""
    message_id: str
    role: str
    content: str
    token_count: int = 0
    context_metadata: Optional[Dict[str, Any]] = None
    generation_metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    session_id: str
    messages: List[ChatMessageResponse]
    total_messages: int
