from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ProjectCreate(BaseModel):
    project_name: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class AddData(BaseModel):
    prompt: str
    response: str

class ChatRequest(BaseModel):
    prompt: str
    top_k: int = 3

class ChatResponseItem(BaseModel):
    response_text: str
    original_prompt: str
    similarity_score: float

class AutoGenerateRequest(BaseModel):
    project_name: str
    url: str
    max_pages: int = 10
    business_context: Optional[str] = None

class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    domain: str

class PromptFileData(BaseModel):
    prompts: List[str]
