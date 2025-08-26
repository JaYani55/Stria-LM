from pydantic import BaseModel
from typing import List

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
