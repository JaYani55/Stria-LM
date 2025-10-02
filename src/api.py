#!/usr/bin/env python3
"""
FastAPI application for the AI Inference Server.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import tiktoken

try:
    from llama_cpp import Llama
except ImportError:
    # Create a dummy class for type hinting if llama_cpp is not installed
    class Llama: ...

from .config import MODEL_NAME

logger = logging.getLogger(__name__)

# --- Globals ---
# This is a simple way to share the model instance.
# In a more complex app, you might use a dedicated state management class.
class AppState:
    model_instance: Optional[Llama] = None

app_state = AppState()

# --- Pydantic Models for OpenAI-compatible API ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME, description="Model identifier")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(default=150, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(default=0.9, description="Top-p sampling")
    stream: Optional[bool] = Field(default=False, description="Stream the response")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str]

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

# --- FastAPI App and Endpoints ---

app = FastAPI(title="AI Inference Server", version="1.0.0")

@app.get("/")
async def root():
    return {
        "message": "AI Inference Server",
        "status": "running",
        "model_loaded": app_state.model_instance is not None
    }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint"""
    if app_state.model_instance is None:
        return {"object": "list", "data": []}
    
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    if app_state.model_instance is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    try:
        # Convert messages to a simple prompt format
        prompt = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in request.messages])
        prompt += "\nAssistant:"
        
        if request.stream:
            return StreamingResponse(
                stream_completion(prompt, request),
                media_type="text/event-stream"
            )
        else:
            return await generate_completion(prompt, request)
    
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_completion(prompt: str, request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Generate a non-streaming completion"""
    response = app_state.model_instance.create_completion(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop or ["User:", "System:"],
        stream=False
    )
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    content = response["choices"][0]["text"].strip()
    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessage(role="assistant", content=content),
        finish_reason=response["choices"][0]["finish_reason"]
    )
    
    # Estimate token usage
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(encoding.encode(content))
    except Exception:
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(content) // 4
    
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[choice],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    )

async def stream_completion(prompt: str, request: ChatCompletionRequest):
    """Generate a streaming completion"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    
    stream = app_state.model_instance.create_completion(
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop=request.stop or ["User:", "System:"],
        stream=True
    )
    
    for chunk in stream:
        if text := chunk["choices"][0]["text"]:
            stream_choice = ChatCompletionStreamChoice(
                index=0,
                delta={"content": text},
                finish_reason=None
            )
            stream_response = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[stream_choice]
            )
            yield f"data: {stream_response.json()}\n\n"
    
    final_choice = ChatCompletionStreamChoice(index=0, delta={}, finish_reason="stop")
    final_response = ChatCompletionStreamResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[final_choice]
    )
    yield f"data: {final_response.json()}\n\n"
    yield "data: [DONE]\n\n"
