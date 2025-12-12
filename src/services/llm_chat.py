"""
LLM Chat Service for Stria-LM.
Handles chat interactions with LLM APIs using QA pairs as context.
"""

import json
import logging
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import httpx

from .token_counter import TokenCounter, count_tokens, count_message_tokens

# Configure logger for this module
logger = logging.getLogger(__name__)

# Ensure logs appear in console (useful for debugging)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


class LLMChatService:
    """
    Service for managing LLM chat interactions.
    Retrieves relevant QA pairs as context and manages chat history.
    """
    
    def __init__(
        self,
        db_backend,
        inference_base_url: str = "http://localhost:8080/v1",
        api_key: Optional[str] = None,
        default_model: str = "gpt-4"
    ):
        """
        Initialize the LLM Chat Service.
        
        Args:
            db_backend: Database backend instance
            inference_base_url: Base URL for OpenAI-compatible API
            api_key: API key for the inference server
            default_model: Default model to use
        """
        self.db = db_backend
        self.inference_base_url = inference_base_url.rstrip('/')
        self.api_key = api_key
        self.default_model = default_model
        self.token_counter = TokenCounter(default_model)
        
        # Log service initialization
        logger.info(f"[LLM Chat] Service initialized")
        logger.info(f"[LLM Chat] Inference URL: {self.inference_base_url}")
        logger.info(f"[LLM Chat] Default model: {self.default_model}")
        logger.info(f"[LLM Chat] API key configured: {'Yes' if self.api_key else 'No'}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _build_context_from_qa(
        self,
        project_name: str,
        query: str,
        top_k: int = 5,
        max_context_tokens: int = 2000
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build context string from relevant QA pairs.
        
        Args:
            project_name: Name of the project
            query: User query to find relevant context for
            top_k: Number of top matches to retrieve
            max_context_tokens: Maximum tokens for context
            
        Returns:
            Tuple of (context_string, context_metadata)
        """
        from ..embedding import generate_embedding
        
        # Get project metadata for embedding model
        project = self.db.get_project(project_name)
        if not project:
            return "", []
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query, project.get("embedding_model"))
        
        # Find similar prompts
        similar = self.db.find_similar_prompts(project_name, query_embedding, top_k=top_k)
        
        if not similar:
            return "", []
        
        # Build context string
        context_parts = []
        context_metadata = []
        total_tokens = 0
        
        for item in similar:
            # Format as Q&A pair
            qa_text = f"Q: {item.get('original_prompt', '')}\nA: {item.get('response_text', '')}"
            
            # Count tokens
            item_tokens = count_tokens(qa_text, self.default_model)
            
            if total_tokens + item_tokens > max_context_tokens:
                break
            
            context_parts.append(qa_text)
            context_metadata.append({
                "id": item.get("id"),
                "similarity": round(item.get("similarity_score", 0), 4),
                "weighted_similarity": round(item.get("weighted_similarity", 0), 4),
                "tokens": item_tokens
            })
            total_tokens += item_tokens
        
        context_string = "\n\n---\n\n".join(context_parts)
        
        return context_string, context_metadata
    
    def _build_messages(
        self,
        actor: Dict[str, Any],
        context: str,
        chat_history: List[Dict[str, Any]],
        user_message: str
    ) -> List[Dict[str, str]]:
        """
        Build the full message list for the LLM API.
        
        Args:
            actor: Actor configuration with prompt_messages
            context: Retrieved context from QA pairs
            chat_history: Previous chat messages
            user_message: Current user message
            
        Returns:
            List of message dicts for the API
        """
        messages = []
        
        # Add actor's system prompts
        prompt_messages = actor.get("prompt_messages", [])
        for msg in prompt_messages:
            if msg.get("role") and msg.get("content"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add context as a system message if available
        if context:
            context_message = (
                "Use the following knowledge base information to help answer questions:\n\n"
                f"{context}\n\n"
                "If the information above is relevant, use it in your response. "
                "If not, respond based on your general knowledge."
            )
            messages.append({
                "role": "system",
                "content": context_message
            })
        
        # Add chat history
        for msg in chat_history:
            if msg.get("role") in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    async def chat_async(
        self,
        project_name: str,
        session_id: str,
        user_message: str,
        use_context: bool = True,
        context_top_k: int = 5,
        max_context_tokens: int = 2000,
        max_history_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Send a chat message and get a response (async version).
        
        Args:
            project_name: Name of the project
            session_id: Chat session ID
            user_message: User's message
            use_context: Whether to retrieve QA context
            context_top_k: Number of context items to retrieve
            max_context_tokens: Max tokens for context
            max_history_tokens: Max tokens for chat history
            
        Returns:
            Dict with response, token counts, and metadata
        """
        # Get session and actor
        session = self.db.get_chat_session(project_name, session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        
        actor = self.db.get_actor(project_name, session["actor_id"])
        if not actor:
            raise ValueError(f"Actor '{session['actor_id']}' not found")
        
        # Update token counter for this model
        model_name = actor.get("model_name", self.default_model)
        self.token_counter = TokenCounter(model_name)
        
        # Get context from QA pairs
        context_string = ""
        context_metadata = []
        if use_context:
            context_string, context_metadata = self._build_context_from_qa(
                project_name, user_message, context_top_k, max_context_tokens
            )
        
        # Get chat history within token budget
        chat_history = self.db.get_chat_context_window(
            project_name, session_id, max_history_tokens
        )
        
        # Build messages
        messages = self._build_messages(actor, context_string, chat_history, user_message)
        
        # Count input tokens
        input_tokens = count_message_tokens(messages, model_name)
        
        # Save user message
        user_msg_tokens = count_tokens(user_message, model_name)
        self.db.add_chat_message(
            project_name,
            session_id,
            role="user",
            content=user_message,
            token_count=user_msg_tokens,
            context_used={"qa_pairs": context_metadata} if context_metadata else None
        )
        
        # Build request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": actor.get("temperature", 0.7),
            "max_tokens": actor.get("max_tokens", 2048)
        }
        
        if actor.get("top_p"):
            payload["top_p"] = actor["top_p"]
        
        # Add any other generation parameters
        other_params = actor.get("other_generation_parameters", {})
        if other_params:
            payload.update(other_params)
        
        # Make API request
        start_time = datetime.now()
        api_url = f"{self.inference_base_url}/chat/completions"
        
        # Log connection attempt and request details
        logger.info(f"[LLM Chat] Connecting to: {api_url}")
        logger.debug(f"[LLM Chat] Request headers: {json.dumps({k: '***' if 'auth' in k.lower() else v for k, v in self._get_headers().items()}, indent=2)}")
        logger.info(f"[LLM Chat] Model: {model_name}, Temperature: {actor.get('temperature', 0.7)}, Max tokens: {actor.get('max_tokens', 2048)}")
        logger.debug(f"[LLM Chat] Messages structure ({len(messages)} messages):")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content_preview = msg.get('content', '')[:200] + ('...' if len(msg.get('content', '')) > 200 else '')
            logger.debug(f"  [{i+1}] {role}: {content_preview}")
        
        try:
            # Configure explicit timeouts: connect=10s, read=120s, write=30s
            timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"[LLM Chat] Sending request (timeout: 120s)...")
                response = await client.post(
                    api_url,
                    headers=self._get_headers(),
                    json=payload
                )
                logger.info(f"[LLM Chat] Response status: {response.status_code}")
                response.raise_for_status()
                result = response.json()
                logger.info(f"[LLM Chat] Successfully received response from LLM")
        except httpx.TimeoutException as e:
            logger.error(f"[LLM Chat] Request timed out: {str(e)}")
            logger.error(f"[LLM Chat] The inference server took too long to respond")
            raise ValueError(f"Request timed out after 120 seconds. The model '{model_name}' may be unavailable or overloaded.")
        except httpx.ConnectError as e:
            logger.error(f"[LLM Chat] Connection failed: {str(e)}")
            logger.error(f"[LLM Chat] Could not connect to {api_url}")
            logger.error(f"[LLM Chat] Please verify the inference server is running and accessible")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"[LLM Chat] HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[LLM Chat] Unexpected error: {type(e).__name__}: {str(e)}")
            raise
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract response
        assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Get token counts from response or estimate
        usage = result.get("usage", {})
        response_input_tokens = usage.get("prompt_tokens", input_tokens)
        response_output_tokens = usage.get("completion_tokens", count_tokens(assistant_message, model_name))
        
        # Save assistant message
        generation_metadata = {
            "model": model_name,
            "input_tokens": response_input_tokens,
            "output_tokens": response_output_tokens,
            "latency_ms": round(latency_ms, 2),
            "temperature": actor.get("temperature"),
            "finish_reason": result.get("choices", [{}])[0].get("finish_reason")
        }
        
        self.db.add_chat_message(
            project_name,
            session_id,
            role="assistant",
            content=assistant_message,
            token_count=response_output_tokens,
            generation_metadata=generation_metadata
        )
        
        # Update session token counts
        self.db.update_session_tokens(
            project_name,
            session_id,
            response_input_tokens,
            response_output_tokens
        )
        
        return {
            "message": assistant_message,
            "session_id": session_id,
            "input_tokens": response_input_tokens,
            "output_tokens": response_output_tokens,
            "total_tokens": response_input_tokens + response_output_tokens,
            "latency_ms": round(latency_ms, 2),
            "context_used": len(context_metadata),
            "model": model_name
        }
    
    def chat(
        self,
        project_name: str,
        session_id: str,
        user_message: str,
        use_context: bool = True,
        context_top_k: int = 5,
        max_context_tokens: int = 2000,
        max_history_tokens: int = 4000
    ) -> Dict[str, Any]:
        """
        Send a chat message and get a response (sync version).
        Uses httpx sync client for synchronous operation.
        """
        import asyncio
        
        # Try to run in existing event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None:
            # We're in an async context, need to use sync httpx
            return self._chat_sync(
                project_name, session_id, user_message,
                use_context, context_top_k, max_context_tokens, max_history_tokens
            )
        else:
            # Run async version
            return asyncio.run(self.chat_async(
                project_name, session_id, user_message,
                use_context, context_top_k, max_context_tokens, max_history_tokens
            ))
    
    def _chat_sync(
        self,
        project_name: str,
        session_id: str,
        user_message: str,
        use_context: bool = True,
        context_top_k: int = 5,
        max_context_tokens: int = 2000,
        max_history_tokens: int = 4000
    ) -> Dict[str, Any]:
        """Synchronous implementation of chat."""
        # Get session and actor
        session = self.db.get_chat_session(project_name, session_id)
        if not session:
            raise ValueError(f"Session '{session_id}' not found")
        
        actor = self.db.get_actor(project_name, session["actor_id"])
        if not actor:
            raise ValueError(f"Actor '{session['actor_id']}' not found")
        
        model_name = actor.get("model_name", self.default_model)
        
        # Get context from QA pairs
        context_string = ""
        context_metadata = []
        if use_context:
            context_string, context_metadata = self._build_context_from_qa(
                project_name, user_message, context_top_k, max_context_tokens
            )
        
        # Get chat history
        chat_history = self.db.get_chat_context_window(
            project_name, session_id, max_history_tokens
        )
        
        # Build messages
        messages = self._build_messages(actor, context_string, chat_history, user_message)
        
        # Count input tokens
        input_tokens = count_message_tokens(messages, model_name)
        
        # Save user message
        user_msg_tokens = count_tokens(user_message, model_name)
        self.db.add_chat_message(
            project_name,
            session_id,
            role="user",
            content=user_message,
            token_count=user_msg_tokens,
            context_used={"qa_pairs": context_metadata} if context_metadata else None
        )
        
        # Build request payload
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": actor.get("temperature", 0.7),
            "max_tokens": actor.get("max_tokens", 2048)
        }
        
        if actor.get("top_p"):
            payload["top_p"] = actor["top_p"]
        
        other_params = actor.get("other_generation_parameters", {})
        if other_params:
            payload.update(other_params)
        
        # Make sync API request
        start_time = datetime.now()
        api_url = f"{self.inference_base_url}/chat/completions"
        
        # Log connection attempt and request details
        logger.info(f"[LLM Chat] Connecting to: {api_url}")
        logger.debug(f"[LLM Chat] Request headers: {json.dumps({k: '***' if 'auth' in k.lower() else v for k, v in self._get_headers().items()}, indent=2)}")
        logger.info(f"[LLM Chat] Model: {model_name}, Temperature: {actor.get('temperature', 0.7)}, Max tokens: {actor.get('max_tokens', 2048)}")
        logger.debug(f"[LLM Chat] Messages structure ({len(messages)} messages):")
        for i, msg in enumerate(messages):
            role = msg.get('role', 'unknown')
            content_preview = msg.get('content', '')[:200] + ('...' if len(msg.get('content', '')) > 200 else '')
            logger.debug(f"  [{i+1}] {role}: {content_preview}")
        
        try:
            # Configure explicit timeouts: connect=10s, read=120s, write=30s
            timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)
            with httpx.Client(timeout=timeout) as client:
                logger.info(f"[LLM Chat] Sending request (timeout: 120s)...")
                response = client.post(
                    api_url,
                    headers=self._get_headers(),
                    json=payload
                )
                logger.info(f"[LLM Chat] Response status: {response.status_code}")
                response.raise_for_status()
                result = response.json()
                logger.info(f"[LLM Chat] Successfully received response from LLM")
        except httpx.TimeoutException as e:
            logger.error(f"[LLM Chat] Request timed out: {str(e)}")
            logger.error(f"[LLM Chat] The inference server took too long to respond")
            raise ValueError(f"Request timed out after 120 seconds. The model '{model_name}' may be unavailable or overloaded.")
        except httpx.ConnectError as e:
            logger.error(f"[LLM Chat] Connection failed: {str(e)}")
            logger.error(f"[LLM Chat] Could not connect to {api_url}")
            logger.error(f"[LLM Chat] Please verify the inference server is running and accessible")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"[LLM Chat] HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"[LLM Chat] Unexpected error: {type(e).__name__}: {str(e)}")
            raise
        
        end_time = datetime.now()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Extract response
        assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Get token counts
        usage = result.get("usage", {})
        response_input_tokens = usage.get("prompt_tokens", input_tokens)
        response_output_tokens = usage.get("completion_tokens", count_tokens(assistant_message, model_name))
        
        # Save assistant message
        generation_metadata = {
            "model": model_name,
            "input_tokens": response_input_tokens,
            "output_tokens": response_output_tokens,
            "latency_ms": round(latency_ms, 2),
            "temperature": actor.get("temperature"),
            "finish_reason": result.get("choices", [{}])[0].get("finish_reason")
        }
        
        self.db.add_chat_message(
            project_name,
            session_id,
            role="assistant",
            content=assistant_message,
            token_count=response_output_tokens,
            generation_metadata=generation_metadata
        )
        
        # Update session token counts
        self.db.update_session_tokens(
            project_name,
            session_id,
            response_input_tokens,
            response_output_tokens
        )
        
        return {
            "message": assistant_message,
            "session_id": session_id,
            "input_tokens": response_input_tokens,
            "output_tokens": response_output_tokens,
            "total_tokens": response_input_tokens + response_output_tokens,
            "latency_ms": round(latency_ms, 2),
            "context_used": len(context_metadata),
            "model": model_name
        }
    
    def get_session_stats(self, project_name: str, session_id: str) -> Dict[str, Any]:
        """Get statistics for a chat session."""
        session = self.db.get_chat_session(project_name, session_id)
        if not session:
            return {}
        
        history = self.db.get_chat_history(project_name, session_id)
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "total_input_tokens": session.get("total_input_tokens", 0),
            "total_output_tokens": session.get("total_output_tokens", 0),
            "total_tokens": session.get("total_input_tokens", 0) + session.get("total_output_tokens", 0),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at")
        }


def create_chat_service(
    db_backend,
    inference_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> LLMChatService:
    """
    Factory function to create a chat service with config defaults.
    
    Args:
        db_backend: Database backend instance
        inference_url: Override inference URL (uses config default if None)
        api_key: Override API key (uses config default if None)
    """
    from ..config import get_config_value
    
    url = inference_url or get_config_value("inference", "base_url", "http://localhost:8080/v1")
    key = api_key or get_config_value("inference", "api_key", None)
    model = get_config_value("inference", "default_model", "gpt-4")
    
    return LLMChatService(
        db_backend=db_backend,
        inference_base_url=url,
        api_key=key,
        default_model=model
    )
