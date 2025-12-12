"""
Token counting service with tiktoken and fallback estimation.
Provides accurate token counts for OpenAI models and approximate counts for others.
"""

from typing import Optional, List, Dict, Any
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Try to import tiktoken, set flag if unavailable
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Using fallback token estimation.")


# Model to encoding mapping
MODEL_ENCODINGS = {
    # GPT-4 models
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-turbo-preview": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    # GPT-3.5 models
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    # Embedding models
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    # Claude models (approximate with cl100k)
    "claude": "cl100k_base",
    "claude-2": "cl100k_base",
    "claude-3": "cl100k_base",
    # Llama models (approximate)
    "llama": "cl100k_base",
    "llama-2": "cl100k_base",
    "llama-3": "cl100k_base",
    # Mistral (approximate)
    "mistral": "cl100k_base",
    "mixtral": "cl100k_base",
}

# Average characters per token for fallback estimation
# Based on typical BPE tokenization (roughly 4 chars per token for English)
CHARS_PER_TOKEN_ESTIMATE = 4.0


@lru_cache(maxsize=10)
def get_encoding(model_name: str) -> Optional[Any]:
    """
    Get tiktoken encoding for a model name.
    Returns None if tiktoken is unavailable or model not found.
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        # Try direct model lookup first
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        pass
    
    # Try to find a matching encoding from our mapping
    model_lower = model_name.lower()
    for model_prefix, encoding_name in MODEL_ENCODINGS.items():
        if model_prefix in model_lower:
            try:
                return tiktoken.get_encoding(encoding_name)
            except Exception:
                pass
    
    # Default to cl100k_base for unknown models
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        logger.warning(f"Failed to get encoding: {e}")
        return None


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Count tokens in a text string.
    Uses tiktoken if available, otherwise falls back to character-based estimation.
    
    Args:
        text: The text to count tokens for
        model_name: The model name to use for tokenization
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    encoding = get_encoding(model_name)
    
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"tiktoken encoding failed: {e}, falling back to estimation")
    
    # Fallback: character-based estimation
    return estimate_tokens(text)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count based on character length.
    Uses average of ~4 characters per token for BPE tokenization.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Count characters, excluding excessive whitespace
    char_count = len(text)
    
    # Adjust for common patterns that affect tokenization:
    # - Punctuation typically gets its own token
    # - Numbers are often split
    # - Special characters and unicode
    
    # Base estimate
    base_estimate = char_count / CHARS_PER_TOKEN_ESTIMATE
    
    # Add adjustments for punctuation (rough estimate)
    punctuation_count = sum(1 for c in text if c in '.,!?;:()[]{}"\'-')
    adjustment = punctuation_count * 0.3  # Punctuation often adds extra tokens
    
    return int(base_estimate + adjustment)


def count_message_tokens(
    messages: List[Dict[str, str]], 
    model_name: str = "gpt-4"
) -> int:
    """
    Count tokens for a list of chat messages.
    Accounts for message formatting overhead.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model_name: The model name to use for tokenization
        
    Returns:
        Total token count including formatting overhead
    """
    if not messages:
        return 0
    
    total = 0
    
    # Token overhead per message (role tokens, formatting)
    # This varies by model but ~4 tokens per message is typical
    message_overhead = 4
    
    for message in messages:
        content = message.get("content", "")
        role = message.get("role", "")
        
        # Count content tokens
        total += count_tokens(content, model_name)
        
        # Add role tokens
        total += count_tokens(role, model_name)
        
        # Add message formatting overhead
        total += message_overhead
    
    # Add reply priming overhead (~3 tokens)
    total += 3
    
    return total


def count_prompt_tokens(
    system_prompts: List[Dict[str, str]],
    context_data: str,
    chat_history: List[Dict[str, str]],
    model_name: str = "gpt-4"
) -> Dict[str, int]:
    """
    Count tokens for the full prompt structure used in LLM chat.
    
    Args:
        system_prompts: List of system prompt dicts from actor prompt_messages
        context_data: Retrieved QA context as formatted string
        chat_history: List of chat history messages
        model_name: Model name for tokenization
        
    Returns:
        Dict with token counts for each section and total
    """
    # Count system prompts
    system_tokens = 0
    for prompt in system_prompts:
        content = prompt.get("content", "")
        system_tokens += count_tokens(content, model_name)
        system_tokens += 4  # Overhead per system message
    
    # Count context data
    context_tokens = count_tokens(context_data, model_name) if context_data else 0
    
    # Count chat history
    history_tokens = count_message_tokens(chat_history, model_name)
    
    total = system_tokens + context_tokens + history_tokens
    
    return {
        "system_tokens": system_tokens,
        "context_tokens": context_tokens,
        "history_tokens": history_tokens,
        "total_input_tokens": total
    }


def truncate_to_token_limit(
    text: str, 
    max_tokens: int, 
    model_name: str = "gpt-4",
    truncation_suffix: str = "..."
) -> str:
    """
    Truncate text to fit within a token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model_name: Model name for tokenization
        truncation_suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if not text:
        return text
    
    current_tokens = count_tokens(text, model_name)
    
    if current_tokens <= max_tokens:
        return text
    
    encoding = get_encoding(model_name)
    
    if encoding is not None:
        try:
            # Use tiktoken for precise truncation
            tokens = encoding.encode(text)
            suffix_tokens = encoding.encode(truncation_suffix)
            
            # Reserve space for suffix
            available_tokens = max_tokens - len(suffix_tokens)
            
            if available_tokens > 0:
                truncated_tokens = tokens[:available_tokens]
                return encoding.decode(truncated_tokens) + truncation_suffix
            else:
                return truncation_suffix
        except Exception:
            pass
    
    # Fallback: character-based truncation
    estimated_chars = int(max_tokens * CHARS_PER_TOKEN_ESTIMATE)
    return text[:estimated_chars - len(truncation_suffix)] + truncation_suffix


def get_tokenizer_info() -> Dict[str, Any]:
    """
    Get information about the tokenizer availability.
    
    Returns:
        Dict with tokenizer status and capabilities
    """
    return {
        "tiktoken_available": TIKTOKEN_AVAILABLE,
        "fallback_mode": not TIKTOKEN_AVAILABLE,
        "chars_per_token_estimate": CHARS_PER_TOKEN_ESTIMATE,
        "supported_model_families": list(set(MODEL_ENCODINGS.values())) if TIKTOKEN_AVAILABLE else []
    }


# Convenience class for managing token context
class TokenCounter:
    """
    Token counter instance for a specific model.
    Provides convenient methods for counting and tracking tokens.
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.encoding = get_encoding(model_name)
        self._total_input = 0
        self._total_output = 0
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        return count_tokens(text, self.model_name)
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages."""
        return count_message_tokens(messages, self.model_name)
    
    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to token limit."""
        return truncate_to_token_limit(text, max_tokens, self.model_name)
    
    def add_input(self, tokens: int):
        """Track input tokens."""
        self._total_input += tokens
    
    def add_output(self, tokens: int):
        """Track output tokens."""
        self._total_output += tokens
    
    @property
    def total_input(self) -> int:
        return self._total_input
    
    @property
    def total_output(self) -> int:
        return self._total_output
    
    @property
    def total_tokens(self) -> int:
        return self._total_input + self._total_output
    
    def reset(self):
        """Reset token counters."""
        self._total_input = 0
        self._total_output = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return {
            "model": self.model_name,
            "tiktoken_available": self.encoding is not None,
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_tokens": self.total_tokens
        }
