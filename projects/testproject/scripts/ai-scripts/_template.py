"""
AI Script Template
==================
A template for creating AI-powered scripts.

Usage:
    Set your API key as an environment variable.
    Customize the prompt and model settings below.
"""
import os
import json
from typing import Optional

# Uncomment the API client you want to use:
# import openai
# from openai import OpenAI


def call_llm(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Optional[str]:
    """
    Call an LLM API with the given prompt.
    
    Supports OpenAI and OpenRouter APIs.
    Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable.
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY")
    
    # Determine base URL based on which key is set
    if os.getenv("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"
    else:
        base_url = "https://api.openai.com/v1"
    
    # Uncomment to use the OpenAI client:
    # client = OpenAI(api_key=api_key, base_url=base_url)
    # 
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=temperature,
    #     max_tokens=max_tokens
    # )
    # 
    # return response.choices[0].message.content
    
    print(f"Would call {model} with prompt: {prompt[:100]}...")
    return "AI response placeholder"


def process_batch(prompts: list[str]) -> list[str]:
    """Process a batch of prompts through the LLM."""
    results = []
    for prompt in prompts:
        result = call_llm(prompt)
        if result:
            results.append(result)
    return results


if __name__ == "__main__":
    # Example usage
    response = call_llm("What is the capital of France?")
    print(f"Response: {response}")
