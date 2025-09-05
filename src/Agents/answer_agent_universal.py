import os
import json
import requests
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AnswerAgent:
    def __init__(self, openrouter_api_key: str = None):
        """Initialize the answer agent with OpenRouter connection"""
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        # Initialize OpenRouter through OpenAI-compatible API
        self.llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct",
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=1000
        )
    
    def generate_answer(self, question: str, business_content: List[Dict], business_context: str = None) -> str:
        """
        Generate an answer for a specific question based on business content
        
        Args:
            question: The question to answer
            business_content: List of scraped content from the business website
            business_context: Optional context about the business
        
        Returns:
            Generated answer string
        """
        
        # Combine relevant content
        content_summary = ""
        for page in business_content:
            content_summary += f"\n\nPage: {page.get('title', 'Untitled')}\n"
            content_summary += page.get('content', '')[:1500]  # Limit content per page
        
        # Truncate if too long
        if len(content_summary) > 6000:
            content_summary = content_summary[:6000] + "... [Content truncated]"
        
        # Create system prompt for answering questions
        system_prompt = f"""
        You are a helpful customer service representative for this business. 
        Answer questions based on the provided business information in a friendly, professional, and conversational tone.
        
        Guidelines:
        - Be helpful and informative
        - Keep answers concise but complete (1-3 sentences typically)
        - Use the provided business content to answer accurately
        - If you don't have specific information, provide a helpful general response
        - Maintain a professional yet friendly tone
        - For casual questions, respond in a conversational way
        - Always stay in character as a representative of this business
        
        Business Context: {business_context or 'General business'}
        
        Business Information:
        {content_summary}
        """
        
        human_message = f"Question: {question}\n\nPlease provide a helpful answer based on the business information provided."
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer for question '{question}': {e}")
            return self._get_fallback_answer(question)
    
    def generate_answers_for_prompts(self, prompts: List[str], business_content: List[Dict], 
                                   business_context: str = None) -> List[Dict]:
        """
        Generate answers for a list of prompts
        
        Args:
            prompts: List of questions/prompts
            business_content: Scraped business content
            business_context: Optional business context
        
        Returns:
            List of dictionaries with prompt and response pairs
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Generating answer {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            try:
                answer = self.generate_answer(prompt, business_content, business_context)
                results.append({
                    "prompt": prompt,
                    "response": answer
                })
            except Exception as e:
                logger.error(f"Failed to generate answer for prompt '{prompt}': {e}")
                results.append({
                    "prompt": prompt,
                    "response": self._get_fallback_answer(prompt)
                })
        
        return results
    
    def _get_fallback_answer(self, question: str) -> str:
        """Provide a fallback answer when generation fails"""
        fallback_responses = {
            "contact": "Please feel free to contact us for more information about our services.",
            "hours": "We'd be happy to discuss our business hours with you. Please give us a call or visit our website.",
            "location": "You can find our location and contact details on our website or by calling us directly.",
            "services": "We offer a variety of services to meet your needs. Please contact us to learn more.",
            "price": "For pricing information, please contact us directly and we'll be happy to provide a quote.",
            "default": "Thank you for your question! Please contact us directly and we'll be happy to help you with that."
        }
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["contact", "phone", "email", "reach"]):
            return fallback_responses["contact"]
        elif any(word in question_lower for word in ["hours", "open", "close", "time"]):
            return fallback_responses["hours"]
        elif any(word in question_lower for word in ["location", "where", "address", "find"]):
            return fallback_responses["location"]
        elif any(word in question_lower for word in ["service", "offer", "do", "provide"]):
            return fallback_responses["services"]
        elif any(word in question_lower for word in ["price", "cost", "fee", "charge", "money"]):
            return fallback_responses["price"]
        else:
            return fallback_responses["default"]

def generate_answers_for_business(prompts: List[str], business_content: List[Dict], 
                                business_context: str = None) -> List[Dict]:
    """
    Main function to generate answers for business prompts
    
    Args:
        prompts: List of questions/prompts
        business_content: Scraped business content
        business_context: Optional business context
    
    Returns:
        List of prompt-response pairs
    """
    try:
        agent = AnswerAgent()
        return agent.generate_answers_for_prompts(prompts, business_content, business_context)
    except Exception as e:
        logger.error(f"Failed to initialize AnswerAgent: {e}")
        # Return fallback responses
        fallback_agent = AnswerAgent(openrouter_api_key="dummy")
        return [
            {
                "prompt": prompt,
                "response": fallback_agent._get_fallback_answer(prompt)
            }
            for prompt in prompts
        ]

if __name__ == "__main__":
    # Test the agent
    test_prompts = [
        "What are your business hours?",
        "Where are you located?",
        "What services do you offer?"
    ]
    
    test_content = [
        {
            "url": "https://example.com",
            "title": "Example Business",
            "content": "We are a local bakery offering fresh bread, pastries, and custom cakes. Open Monday-Friday 6am-6pm, located at 123 Main Street.",
            "domain": "example.com"
        }
    ]
    
    results = generate_answers_for_business(test_prompts, test_content, "Local bakery business")
    for result in results:
        print(f"Q: {result['prompt']}")
        print(f"A: {result['response']}")
        print("---")
