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

class PromptFileAgent:
    def __init__(self, openrouter_api_key: str = None):
        """Initialize the prompt file agent with OpenRouter connection"""
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        
        # Initialize OpenRouter through OpenAI-compatible API
        self.llm = ChatOpenAI(
            model="mistralai/mistral-7b-instruct",
            openai_api_key=self.api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=2000
        )
    
    def generate_prompt_file(self, scraped_content: List[Dict], business_context: str = None) -> Dict:
        """
        Generate a prompt file based on scraped website content
        
        Args:
            scraped_content: List of dictionaries containing scraped page data
            business_context: Optional context about the business
        
        Returns:
            Dictionary in the format of basic-chatbot-promptfiles.json
        """
        
        # Combine all content for analysis
        all_content = ""
        for page in scraped_content:
            all_content += f"\n\nPage: {page.get('title', 'Untitled')}\nURL: {page.get('url', '')}\n"
            all_content += page.get('content', '')[:2000]  # Limit content length
        
        # Truncate if too long
        if len(all_content) > 8000:
            all_content = all_content[:8000] + "... [Content truncated]"
        
        # Create the system prompt
        system_prompt = """
        You are an expert prompt generator for chatbots. Based on the provided website content, 
        generate 30 diverse questions that customers might ask about this business or organization.
        
        The questions should cover:
        - General information about the business
        - Products or services offered
        - Contact information and location
        - Business hours and policies
        - Frequently asked questions
        - Common customer concerns
        - Casual conversation starters
        - Specific details mentioned in the content
        
        Return ONLY a valid JSON object in this exact format:
        {
          "prompts": [
            "Question 1",
            "Question 2",
            ...
          ]
        }
        
        Make the questions natural and conversational, as if a real customer is asking them.
        Include both formal business questions and casual conversation starters.
        """
        
        # Create the human message with content and context
        human_message_content = f"""
        Website Content Analysis:
        {all_content}
        
        Business Context: {business_context or 'General business/organization'}
        
        Please generate 30 diverse questions that customers might ask about this business based on the content above.
        """
        
        try:
            # Generate the prompt file
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message_content)
            ]
            
            response = self.llm.invoke(messages)
            response_content = response.content.strip()
            
            # Try to extract JSON from the response
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = response_content[json_start:json_end]
                prompt_data = json.loads(json_content)
                
                # Validate the structure
                if 'prompts' in prompt_data and isinstance(prompt_data['prompts'], list):
                    return prompt_data
                else:
                    raise ValueError("Invalid prompt file structure")
            else:
                raise ValueError("Could not extract JSON from response")
                
        except Exception as e:
            logger.error(f"Error generating prompt file: {e}")
            # Return a fallback prompt file
            return self._get_fallback_prompts(business_context)
    
    def _get_fallback_prompts(self, business_context: str = None) -> Dict:
        """Return a fallback set of prompts if generation fails"""
        fallback_prompts = [
            "What services do you offer?",
            "What are your business hours?",
            "How can I contact you?",
            "Where are you located?",
            "What makes your business special?",
            "Do you offer any discounts or promotions?",
            "How long have you been in business?",
            "What is your return policy?",
            "Do you accept credit cards?",
            "Can I schedule an appointment?",
            "What are your prices?",
            "Do you offer delivery?",
            "What's your phone number?",
            "Do you have a website?",
            "What's your email address?",
            "Are you hiring?",
            "Do you offer warranties?",
            "What's your most popular product/service?",
            "Do you have any certifications?",
            "What's your experience level?",
            "Do you work weekends?",
            "How far do you travel?",
            "What payment methods do you accept?",
            "Do you offer free estimates?",
            "What's included in your service?",
            "How quickly can you respond?",
            "Do you have insurance?",
            "What's your cancellation policy?",
            "Can you provide references?",
            "Tell me about your business."
        ]
        
        return {"prompts": fallback_prompts}

def generate_prompts_from_content(scraped_content: List[Dict], business_context: str = None) -> Dict:
    """
    Main function to generate prompts from scraped content
    
    Args:
        scraped_content: List of scraped page data
        business_context: Optional business context
    
    Returns:
        Dictionary containing generated prompts
    """
    try:
        agent = PromptFileAgent()
        return agent.generate_prompt_file(scraped_content, business_context)
    except Exception as e:
        logger.error(f"Failed to initialize PromptFileAgent: {e}")
        # Return fallback prompts
        return PromptFileAgent(openrouter_api_key="dummy")._get_fallback_prompts(business_context)

if __name__ == "__main__":
    # Test the agent
    test_content = [
        {
            "url": "https://example.com",
            "title": "Example Business",
            "content": "We are a local bakery offering fresh bread, pastries, and custom cakes. Open Monday-Friday 6am-6pm.",
            "domain": "example.com"
        }
    ]
    
    result = generate_prompts_from_content(test_content, "Local bakery business")
    print(json.dumps(result, indent=2))
