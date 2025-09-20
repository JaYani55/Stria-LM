import os
import json
import requests
from dotenv import load_dotenv
from tqdm import tqdm
import time

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
STRIA_LM_URL = "http://127.0.0.1:8000"
PROJECT_NAME = "weighted-bar-buddy"
PROMPT_FILE = "scripts/weighted-chatbot-promptfiles.json"

# Enable weights for this project
os.environ["ENABLE_WEIGHTS"] = "true"

# Dictionary to define the model
MODEL_TO_USE = {
    "name": "mistralai/mistral-nemo:free",
    "context_length": 8000
}

# --- Helper Functions ---

def create_stria_project():
    """Creates a new project in Stria-LM."""
    url = f"{STRIA_LM_URL}/projects"
    payload = {"project_name": PROJECT_NAME}
    print(f"Attempting to create project '{PROJECT_NAME}'...")
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 201:
            print(f"Project '{PROJECT_NAME}' created successfully.")
            return True
        elif response.status_code == 409:
            print(f"Project '{PROJECT_NAME}' already exists. Skipping creation.")
            return True
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error creating project: {e}")
        return False
    return False


def get_response_from_openrouter(prompt: str, weight_category: str) -> str:
    """Gets a generated response from the OpenRouter API with weight-specific instructions."""
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")

    # Customize system prompt based on weight category
    if weight_category == "high_weight":
        system_prompt = """You are 'Sam', a witty and experienced bartender who's been working at Murphy's Pub for 15 years. 
        You're originally from Chicago, love craft beer, and have a dry sense of humor. You're knowledgeable about drinks, 
        good at reading people, and always have interesting stories from your years behind the bar. This is core information 
        about who you are - be consistent and detailed."""
    elif weight_category == "medium_weight":
        system_prompt = """You are Sam, the bartender from Murphy's Pub. Share interesting stories and experiences 
        from your time working at the bar. Make them engaging and memorable, with good details that show your 
        personality and background."""
    else:  # low_weight
        system_prompt = """You are Sam, a bartender. Give casual, quick responses to these conversational questions. 
        Keep it light and don't worry too much about deep details."""

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL_TO_USE["name"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.7,
                "min_tokens": 10
            }
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenRouter API: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing OpenRouter response: {e}")
        return None


def add_to_stria_database_with_weight(prompt: str, response: str, weight: float):
    """Adds a prompt-response pair to the Stria-LM project with specified weight."""
    url = f"{STRIA_LM_URL}/projects/{PROJECT_NAME}/add"
    payload = {
        "prompt": prompt, 
        "response": response,
        "weight": weight
    }
    try:
        response_obj = requests.post(url, json=payload)
        response_obj.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nError adding to Stria DB for prompt '{prompt}': {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
        return False


def load_weighted_prompts(file_path: str):
    """Load prompts with their weight categories from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        weighted_prompts = []
        for category, category_data in data.items():
            weight = category_data.get("weight", 1.0)
            description = category_data.get("description", "")
            prompts = category_data.get("prompts", [])
            
            for prompt in prompts:
                weighted_prompts.append({
                    "prompt": prompt,
                    "weight": weight,
                    "category": category,
                    "description": description
                })
        
        return weighted_prompts
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at '{file_path}'")
        return []
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{file_path}'")
        return []


def display_weight_statistics(weighted_prompts):
    """Display statistics about the weighted prompts."""
    categories = {}
    for item in weighted_prompts:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = {"count": 0, "weight": item["weight"], "description": item["description"]}
        categories[cat]["count"] += 1
    
    print("\n--- Weight Category Statistics ---")
    for category, stats in categories.items():
        print(f"{category}: {stats['count']} prompts, weight {stats['weight']}")
        print(f"  Description: {stats['description']}")
    print()


def test_weighted_search(project_name: str, test_queries: list):
    """Test the weighted search functionality."""
    print("\n--- Testing Weighted Search ---")
    url = f"{STRIA_LM_URL}/chat/{project_name}"
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = requests.post(url, json={"prompt": query, "top_k": 3})
            response.raise_for_status()
            data = response.json()
            
            print(f"Response: {data[0].get('response_text', 'No response') if data else 'No results'}")
            
            # Display similarity info from the response list
            if data:
                print("Top matches found:")
                for i, result in enumerate(data[:3], 1):
                    weight = result.get('weight', 'N/A')
                    weighted_sim = result.get('weighted_similarity')
                    sim_score = result.get('similarity_score', 0)
                    
                    print(f"  {i}. Weight: {weight}, Similarity: {sim_score:.3f}")
                    if weighted_sim is not None:
                        print(f"     Weighted Similarity: {weighted_sim:.3f}")
                    print(f"     Original: '{result.get('original_prompt', '')[:50]}...'")
            
        except requests.exceptions.RequestException as e:
            print(f"Error testing query: {e}")
        
        time.sleep(1)  # Be nice to the API


# --- Main Script ---

if __name__ == "__main__":
    print("--- Starting Weighted Chatbot Data Generator ---")
    print("This script demonstrates the weighting feature by creating responses with different priorities.")

    # 1. Check for API Key
    if not API_KEY:
        print("\nERROR: OPENROUTER_API_KEY is not set in your .env file.")
        print("Please add your key and try again.")
        exit()

    # 2. Create the Stria-LM project
    if not create_stria_project():
        print("Halting script because project could not be created or found.")
        exit()

    # 3. Load weighted prompts from the JSON file
    weighted_prompts = load_weighted_prompts(PROMPT_FILE)
    
    if not weighted_prompts:
        print("No prompts found in the JSON file. Nothing to do.")
        exit()

    # 4. Display statistics
    display_weight_statistics(weighted_prompts)

    # 5. Generate responses and populate the database with weights
    print(f"Found {len(weighted_prompts)} prompts across different weight categories.")
    print(f"Generating responses using '{MODEL_TO_USE['name']}'...")
    
    success_count = 0
    total_count = len(weighted_prompts)
    
    # Process prompts by weight category (high to low priority)
    categories_order = ["high_weight_prompts", "medium_weight_prompts", "low_weight_prompts"]
    
    for category in categories_order:
        category_prompts = [p for p in weighted_prompts if p["category"] == category]
        if not category_prompts:
            continue
            
        weight = category_prompts[0]["weight"]
        description = category_prompts[0]["description"]
        
        print(f"\n--- Processing {category} (Weight: {weight}) ---")
        print(f"Description: {description}")
        
        for prompt_data in tqdm(category_prompts, desc=f"Processing {category}"):
            prompt = prompt_data["prompt"]
            weight = prompt_data["weight"]
            category_name = prompt_data["category"]
            
            # Generate response with category-specific instructions
            response_text = get_response_from_openrouter(prompt, category_name.replace("_prompts", ""))
            
            if response_text:
                if add_to_stria_database_with_weight(prompt, response_text, weight):
                    success_count += 1
                else:
                    tqdm.write(f"Failed to add to database: '{prompt}'")
            else:
                tqdm.write(f"Skipping prompt due to generation error: '{prompt}'")
            
            # Be nice to the API
            time.sleep(1)

    print(f"\n--- Generation Complete! ---")
    print(f"Successfully processed {success_count}/{total_count} prompts")
    print(f"The '{PROJECT_NAME}' project now contains weighted responses.")
    
    # 6. Test the weighted search functionality
    test_queries = [
        "What's your name?",  # Should match high-weight responses
        "Tell me a story.",   # Should match medium-weight responses  
        "Tell me a joke."     # Should match low-weight responses
    ]
    
    test_weighted_search(PROJECT_NAME, test_queries)
    
    print(f"\n--- Script Complete! ---")
    print(f"You can now chat with the weighted bot via: {STRIA_LM_URL}/chat/{PROJECT_NAME}")
    print("Try asking questions from different weight categories to see how the weighting affects results!")