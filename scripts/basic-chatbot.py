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
PROJECT_NAME = "bar-buddy-bot"
PROMPT_FILE = "scripts/basic-chatbot-promptfiles.json"

# Dictionary to define the model. You can easily swap this out.
# Find more models at https://openrouter.ai/docs#models
MODEL_TO_USE = {
    "name": "mistralai/mistral-7b-instruct",
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


def get_response_from_openrouter(prompt: str) -> str:
    """Gets a generated response from the OpenRouter API."""
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in .env file.")

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
            },
            json={
                "model": MODEL_TO_USE["name"],
                "messages": [
                    {"role": "system", "content": "You are a friendly, slightly sarcastic, and humorous chatbot you'd meet at a bar. Keep your answers conversational and not too long, like you're just chatting. You have some interesting stories and a dry wit."},
                    {"role": "user", "content": prompt}
                ]
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


def add_to_stria_database(prompt: str, response: str):
    """Adds a prompt-response pair to the Stria-LM project."""
    url = f"{STRIA_LM_URL}/projects/{PROJECT_NAME}/add"
    payload = {"prompt": prompt, "response": response}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\nError adding to Stria DB for prompt '{prompt}': {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")


# --- Main Script ---

if __name__ == "__main__":
    print("--- Starting Basic Chatbot Data Generator ---")

    # 1. Check for API Key
    if not API_KEY:
        print("\nERROR: OPENROUTER_API_KEY is not set in your .env file.")
        print("Please add your key and try again.")
        exit()

    # 2. Create the Stria-LM project
    if not create_stria_project():
        print("Halting script because project could not be created or found.")
        exit()

    # 3. Load prompts from the JSON file
    try:
        with open(PROMPT_FILE, 'r') as f:
            prompts = json.load(f).get("prompts", [])
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at '{PROMPT_FILE}'")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from '{PROMPT_FILE}'")
        exit()

    if not prompts:
        print("No prompts found in the JSON file. Nothing to do.")
        exit()

    # 4. Generate responses and populate the database
    print(f"\nFound {len(prompts)} prompts. Generating responses using '{MODEL_TO_USE['name']}'...")
    
    for prompt in tqdm(prompts, desc="Generating and Storing Prompts"):
        response_text = get_response_from_openrouter(prompt)
        
        if response_text:
            add_to_stria_database(prompt, response_text)
        else:
            tqdm.write(f"Skipping prompt due to generation error: '{prompt}'")
        
        # Be nice to the API
        time.sleep(1)

    print("\n--- Chatbot data generation complete! ---")
    print(f"The '{PROJECT_NAME}' project in your Stria-LM instance is now populated.")
    print("You can now chat with it via the /chat/{project_name} endpoint.")
