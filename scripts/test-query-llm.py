import requests
import json

# --- Configuration ---
HOST = "127.0.0.1"
PORT = 8008
API_URL = f"http://{HOST}:{PORT}/v1/chat/completions"

# --- Headers ---
HEADERS = {
    "Content-Type": "application/json"
}

def test_standard_completion():
    """Tests a standard, non-streaming chat completion request."""
    print("--- Testing Standard Completion ---")
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        print("Full response:")
        print(json.dumps(data, indent=2))
        
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "No content found.")
        print("\nAssistant's Message:")
        print(content.strip())
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    print("-" * 30 + "\n")


def test_streaming_completion():
    """Tests a streaming chat completion request."""
    print("--- Testing Streaming Completion ---")
    
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "user", "content": "Write a short story about a robot who discovers music."}
        ],
        "max_tokens": 200,
        "temperature": 0.8,
        "stream": True
    }
    
    try:
        with requests.post(API_URL, headers=HEADERS, json=payload, stream=True) as response:
            response.raise_for_status()
            print("Assistant's Streamed Message:")
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    if decoded_chunk.startswith("data:"):
                        data_str = decoded_chunk[len("data: "):]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            print(content, end="", flush=True)
                        except json.JSONDecodeError:
                            print(f"\nError decoding JSON from chunk: {data_str}")
        print("\n" + "-" * 30 + "\n")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    print(f"Querying LLM at {API_URL}")
    test_standard_completion()
    test_streaming_completion()