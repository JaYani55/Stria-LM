#!/usr/bin/env python3
"""
Test script for the Auto-Generate-LM-Content endpoint
"""

import requests
import json
import time

# Configuration
STRIA_LM_URL = "http://127.0.0.1:8000"
TEST_PROJECT_NAME = "auto-generated-test"
TEST_URL = "https://example.com"  # You can change this to any website

def test_auto_generate_endpoint():
    """Test the complete auto-generation workflow"""
    
    print("🤖 Testing Auto-Generate-LM-Content Endpoint")
    print("=" * 50)
    
    # Step 1: Create a test project
    print("1. Creating test project...")
    create_project_payload = {
        "project_name": TEST_PROJECT_NAME,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    try:
        response = requests.post(f"{STRIA_LM_URL}/projects", json=create_project_payload)
        if response.status_code == 201:
            print(f"✅ Project '{TEST_PROJECT_NAME}' created successfully")
        elif response.status_code == 409:
            print(f"ℹ️  Project '{TEST_PROJECT_NAME}' already exists")
        else:
            print(f"❌ Failed to create project: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error creating project: {e}")
        return False
    
    # Step 2: Test the auto-generation endpoint
    print(f"\n2. Auto-generating content from {TEST_URL}...")
    auto_generate_payload = {
        "project_name": TEST_PROJECT_NAME,
        "url": TEST_URL,
        "max_pages": 5,
        "business_context": "Example business for testing"
    }
    
    try:
        start_time = time.time()
        response = requests.post(f"{STRIA_LM_URL}/auto-generate-lm-content", json=auto_generate_payload)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Auto-generation completed successfully!")
            print(f"⏱️  Time taken: {end_time - start_time:.2f} seconds")
            print(f"📄 Pages scraped: {result.get('pages_scraped', 0)}")
            print(f"❓ Prompts generated: {result.get('prompts_generated', 0)}")
            print(f"💬 Q&A pairs added: {result.get('qa_pairs_added', 0)}")
            print(f"🌐 Domains: {', '.join(result.get('scraped_domains', []))}")
        else:
            print(f"❌ Auto-generation failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during auto-generation: {e}")
        return False
    
    # Step 3: Test the chat functionality
    print(f"\n3. Testing chat with generated content...")
    chat_payload = {
        "prompt": "What services do you offer?",
        "top_k": 3
    }
    
    try:
        response = requests.post(f"{STRIA_LM_URL}/chat/{TEST_PROJECT_NAME}", json=chat_payload)
        if response.status_code == 200:
            chat_results = response.json()
            print(f"✅ Chat test successful! Found {len(chat_results)} responses")
            
            if chat_results:
                print("\n📋 Sample responses:")
                for i, result in enumerate(chat_results[:2], 1):
                    print(f"\n{i}. Original prompt: {result['original_prompt']}")
                    print(f"   Response: {result['response_text']}")
                    print(f"   Similarity: {result['similarity_score']:.3f}")
        else:
            print(f"❌ Chat test failed: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during chat test: {e}")
        return False
    
    print(f"\n🎉 All tests completed successfully!")
    return True

def test_individual_components():
    """Test individual components separately"""
    
    print("\n🔧 Testing Individual Components")
    print("=" * 50)
    
    # Test scraper
    print("1. Testing web scraper...")
    try:
        from src.Scraper.scraper_universal import scrape_website
        scraped_data = scrape_website("https://httpbin.org/html", max_pages=1)
        if scraped_data:
            print(f"✅ Scraper test successful! Scraped {len(scraped_data)} pages")
        else:
            print("❌ Scraper test failed - no data returned")
    except Exception as e:
        print(f"❌ Scraper test failed: {e}")
    
    # Test prompt generation
    print("\n2. Testing prompt generation...")
    try:
        from src.Agents.promptfile_agent import generate_prompts_from_content
        test_content = [{
            "url": "https://example.com",
            "title": "Test Business",
            "content": "We are a test business offering various services.",
            "domain": "example.com"
        }]
        prompts = generate_prompts_from_content(test_content, "Test business")
        if prompts and "prompts" in prompts:
            print(f"✅ Prompt generation test successful! Generated {len(prompts['prompts'])} prompts")
        else:
            print("❌ Prompt generation test failed")
    except Exception as e:
        print(f"❌ Prompt generation test failed: {e}")
    
    # Test answer generation
    print("\n3. Testing answer generation...")
    try:
        from src.Agents.answer_agent_universal import generate_answers_for_business
        test_prompts = ["What do you do?", "Where are you located?"]
        test_content = [{
            "url": "https://example.com",
            "title": "Test Business",
            "content": "We are a test business located in Test City.",
            "domain": "example.com"
        }]
        answers = generate_answers_for_business(test_prompts, test_content, "Test business")
        if answers and len(answers) == len(test_prompts):
            print(f"✅ Answer generation test successful! Generated {len(answers)} answers")
        else:
            print("❌ Answer generation test failed")
    except Exception as e:
        print(f"❌ Answer generation test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Stria-LM Auto-Generate Tests")
    print("=" * 60)
    
    # Check if the server is running
    try:
        response = requests.get(f"{STRIA_LM_URL}/projects")
        if response.status_code == 200:
            print("✅ Stria-LM server is running")
        else:
            print("❌ Stria-LM server is not responding correctly")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to Stria-LM server. Please make sure it's running at http://127.0.0.1:8000")
        print("   Start it with: uvicorn src.main:app --reload")
        exit(1)
    
    # Run tests
    test_individual_components()
    print()
    test_auto_generate_endpoint()
    
    print(f"\n✨ Testing complete! You can now use the endpoint:")
    print(f"   POST {STRIA_LM_URL}/auto-generate-lm-content")
    print(f"   Payload: {json.dumps({'project_name': 'your-project', 'url': 'https://your-website.com'}, indent=2)}")
