from fastapi.testclient import TestClient
import pytest
import os
import shutil

# Make sure we use the test client for the main app
from src.main import app
from src.config import PROJECTS_DIR

client = TestClient(app)

# Define a test project name
TEST_PROJECT_NAME = "test-bot"

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: ensure the projects directory is clean before tests
    if PROJECTS_DIR.exists():
        shutil.rmtree(PROJECTS_DIR)
    PROJECTS_DIR.mkdir()
    
    yield
    
    # Teardown: clean up the created project files after all tests are done
    if PROJECTS_DIR.exists():
        shutil.rmtree(PROJECTS_DIR)

def test_list_projects_empty():
    response = client.get("/projects")
    assert response.status_code == 200
    assert response.json() == []

def test_create_project():
    response = client.post(
        "/projects",
        json={"project_name": TEST_PROJECT_NAME, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    assert response.status_code == 201
    assert response.json() == {"message": f"Project '{TEST_PROJECT_NAME}' created successfully."}
    
    # Verify the directory and db file were created
    project_dir = PROJECTS_DIR / TEST_PROJECT_NAME
    db_file = project_dir / f"{TEST_PROJECT_NAME}.db"
    assert project_dir.is_dir()
    assert db_file.is_file()

def test_create_existing_project():
    response = client.post(
        "/projects",
        json={"project_name": TEST_PROJECT_NAME, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}
    )
    assert response.status_code == 409 # Conflict

def test_list_projects_after_creation():
    response = client.get("/projects")
    assert response.status_code == 200
    assert response.json() == [TEST_PROJECT_NAME]

def test_add_data_to_project():
    response = client.post(
        f"/projects/{TEST_PROJECT_NAME}/add",
        json={"prompt": "What is your name?", "response": "I am the test bot."}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Data added successfully."}

def test_add_data_to_nonexistent_project():
    response = client.post(
        "/projects/nonexistent-project/add",
        json={"prompt": "test", "response": "test"}
    )
    assert response.status_code == 404

def test_chat_with_project():
    # Add another piece of data for a better test
    client.post(
        f"/projects/{TEST_PROJECT_NAME}/add",
        json={"prompt": "What can you do?", "response": "I can answer questions."}
    )

    # Now, ask a similar question
    response = client.post(
        f"/chat/{TEST_PROJECT_NAME}",
        json={"prompt": "What are you called?", "top_k": 1}
    )
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert results[0]["response_text"] == "I am the test bot."
    assert results[0]["original_prompt"] == "What is your name?"
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] > 0.8 # Expect a high similarity

def test_chat_with_nonexistent_project():
    response = client.post(
        "/chat/nonexistent-project",
        json={"prompt": "test", "top_k": 1}
    )
    assert response.status_code == 404
