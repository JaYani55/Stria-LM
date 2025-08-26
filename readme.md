# Stria-LM: Portable, File-Based Retrieval Models

Stria-LM is a lightweight framework for building, managing, and querying retrieval-based chatbot models. The core philosophy is **simplicity and portability**. Each "trained" model is a self-contained SQLite database file that can be easily copied, shared, and deployed.

Instead of traditional model training (i.e., adjusting weights), "training" in Stria-LM is the process of populating a knowledge base with prompt-response pairs. The system's intelligence comes from using sentence embeddings to find the most semantically similar prompt in its database to a user's query and returning the corresponding response.

## Core Concepts

*   **The Model is the Database:** Each chatbot is a single `sqlite` file. This file contains all the data (prompt/response pairs) and the vector index needed for fast semantic search.
*   **Intelligence via Embeddings:** User queries are converted into numerical vectors (embeddings). The system finds the closest matching vector in the database to retrieve the most relevant pre-generated response.
*   **Portability:** Because a trained bot is just a single file, it can be moved, backed up, or deployed with minimal effort.

## Project Structure

```
stria-lm/
├── projects/                # Each sub-directory holds a trained model
│   └── my-new-bot/
│       └── my-new-bot.db
├── src/
│   ├── main.py              # FastAPI application logic
│   ├── database.py          # Functions for DB creation and queries
│   ├── embedding.py         # Embedding model loader and functions
│   ├── models.py            # Pydantic models for API request/response
│   └── config.py            # Application settings
├── scripts/
│   └── bulk_importer.py     # Example script to populate a model from a CSV
├── tests/
│   └── test_api.py          # Tests for the API endpoints
├── .env                     # Environment variables
└── requirements.txt         # Python dependencies
```

## Setup and Installation

1.  **Create a Virtual Environment:** It is highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv venv
    ```
    On Windows:
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
    On macOS/Linux:
    ```bash
    source venv/bin/activate
    ```

2.  **Install Dependencies:** Install all required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server:** Start the FastAPI application using Uvicorn.

    ```bash
    uvicorn src.main:app --reload
    ```
    The server will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Usage: API Endpoints

You can interact with the Stria-LM server using any HTTP client, such as `curl` or Postman.

#### 1. Create a New Project

This creates a new directory and an empty SQLite database for your new bot.

*   **Endpoint:** `POST /projects`
*   **Body:**
    ```json
    {
      "project_name": "my-faq-bot",
      "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    ```
*   **Example `curl` command:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/projects" \
    -H "Content-Type: application/json" \
    -d '{"project_name": "my-faq-bot"}'
    ```

#### 2. Add Data to a Project ("Training")

This adds a new prompt-response pair to your project's knowledge base. The prompt is automatically converted into a vector embedding.

*   **Endpoint:** `POST /projects/{project_name}/add`
*   **Body:**
    ```json
    {
      "prompt": "What are your business hours?",
      "response": "We are open from 9 AM to 5 PM, Monday to Friday."
    }
    ```
*   **Example `curl` command:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/projects/my-faq-bot/add" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What are your business hours?", "response": "We are open from 9 AM to 5 PM, Monday to Friday."}'
    ```

#### 3. Chat with a Project (Inference)

This takes a user's query, finds the most similar prompts in the database, and returns their corresponding responses.

*   **Endpoint:** `POST /chat/{project_name}`
*   **Body:**
    ```json
    {
      "prompt": "When are you open?",
      "top_k": 2
    }
    ```
*   **Example `curl` command:**
    ```bash
    curl -X POST "http://127.0.0.1:8000/chat/my-faq-bot" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "When are you open?", "top_k": 2}'
    ```
*   **Example Response:**
    ```json
    [
      {
        "response_text": "We are open from 9 AM to 5 PM, Monday to Friday.",
        "original_prompt": "What are your business hours?",
        "similarity_score": 0.9812
      }
    ]
    ```

## Bulk Importing Data

To populate a project from a large dataset, you can use the `bulk_importer.py` script. The script reads from a CSV file which must contain `prompt` and `response` columns.

1.  Create a CSV file named `my_data.csv`:
    ```csv
    prompt,response
    "What is your return policy?","You can return any item within 30 days of purchase."
    "Do you ship internationally?","Yes, we ship to most countries worldwide."
    "How can I track my order?","You will receive a tracking link via email once your order has shipped."
    ```

2.  Run the importer script from your terminal:
    ```bash
    python scripts/bulk_importer.py my-faq-bot my_data.csv
    ```

## Running Tests

The project includes a suite of tests to verify the API's functionality. To run them, use `pytest`:

```bash
pytest
```
This will automatically discover and run the tests in the `tests/` directory.
