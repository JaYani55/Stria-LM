Vision: An extensible vector database framework to store actor data for LLMs, QA pairs for fine-tuning or scrape, transform and store context data for RAG.

# Stria-LM: Portable, File-Based Retrieval Models

Stria-LM is a lightweight framework for building, managing, and querying retrieval-based chatbot models. The core philosophy is **simplicity and portability**. Each "trained" model can be a self-contained SQLite database file that can be easily copied, shared, and deployed—or stored in a centralized PostgreSQL server for team collaboration.

Instead of traditional model training (i.e., adjusting weights), "training" in Stria-LM is the process of populating a knowledge base with prompt-response pairs. The system's intelligence comes from using sentence embeddings to find the most semantically similar prompt in its database to a user's query and returning the corresponding response.

## Core Concepts

*   **The Model is the Database:** Each chatbot project contains all the data (prompt/response pairs) and the vector index needed for fast semantic search.
*   **Flexible Database Backends:** Choose between SQLite (local, portable files) or PostgreSQL (centralized, scalable server).
*   **Intelligence via Embeddings:** User queries are converted into numerical vectors (embeddings). The system finds the closest matching vector in the database to retrieve the most relevant pre-generated response.
*   **Portability:** Export any project to a portable SQLite file for backup, sharing, or offline deployment.

## Database Options

Stria-LM supports two database backends through a unified abstraction layer:

### SQLite + sqlite-vec (Default)
- **Best for:** Local development, offline use, portable deployments
- **Storage:** Each project is a separate `.db` file in the `projects/` directory
- **Vector search:** Uses [sqlite-vec](https://github.com/asg017/sqlite-vec) extension

### PostgreSQL + pgvector
- **Best for:** Team collaboration, cloud deployment, large-scale applications
- **Storage:** All projects in a single centralized database
- **Vector search:** Uses [pgvector](https://github.com/pgvector/pgvector) extension

### Switching Databases

Set the `DATABASE_TYPE` in your `.env` file:

```bash
# For local SQLite databases (default)
DATABASE_TYPE=sqlite

# For PostgreSQL server
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/strialm
```

## Project Structure

```
stria-lm/
├── projects/                # SQLite databases (one per project)
│   └── my-new-bot/
│       └── my-new-bot.db
├── exports/                 # Exported portable databases
├── src/
│   ├── main.py              # FastAPI application logic
│   ├── database/            # Database abstraction layer
│   │   ├── base.py          # Abstract backend interface
│   │   ├── sqlite.py        # SQLite + sqlite-vec implementation
│   │   └── postgresql.py    # PostgreSQL + pgvector implementation
│   ├── embedding.py         # Embedding model loader and functions
│   ├── models.py            # Pydantic models for API request/response
│   ├── models_db.py         # SQLAlchemy models for PostgreSQL
│   └── config.py            # Application settings
├── scripts/
│   ├── bulk_importer.py     # Populate a model from CSV
│   └── export_to_sqlite.py  # Export projects to portable SQLite files
├── docs/
│   └── data-structures.md   # Detailed data structure documentation
├── tests/
│   └── test_api.py          # Tests for the API endpoints
├── .env                     # Environment variables (secrets)
├── .env.example             # Template for .env file
├── config.toml              # Application configuration
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

## Import/Export Projects

Stria-LM allows you to export projects to portable SQLite database files and import them into any backend. This enables:

- **Backup:** Create portable backups of your projects
- **Sharing:** Share projects as single `.db` files
- **Migration:** Move projects between SQLite and PostgreSQL
- **Offline Deployment:** Deploy exported databases to offline environments

### Export via API

```bash
curl -X POST "http://127.0.0.1:8000/projects/my-faq-bot/export" \
  -H "Content-Type: application/json" \
  -d '{"output_path": "exports/my-faq-bot.db"}'
```

### Export via Command Line

```bash
# Export a single project
python scripts/export_to_sqlite.py my-faq-bot

# Export with custom path
python scripts/export_to_sqlite.py my-faq-bot ~/backups/my-faq-bot.db

# Export all projects
python scripts/export_to_sqlite.py --all

# List available projects
python scripts/export_to_sqlite.py --list
```

### Import via API

```bash
curl -X POST "http://127.0.0.1:8000/projects/imported-bot/import" \
  -H "Content-Type: application/json" \
  -d '{"input_path": "exports/my-faq-bot.db"}'
```

## GUI Utilities

### Project Manager (Tkinter)
`run_gui_project_manager.py` - Setup and configuration interface

- **Projects Tab:** Create, view, and delete projects
- **Environment Variables Tab:** Edit API keys and database settings
- **Database Tab:** Switch database types, import/export projects, PostgreSQL setup guide
- **About Tab:** Application info and quick actions

```bash
python run_gui_project_manager.py
```

### Data Manager (Streamlit)
`streamlit_project_manager.py` - Interactive data browsing and editing

- **Q&A Pairs Tab:** Browse and edit prompt/response data
- **Semantic Search Tab:** Test vector similarity search
- **Operations Tab:** Re-embed prompts, export projects, delete data
- **Add Data Tab:** Add individual Q&A pairs or bulk import from JSON

```bash
streamlit run streamlit_project_manager.py
```

### Inference Server (Tkinter)
`run_gui_inferenceServer.py` - Local GGUF model inference server with start/stop controls

## Configuration

### Environment Variables (.env)

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_TYPE` | Database backend (`sqlite` or `postgresql`) | `sqlite` |
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM agents | - |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | - |
| `ENABLE_WEIGHTS` | Enable weighted similarity scoring | `true` |

### config.toml

Non-sensitive configuration:

```toml
[database]
type = "sqlite"
sqlite_path = "projects"

[embedding]
default_model = "local_default"

[embedding.models.local_default]
category = "Local"
model = "sentence-transformers/all-MiniLM-L6-v2"

[experimental]
enable_weights = true
```

## PostgreSQL Setup

To use PostgreSQL instead of SQLite:

1. **Install PostgreSQL** from https://www.postgresql.org/download/

2. **Create database and enable pgvector:**
   ```sql
   CREATE DATABASE strialm;
   \c strialm
   CREATE EXTENSION vector;
   ```

3. **Configure connection in `.env`:**
   ```bash
   DATABASE_TYPE=postgresql
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/strialm
   ```

4. **Restart the application**

## Documentation

- [Data Structures Reference](docs/data-structures.md) - Detailed schema and extension guide

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

Copyright © 2025 Pluracon, Jay Rathjen

---

<p align="center">
  Made with ❤️ by <strong>Pluracon, Jay Rathjen</strong>
</p>
