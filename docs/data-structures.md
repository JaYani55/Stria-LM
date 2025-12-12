# Stria-LM Data Structures

This document provides a comprehensive reference of all data structures used in Stria-LM, intended to guide future feature extensions and integrations.

## Table of Contents

1. [Overview](#overview)
2. [Database Architecture](#database-architecture)
3. [Core Data Models](#core-data-models)
4. [SQLite Schema](#sqlite-schema)
5. [PostgreSQL Schema](#postgresql-schema)
6. [API Models (Pydantic)](#api-models-pydantic)
7. [Configuration Structures](#configuration-structures)
8. [Vector Embeddings](#vector-embeddings)
9. [Extension Points](#extension-points)

---

## Overview

Stria-LM uses a dual-database architecture that supports both:
- **SQLite + sqlite-vec**: Local, file-based databases (one per project)
- **PostgreSQL + pgvector**: Centralized remote database with all projects

Both backends implement the same abstract interface (`DatabaseBackend`), ensuring consistent behavior regardless of the underlying storage.

---

## Database Architecture

### Abstraction Layer

```
┌─────────────────────────────────────┐
│           Application Code          │
│    (FastAPI, Streamlit, Tkinter)    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         DatabaseBackend (ABC)       │
│      src/database/base.py           │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│ SQLiteBackend │   │ PostgreSQL    │
│               │   │ Backend       │
│ sqlite-vec    │   │ pgvector      │
└───────────────┘   └───────────────┘
```

### Backend Selection

The active backend is determined by the `DATABASE_TYPE` environment variable:
- `sqlite` (default): Uses `SQLiteBackend`
- `postgresql`: Uses `PostgreSQLBackend`

---

## Core Data Models

### Project

Represents a collection of QA pairs with a specific embedding model.

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key (auto-increment) |
| `name` | String | Unique project identifier |
| `embedding_model` | String | Model ID or path (e.g., `local_default`) |
| `vector_dimension` | Integer | Dimension of embedding vectors (e.g., 384, 768) |
| `created_at` | DateTime | Project creation timestamp |

**Relationships:**
- One-to-many with `QAPair`
- One-to-many with `ScrapedContent`
- One-to-many with `PromptFile`

### QAPair

A question-answer pair with its vector embedding.

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key (auto-increment) |
| `project_id` | Integer | Foreign key to Project |
| `prompt_text` | Text | The question/prompt text |
| `response_text` | Text | The answer/response text |
| `weight` | Float | Similarity weight multiplier (default: 1.0) |
| `embedding` | Vector | Prompt embedding (float32 array) |

**Notes:**
- The `weight` field allows prioritization of certain responses
- Weighted similarity = `similarity_score * weight`
- Vector dimension must match the project's `vector_dimension`

### ScrapedContent

Stores web-scraped content for context and RAG applications.

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `project_id` | Integer | Foreign key to Project |
| `url` | String | Source URL |
| `title` | String (nullable) | Page title |
| `content` | Text | Scraped text content |
| `domain` | String | Source domain |
| `created_at` | DateTime | Scrape timestamp |

### PromptFile

Stores generated prompt files from the auto-generation pipeline.

| Field | Type | Description |
|-------|------|-------------|
| `id` | Integer | Primary key |
| `project_id` | Integer | Foreign key to Project |
| `prompt_data` | Text | JSON string containing prompts |
| `business_context` | Text (nullable) | Business context description |
| `created_at` | DateTime | Generation timestamp |

**prompt_data JSON structure:**
```json
{
  "prompts": [
    "What are your business hours?",
    "How can I contact support?",
    "..."
  ],
  "categories": {
    "general": ["prompt1", "prompt2"],
    "support": ["prompt3"]
  }
}
```

---

## SQLite Schema

Each project is stored as a separate `.db` file in the `projects/` directory.

### File Structure
```
projects/
├── my-project/
│   └── my-project.db
├── another-project/
│   └── another-project.db
```

### Table Definitions

#### metadata
```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Stores: embedding_model, vector_dimension
```

#### qa_pairs (Virtual Table)
```sql
CREATE VIRTUAL TABLE qa_pairs USING vec0(
    prompt_embedding float[{vector_dim}]
);
-- rowid corresponds to qa_text.id
```

#### qa_text
```sql
CREATE TABLE qa_text (
    id INTEGER PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    weight REAL DEFAULT 1.0
);
```

#### scraped_content
```sql
CREATE TABLE scraped_content (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    domain TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### prompt_files
```sql
CREATE TABLE prompt_files (
    id INTEGER PRIMARY KEY,
    prompt_data TEXT NOT NULL,
    business_context TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Vector Search Query
```sql
SELECT 
    t.id,
    t.prompt_text,
    t.response_text,
    t.weight,
    v.distance
FROM qa_pairs v
INNER JOIN qa_text t ON v.rowid = t.id
WHERE v.prompt_embedding MATCH ?
    AND k = ?
ORDER BY v.distance ASC
```

---

## PostgreSQL Schema

All projects share a single database with foreign key relationships.

### SQLAlchemy Models

Located in `src/models_db.py`:

```python
class Project(Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    embedding_model: Mapped[str] = mapped_column(String)
    vector_dimension: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

class QAPair(Base):
    __tablename__ = "qa_pairs"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    prompt_text: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    embedding: Mapped[Vector] = mapped_column(Vector)

class ScrapedContent(Base):
    __tablename__ = "scraped_content"
    # ... similar structure

class PromptFile(Base):
    __tablename__ = "prompt_files"
    # ... similar structure
```

### Vector Search Query (pgvector)
```python
# Cosine distance search
results = await session.execute(
    select(QAPair)
    .where(QAPair.project_id == project.id)
    .order_by(QAPair.embedding.cosine_distance(query_embedding))
    .limit(top_k)
)
```

---

## API Models (Pydantic)

Located in `src/models.py`:

### Request Models

```python
class ProjectCreate(BaseModel):
    project_name: str
    embedding_model: str = "local_default"

class AddData(BaseModel):
    prompt: str
    response: str
    weight: float = 1.0

class ChatRequest(BaseModel):
    prompt: str
    top_k: int = 5

class AutoGenerateRequest(BaseModel):
    project_name: str
    url: str
    max_pages: int = 10
    business_context: Optional[str] = None
    default_weight: float = 1.0
```

### Response Models

```python
class ChatResponseItem(BaseModel):
    response_text: str
    original_prompt: str
    similarity_score: float
    weight: float = 1.0
    weighted_similarity: Optional[float] = None
```

---

## Configuration Structures

### config.toml
```toml
[database]
type = "sqlite"  # or "postgresql"
url = "postgresql+asyncpg://localhost/strialm"
sqlite_path = "projects"

[embedding]
default_model = "local_default"

[embedding.models.local_default]
category = "Local"
model = "sentence-transformers/all-MiniLM-L6-v2"

[embedding.models.openai_default]
category = "OpenAI API"
model = "text-embedding-ada-002"
base_url = "https://api.openai.com/v1"

[experimental]
enable_weights = true
```

### .env File
```bash
DATABASE_TYPE=sqlite
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/strialm
OPENROUTER_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ENABLE_WEIGHTS=true
```

---

## Vector Embeddings

### Supported Models

| Model ID | Category | Model Path | Dimension |
|----------|----------|------------|-----------|
| `local_default` | Local | `sentence-transformers/all-MiniLM-L6-v2` | 384 |
| `openai_default` | OpenAI API | `text-embedding-ada-002` | 1536 |

### Embedding Format

- **Type**: `numpy.ndarray` with `dtype=float32`
- **Shape**: `(vector_dimension,)`
- **Normalization**: Depends on the model (sentence-transformers typically L2-normalized)

### Storage Format

**SQLite (sqlite-vec):**
- Stored as raw bytes: `embedding.astype(np.float32).tobytes()`
- Query format: JSON array string

**PostgreSQL (pgvector):**
- Stored as `Vector` type
- Native array operations supported

---

## Extension Points

### Adding New Data Types

1. **Define the model** in `src/models_db.py` (PostgreSQL)
2. **Add table creation** in `src/database/sqlite.py` (SQLite)
3. **Add abstract methods** to `src/database/base.py`
4. **Implement methods** in both backends

### Adding New Embedding Models

1. **Add config entry** in `config.toml` under `[embedding.models.your_model]`
2. **Implement handler** in `src/embedding.py` if new category

### Adding New Search Algorithms

The `find_similar_prompts` method in each backend can be extended to support:
- Different distance metrics (L2, cosine, inner product)
- Hybrid search (combining vector + keyword)
- Filtered search (with additional WHERE clauses)

### Export/Import Extensions

The `export_to_sqlite` and `import_from_sqlite` methods provide:
- **Export**: Convert any backend to portable SQLite format
- **Import**: Load SQLite databases into any backend

This enables:
- Offline-to-online migration
- Backup and restore
- Cross-environment deployment

---

## Version History

| Version | Changes |
|---------|---------|
| 0.1.0 | Initial SQLite-only implementation |
| 0.2.0 | Added PostgreSQL support, database abstraction layer, import/export |
