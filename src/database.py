import sqlite3
import json
from pathlib import Path
import numpy as np
import sqlite_vec
from .config import ENABLE_WEIGHTS
from . import embedding

def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Establishes a database connection and loads the vec extension."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def get_db_path(project_name: str, projects_dir: Path) -> Path:
    return projects_dir / project_name / f"{project_name}.db"

def init_db(project_name: str, embedding_model: str, vector_dim: int, projects_dir: Path):
    """
    Initializes a new SQLite database for a project.
    """
    project_dir = projects_dir / project_name
    project_dir.mkdir(exist_ok=True)
    db_path = get_db_path(project_name, projects_dir)

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Create metadata table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)
        
        metadata = {
            "embedding_model": embedding_model,
            "vector_dimension": vector_dim
        }
        for key, value in metadata.items():
            cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, str(value)))

        # Create virtual table for vector search using vec0
        cursor.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS qa_pairs USING vec0(
            prompt_embedding float[{vector_dim}]
        )
        """)

        # Create a regular table to store the text data
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_text (
            id INTEGER PRIMARY KEY,
            prompt_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            weight REAL DEFAULT 1.0
        )
        """)
        
        # Create table for scraped website content
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraped_content (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            domain TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create table for generated prompt files
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompt_files (
            id INTEGER PRIMARY KEY,
            prompt_data TEXT NOT NULL,  -- JSON string of prompts
            business_context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()

def add_qa_pair(project_name: str, prompt: str, response: str, embedding: np.ndarray, projects_dir: Path, weight: float = 1.0):
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Insert text data with weight if enabled, otherwise use default
        if ENABLE_WEIGHTS:
            cursor.execute("INSERT INTO qa_text (prompt_text, response_text, weight) VALUES (?, ?, ?)", (prompt, response, weight))
        else:
            cursor.execute("INSERT INTO qa_text (prompt_text, response_text) VALUES (?, ?)", (prompt, response))
        rowid = cursor.lastrowid

        # Insert the embedding into the vec table with the same rowid
        # Using the compact binary format for storage
        embedding_bytes = embedding.astype(np.float32).tobytes()
        cursor.execute("INSERT INTO qa_pairs (rowid, prompt_embedding) VALUES (?, ?)", (rowid, embedding_bytes))
        
        conn.commit()

def find_similar_prompts(project_name: str, query_embedding: np.ndarray, top_k: int, projects_dir: Path):
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # For querying with MATCH, the vector needs to be a JSON string
        query_embedding_json = json.dumps(query_embedding.tolist())

        if ENABLE_WEIGHTS:
            # Query with weight-based ranking when weights are enabled
            query = """
            SELECT
                t.id,
                t.response_text,
                t.prompt_text,
                v.distance,
                COALESCE(t.weight, 1.0) AS weight,
                (1 - v.distance) * COALESCE(t.weight, 1.0) AS weighted_similarity
            FROM qa_pairs v
            JOIN qa_text t ON v.rowid = t.id
            WHERE v.prompt_embedding MATCH ? AND k = ?
            ORDER BY weighted_similarity DESC
            """

            cursor.execute(query, (query_embedding_json, top_k))
            results = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "response_text": row[1],
                    "original_prompt": row[2],
                    "similarity_score": 1 - row[3],
                    "weight": row[4],
                    "weighted_similarity": row[5],
                }
                for row in results
            ]
        else:
            # Original query without weights when weights are disabled
            query = """
            SELECT
                t.id,
                t.response_text,
                t.prompt_text,
                v.distance,
                COALESCE(t.weight, 1.0) AS weight,
                (1 - v.distance) * COALESCE(t.weight, 1.0) AS weighted_similarity
            FROM qa_pairs v
            JOIN qa_text t ON v.rowid = t.id
            WHERE v.prompt_embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
            """

            cursor.execute(query, (query_embedding_json, top_k))
            results = cursor.fetchall()

            return [
                {
                    "id": row[0],
                    "response_text": row[1],
                    "original_prompt": row[2],
                    "similarity_score": 1 - row[3],
                    "weight": row[4],
                    "weighted_similarity": row[5],
                }
                for row in results
            ]

def re_embed_prompts(project_name: str, projects_dir: Path, ids: str | list[int] = "all", progress_callback=None):
    """
    Re-generates embeddings for specified prompts in the qa_text table.
    """
    db_path = get_db_path(project_name, projects_dir)
    metadata = get_project_metadata(project_name, projects_dir)
    if not metadata:
        raise ValueError(f"Project '{project_name}' not found.")

    model_name = metadata.get("embedding_model")
    if not model_name:
        raise ValueError("Embedding model not found in project metadata.")

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        if ids == "all":
            cursor.execute("SELECT id, prompt_text FROM qa_text")
        else:
            placeholders = ','.join('?' for _ in ids)
            cursor.execute(f"SELECT id, prompt_text FROM qa_text WHERE id IN ({placeholders})", ids)
        
        prompts_to_re_embed = cursor.fetchall()
        total_prompts = len(prompts_to_re_embed)

        if total_prompts == 0:
            return 0

        for i, (row_id, prompt_text) in enumerate(prompts_to_re_embed):
            # First, delete the old embedding for this rowid
            cursor.execute("DELETE FROM qa_pairs WHERE rowid = ?", (row_id,))

            new_embedding = embedding.generate_embedding(prompt_text, model_name)
            embedding_bytes = new_embedding.astype(np.float32).tobytes()
            
            cursor.execute("INSERT INTO qa_pairs (rowid, prompt_embedding) VALUES (?, ?)", (row_id, embedding_bytes))
            
            if progress_callback:
                progress_callback(i + 1, total_prompts)

        conn.commit()
        return total_prompts

def get_project_metadata(project_name: str, projects_dir: Path):
    db_path = get_db_path(project_name, projects_dir)
    if not db_path.exists():
        return None
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM metadata")
        return dict(cursor.fetchall())

def add_scraped_content(project_name: str, scraped_data: list, projects_dir: Path):
    """Add scraped website content to the database"""
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        for page_data in scraped_data:
            cursor.execute("""
                INSERT INTO scraped_content (url, title, content, domain)
                VALUES (?, ?, ?, ?)
            """, (
                page_data.get('url', ''),
                page_data.get('title', ''),
                page_data.get('content', ''),
                page_data.get('domain', '')
            ))
        
        conn.commit()

def get_scraped_content(project_name: str, projects_dir: Path):
    """Retrieve all scraped content for a project"""
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, content, domain FROM scraped_content")
        results = cursor.fetchall()
        
        return [
            {
                "url": row[0],
                "title": row[1],
                "content": row[2],
                "domain": row[3]
            }
            for row in results
        ]

def add_prompt_file(project_name: str, prompt_data: str, business_context: str, projects_dir: Path):
    """Add generated prompt file to the database"""
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prompt_files (prompt_data, business_context)
            VALUES (?, ?)
        """, (prompt_data, business_context))
        conn.commit()
        return cursor.lastrowid

def get_latest_prompt_file(project_name: str, projects_dir: Path):
    """Get the latest generated prompt file for a project"""
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT prompt_data, business_context 
            FROM prompt_files 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        result = cursor.fetchone()
        
        if result:
            return {
                "prompt_data": result[0],
                "business_context": result[1]
            }
        return None
