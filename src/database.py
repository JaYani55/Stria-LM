import sqlite3
import json
from pathlib import Path
import numpy as np
import sqlite_vec

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
            response_text TEXT NOT NULL
        )
        """)
        conn.commit()

def add_qa_pair(project_name: str, prompt: str, response: str, embedding: np.ndarray, projects_dir: Path):
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # Insert text data and get the rowid
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

        # Query for similar prompts using the MATCH operator and the 'k' parameter
        query = """
        SELECT
            t.response_text,
            t.prompt_text,
            v.distance
        FROM qa_pairs v
        JOIN qa_text t ON v.rowid = t.id
        WHERE v.prompt_embedding MATCH ? AND k = ?
        """
        
        cursor.execute(query, (query_embedding_json, top_k))
        results = cursor.fetchall()
        
        return [
            {
                "response_text": row[0],
                "original_prompt": row[1],
                "similarity_score": 1 - row[2] # vec distance is L2, 1-dist is a simple similarity metric
            }
            for row in results
        ]

def get_project_metadata(project_name: str, projects_dir: Path):
    db_path = get_db_path(project_name, projects_dir)
    if not db_path.exists():
        return None
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM metadata")
        return dict(cursor.fetchall())
