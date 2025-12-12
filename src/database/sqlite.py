"""
SQLite + sqlite-vec database backend.
Each project is stored as a separate .db file in the projects directory.
"""

import sqlite3
import json
import uuid as uuid_module
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import sqlite_vec

from .base import DatabaseBackend


class SQLiteBackend(DatabaseBackend):
    """SQLite + sqlite-vec database backend."""
    
    db_type = "sqlite"
    
    def __init__(self):
        from ..config import PROJECTS_DIR, ENABLE_WEIGHTS
        
        self.projects_dir = Path(PROJECTS_DIR)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.enable_weights = ENABLE_WEIGHTS
        self._connections: Dict[str, sqlite3.Connection] = {}
    
    def _get_db_path(self, project_name: str) -> Path:
        """Get the database file path for a project."""
        return self.projects_dir / project_name / f"{project_name}.db"
    
    def _get_connection(self, project_name: str) -> sqlite3.Connection:
        """Get or create a connection for a project."""
        if project_name not in self._connections:
            db_path = self._get_db_path(project_name)
            if not db_path.exists():
                raise ValueError(f"Project '{project_name}' not found at {db_path}")
            
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._connections[project_name] = conn
        
        return self._connections[project_name]
    
    def _create_new_connection(self, db_path: Path) -> sqlite3.Connection:
        """Create a new connection and load sqlite-vec."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn
    
    def close(self):
        """Close all open connections."""
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()
    
    # ==================== Init ====================
    
    def init_db(self, project_name: Optional[str] = None) -> None:
        """
        Initialize database for a project.
        For SQLite, this creates the project's .db file and tables.
        """
        if project_name is None:
            # Nothing to do - projects are initialized on creation
            return
        
        db_path = self._get_db_path(project_name)
        if db_path.exists():
            return  # Already initialized
        
        # Will be created in create_project
    
    # ==================== Project Operations ====================
    
    def create_project(self, project_name: str, embedding_model: str, vector_dim: int) -> Dict[str, Any]:
        """Create a new SQLite database file for the project."""
        db_path = self._get_db_path(project_name)
        
        if db_path.exists():
            raise ValueError(f"Project '{project_name}' already exists.")
        
        conn = self._create_new_connection(db_path)
        cursor = conn.cursor()
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        # Store metadata
        cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", 
                      ("embedding_model", embedding_model))
        cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)", 
                      ("vector_dimension", str(vector_dim)))
        
        # Create vector table
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS qa_pairs USING vec0(
                prompt_embedding float[{vector_dim}]
            )
        """)
        
        # Create text data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_text (
                id INTEGER PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                weight REAL DEFAULT 1.0
            )
        """)
        
        # Create scraped content table
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
        
        # Create prompt files table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_files (
                id INTEGER PRIMARY KEY,
                prompt_data TEXT NOT NULL,
                business_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create actors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS actors (
                actor_id INTEGER PRIMARY KEY,
                actor_name TEXT NOT NULL UNIQUE,
                description TEXT,
                prompt_messages TEXT NOT NULL DEFAULT '[]',
                model_name TEXT NOT NULL,
                temperature REAL DEFAULT 0.7,
                max_tokens INTEGER DEFAULT 2048,
                top_p REAL DEFAULT 1.0,
                top_k INTEGER,
                repetition_penalty REAL,
                other_generation_parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create personas table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personas (
                persona_id TEXT PRIMARY KEY,
                persona_name TEXT NOT NULL,
                display_name TEXT,
                description TEXT,
                avatar_url TEXT,
                is_ai INTEGER DEFAULT 0,
                fallback_actor_id INTEGER REFERENCES actors(actor_id),
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL REFERENCES personas(persona_id),
                actor_id INTEGER NOT NULL REFERENCES actors(actor_id),
                title TEXT,
                total_input_tokens INTEGER DEFAULT 0,
                total_output_tokens INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create chat messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER DEFAULT 0,
                context_metadata TEXT,
                generation_metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        self._connections[project_name] = conn
        
        return {
            "name": project_name,
            "embedding_model": embedding_model,
            "vector_dimension": vector_dim
        }
    
    def get_project(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get project metadata."""
        db_path = self._get_db_path(project_name)
        if not db_path.exists():
            return None
        
        try:
            conn = self._get_connection(project_name)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM metadata")
            metadata = dict(cursor.fetchall())
            
            return {
                "name": project_name,
                "embedding_model": metadata.get("embedding_model", ""),
                "vector_dimension": int(metadata.get("vector_dimension", 384))
            }
        except Exception:
            return None
    
    def list_projects(self) -> List[str]:
        """List all projects by scanning the projects directory."""
        projects = []
        if not self.projects_dir.exists():
            return projects
        
        for entry in self.projects_dir.iterdir():
            if not entry.is_dir():
                continue
            db_path = entry / f"{entry.name}.db"
            if db_path.exists():
                projects.append(entry.name)
        
        return sorted(projects)
    
    def delete_project(self, project_name: str) -> bool:
        """Delete a project by removing its database file."""
        db_path = self._get_db_path(project_name)
        
        if project_name in self._connections:
            self._connections[project_name].close()
            del self._connections[project_name]
        
        if db_path.exists():
            db_path.unlink()
            # Remove empty directory
            if db_path.parent.exists() and not any(db_path.parent.iterdir()):
                db_path.parent.rmdir()
            return True
        return False
    
    def get_project_metadata(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a project."""
        project = self.get_project(project_name)
        if project:
            return {
                "embedding_model": project["embedding_model"],
                "vector_dimension": project["vector_dimension"]
            }
        return None
    
    # ==================== QA Pair Operations ====================
    
    def add_qa_pair(
        self, 
        project_name: str, 
        prompt: str, 
        response: str, 
        embedding: np.ndarray, 
        weight: float = 1.0
    ) -> int:
        """Add a QA pair to the project."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        # Insert text data
        if self.enable_weights:
            cursor.execute(
                "INSERT INTO qa_text (prompt_text, response_text, weight) VALUES (?, ?, ?)",
                (prompt, response, weight)
            )
        else:
            cursor.execute(
                "INSERT INTO qa_text (prompt_text, response_text) VALUES (?, ?)",
                (prompt, response)
            )
        rowid = cursor.lastrowid
        
        # Insert embedding
        embedding_bytes = embedding.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO qa_pairs (rowid, prompt_embedding) VALUES (?, ?)",
            (rowid, embedding_bytes)
        )
        
        conn.commit()
        return rowid
    
    def get_qa_pairs(self, project_name: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Get all QA pairs for a project."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, prompt_text, response_text, weight FROM qa_text ORDER BY id LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        
        return [
            {
                "id": row["id"],
                "prompt_text": row["prompt_text"],
                "response_text": row["response_text"],
                "weight": row["weight"]
            }
            for row in rows
        ]
    
    def update_qa_pair(
        self, 
        project_name: str, 
        qa_id: int, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update a QA pair with the provided field values."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        sql_updates = []
        values = []
        
        if 'prompt' in updates:
            sql_updates.append("prompt_text = ?")
            values.append(updates['prompt'])
        if 'response' in updates:
            sql_updates.append("response_text = ?")
            values.append(updates['response'])
        if 'weight' in updates:
            sql_updates.append("weight = ?")
            values.append(updates['weight'])
        
        if sql_updates:
            values.append(qa_id)
            cursor.execute(
                f"UPDATE qa_text SET {', '.join(sql_updates)} WHERE id = ?",
                tuple(values)
            )
        
        if 'embedding' in updates:
            embedding = updates['embedding']
            # Delete old embedding and insert new
            cursor.execute("DELETE FROM qa_pairs WHERE rowid = ?", (qa_id,))
            embedding_bytes = embedding.astype(np.float32).tobytes()
            cursor.execute(
                "INSERT INTO qa_pairs (rowid, prompt_embedding) VALUES (?, ?)",
                (qa_id, embedding_bytes)
            )
        
        conn.commit()
        return True
    
    def delete_qa_pair(self, project_name: str, qa_id: int) -> bool:
        """Delete a QA pair."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM qa_text WHERE id = ?", (qa_id,))
        cursor.execute("DELETE FROM qa_pairs WHERE rowid = ?", (qa_id,))
        
        conn.commit()
        return cursor.rowcount > 0
    
    def find_similar_prompts(
        self, 
        project_name: str, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar prompts using vector similarity search."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        # Convert query to JSON for sqlite-vec MATCH
        query_json = json.dumps(query_embedding.tolist())
        
        if self.enable_weights:
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
        else:
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
        
        cursor.execute(query, (query_json, top_k))
        results = cursor.fetchall()
        
        return [
            {
                "id": row[0],
                "response_text": row[1],
                "original_prompt": row[2],
                "similarity_score": 1 - row[3],
                "weight": row[4],
                "weighted_similarity": row[5] if self.enable_weights else None,
            }
            for row in results
        ]
    
    def re_embed_prompts(
        self, 
        project_name: str, 
        ids: str | List[int] = "all",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """Re-generate embeddings for prompts."""
        from .. import embedding as emb_module
        
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        # Get metadata for model name
        cursor.execute("SELECT value FROM metadata WHERE key = 'embedding_model'")
        row = cursor.fetchone()
        model_name = row[0] if row else "sentence-transformers/all-MiniLM-L6-v2"
        
        # Get prompts to re-embed
        if ids == "all":
            cursor.execute("SELECT id, prompt_text FROM qa_text")
        else:
            placeholders = ','.join('?' for _ in ids)
            cursor.execute(f"SELECT id, prompt_text FROM qa_text WHERE id IN ({placeholders})", ids)
        
        prompts = cursor.fetchall()
        total = len(prompts)
        
        if total == 0:
            return 0
        
        for i, (row_id, prompt_text) in enumerate(prompts):
            # Delete old embedding
            cursor.execute("DELETE FROM qa_pairs WHERE rowid = ?", (row_id,))
            
            # Generate new embedding
            new_embedding = emb_module.generate_embedding(prompt_text, model_name)
            embedding_bytes = new_embedding.astype(np.float32).tobytes()
            
            cursor.execute(
                "INSERT INTO qa_pairs (rowid, prompt_embedding) VALUES (?, ?)",
                (row_id, embedding_bytes)
            )
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        conn.commit()
        return total
    
    # ==================== Scraped Content Operations ====================
    
    def add_scraped_content(self, project_name: str, scraped_data: List[Dict[str, Any]]) -> int:
        """Add scraped content to the project."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        for page_data in scraped_data:
            cursor.execute(
                "INSERT INTO scraped_content (url, title, content, domain) VALUES (?, ?, ?, ?)",
                (
                    page_data.get('url', ''),
                    page_data.get('title', ''),
                    page_data.get('content', ''),
                    page_data.get('domain', '')
                )
            )
        
        conn.commit()
        return len(scraped_data)
    
    def get_scraped_content(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all scraped content for a project."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT url, title, content, domain FROM scraped_content")
        rows = cursor.fetchall()
        
        return [
            {
                "url": row["url"],
                "title": row["title"],
                "content": row["content"],
                "domain": row["domain"]
            }
            for row in rows
        ]
    
    # ==================== Prompt File Operations ====================
    
    def add_prompt_file(self, project_name: str, prompt_data: str, business_context: str) -> int:
        """Add a prompt file to the project."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO prompt_files (prompt_data, business_context) VALUES (?, ?)",
            (prompt_data, business_context)
        )
        conn.commit()
        return cursor.lastrowid
    
    def get_latest_prompt_file(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest prompt file for a project."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT prompt_data, business_context FROM prompt_files ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        
        if row:
            return {
                "prompt_data": row["prompt_data"],
                "business_context": row["business_context"]
            }
        return None
    
    # ==================== Weight Operations ====================
    
    def update_qa_weight(self, project_name: str, qa_id: int, new_weight: float) -> bool:
        """Update the weight of a QA pair."""
        if not self.enable_weights:
            return False
        if not (0.1 <= new_weight <= 10.0):
            raise ValueError("Weight must be between 0.1 and 10.0")
        
        return self.update_qa_pair(project_name, qa_id, weight=new_weight)
    
    def get_qa_pairs_with_weights(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all QA pairs with weights."""
        return self.get_qa_pairs(project_name, limit=10000)
    
    def reset_all_weights(self, project_name: str) -> int:
        """Reset all weights to 1.0."""
        if not self.enable_weights:
            return 0
        
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE qa_text SET weight = 1.0")
        conn.commit()
        return cursor.rowcount
    
    def get_weight_statistics(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get weight statistics for a project."""
        if not self.enable_weights:
            return None
        
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(*) as total_pairs,
                AVG(weight) as avg_weight,
                MIN(weight) as min_weight,
                MAX(weight) as max_weight,
                COUNT(CASE WHEN weight != 1.0 THEN 1 END) as modified_weights
            FROM qa_text
        """)
        row = cursor.fetchone()
        
        if row:
            return {
                "total_pairs": row[0],
                "avg_weight": round(row[1], 3) if row[1] else 1.0,
                "min_weight": row[2] if row[2] else 1.0,
                "max_weight": row[3] if row[3] else 1.0,
                "modified_weights": row[4],
                "weights_enabled": True
            }
        return None
    
    # ==================== Utility Operations ====================
    
    def get_tables(self, project_name: Optional[str] = None) -> List[str]:
        """Get list of tables in the database."""
        if project_name is None:
            return []
        
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        )
        return [row[0] for row in cursor.fetchall()]
    
    def get_table_data(
        self, 
        project_name: str, 
        table_name: str, 
        limit: int = 200
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Get data from a specific table."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Get data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        data = [dict(row) for row in rows]
        
        return columns, data
    
    def execute_update(
        self, 
        project_name: str, 
        table_name: str, 
        pk_column: str,
        pk_value: Any,
        column: str, 
        new_value: Any
    ) -> bool:
        """Execute an update on a specific cell."""
        conn = self._get_connection(project_name)
        cursor = conn.cursor()
        
        cursor.execute(
            f'UPDATE "{table_name}" SET "{column}" = ? WHERE "{pk_column}" = ?',
            (new_value, pk_value)
        )
        conn.commit()
        return cursor.rowcount > 0
    
    # ==================== Export/Import ====================
    
    def export_to_sqlite(self, project_name: str, output_path: str) -> bool:
        """Export (copy) the SQLite database to a new location."""
        import shutil
        
        db_path = self._get_db_path(project_name)
        if not db_path.exists():
            raise ValueError(f"Project '{project_name}' not found.")
        
        # Close connection if open
        if project_name in self._connections:
            self._connections[project_name].close()
            del self._connections[project_name]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, output_path)
        return True
    
    def import_from_sqlite(self, project_name: str, input_path: str) -> bool:
        """Import a SQLite database file as a new project."""
        import shutil
        
        input_file = Path(input_path)
        if not input_file.exists():
            raise ValueError(f"Input file '{input_path}' not found.")
        
        db_path = self._get_db_path(project_name)
        if db_path.exists():
            raise ValueError(f"Project '{project_name}' already exists.")
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, db_path)
        return True

    # ==================== Actor Operations ====================
    
    def _ensure_actor_tables(self, conn: sqlite3.Connection):
        """Ensure actor/persona/session tables exist (for existing projects)."""
        cursor = conn.cursor()
        
        # Check if actors table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='actors'")
        if not cursor.fetchone():
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actors (
                    actor_id INTEGER PRIMARY KEY,
                    actor_name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    prompt_messages TEXT NOT NULL DEFAULT '[]',
                    model_name TEXT NOT NULL,
                    temperature REAL DEFAULT 0.7,
                    max_tokens INTEGER DEFAULT 2048,
                    top_p REAL DEFAULT 1.0,
                    top_k INTEGER,
                    repetition_penalty REAL,
                    other_generation_parameters TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personas (
                    persona_id TEXT PRIMARY KEY,
                    persona_name TEXT NOT NULL,
                    display_name TEXT,
                    description TEXT,
                    avatar_url TEXT,
                    is_ai INTEGER DEFAULT 0,
                    fallback_actor_id INTEGER REFERENCES actors(actor_id),
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    persona_id TEXT NOT NULL REFERENCES personas(persona_id),
                    actor_id INTEGER NOT NULL REFERENCES actors(actor_id),
                    title TEXT,
                    total_input_tokens INTEGER DEFAULT 0,
                    total_output_tokens INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    context_metadata TEXT,
                    generation_metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def create_actor(
        self, 
        project_name: str,
        actor_name: str,
        description: str = "",
        prompt_messages: Optional[List[Dict[str, str]]] = None,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        other_generation_parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO actors (
                actor_name, description, prompt_messages, model_name,
                temperature, max_tokens, top_p, top_k, repetition_penalty,
                other_generation_parameters
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            actor_name,
            description,
            json.dumps(prompt_messages or []),
            model_name,
            temperature,
            max_tokens,
            top_p,
            top_k,
            repetition_penalty,
            json.dumps(other_generation_parameters) if other_generation_parameters else None
        ))
        conn.commit()
        return str(cursor.lastrowid)
    
    def get_actor(self, project_name: str, actor_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM actors WHERE actor_id = ?", (int(actor_id),))
        row = cursor.fetchone()
        
        if row:
            return {
                "actor_id": str(row["actor_id"]),
                "actor_name": row["actor_name"],
                "description": row["description"],
                "prompt_messages": json.loads(row["prompt_messages"]) if row["prompt_messages"] else [],
                "model_name": row["model_name"],
                "temperature": row["temperature"],
                "max_tokens": row["max_tokens"],
                "top_p": row["top_p"],
                "top_k": row["top_k"],
                "repetition_penalty": row["repetition_penalty"],
                "other_generation_parameters": json.loads(row["other_generation_parameters"]) if row["other_generation_parameters"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        return None
    
    def get_actor_by_name(self, project_name: str, actor_name: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM actors WHERE actor_name = ?", (actor_name,))
        row = cursor.fetchone()
        
        if row:
            return {
                "actor_id": str(row["actor_id"]),
                "actor_name": row["actor_name"],
                "description": row["description"],
                "prompt_messages": json.loads(row["prompt_messages"]) if row["prompt_messages"] else [],
                "model_name": row["model_name"],
                "temperature": row["temperature"],
                "max_tokens": row["max_tokens"],
                "top_p": row["top_p"],
                "top_k": row["top_k"],
                "repetition_penalty": row["repetition_penalty"],
                "other_generation_parameters": json.loads(row["other_generation_parameters"]) if row["other_generation_parameters"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        return None
    
    def list_actors(self, project_name: str) -> List[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM actors ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        return [
            {
                "actor_id": str(row["actor_id"]),
                "actor_name": row["actor_name"],
                "description": row["description"],
                "model_name": row["model_name"],
                "temperature": row["temperature"],
                "max_tokens": row["max_tokens"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    
    def update_actor(self, project_name: str, actor_id: str, updates: Dict[str, Any]) -> bool:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        valid_columns = {
            'actor_name', 'description', 'prompt_messages', 'model_name',
            'temperature', 'max_tokens', 'top_p', 'top_k', 
            'repetition_penalty', 'other_generation_parameters'
        }
        
        set_parts = []
        values = []
        for k, v in updates.items():
            if k in valid_columns:
                set_parts.append(f"{k} = ?")
                if k in ('prompt_messages', 'other_generation_parameters') and isinstance(v, (dict, list)):
                    values.append(json.dumps(v))
                else:
                    values.append(v)
        
        if not set_parts:
            return False
        
        set_parts.append("updated_at = CURRENT_TIMESTAMP")
        values.append(int(actor_id))
        
        cursor.execute(
            f"UPDATE actors SET {', '.join(set_parts)} WHERE actor_id = ?",
            values
        )
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_actor(self, project_name: str, actor_id: str) -> bool:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM actors WHERE actor_id = ?", (int(actor_id),))
        conn.commit()
        return cursor.rowcount > 0
    
    # ==================== Persona Operations ====================
    
    def create_persona(
        self, 
        project_name: str,
        persona_name: str,
        display_name: str,
        is_ai: bool = False,
        fallback_actor_id: Optional[str] = None
    ) -> str:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        persona_id = str(uuid_module.uuid4())
        
        cursor.execute("""
            INSERT INTO personas (persona_id, persona_name, display_name, is_ai, fallback_actor_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            persona_id,
            persona_name,
            display_name,
            1 if is_ai else 0,
            int(fallback_actor_id) if fallback_actor_id else None
        ))
        conn.commit()
        return persona_id
    
    def get_persona(self, project_name: str, persona_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM personas WHERE persona_id = ?", (persona_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "persona_id": row["persona_id"],
                "persona_name": row["persona_name"],
                "display_name": row["display_name"],
                "description": row["description"],
                "avatar_url": row["avatar_url"],
                "is_ai": bool(row["is_ai"]),
                "fallback_actor_id": str(row["fallback_actor_id"]) if row["fallback_actor_id"] else None,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                "created_at": row["created_at"]
            }
        return None
    
    def get_persona_by_name(self, project_name: str, persona_name: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM personas WHERE persona_name = ?", (persona_name,))
        row = cursor.fetchone()
        
        if row:
            return {
                "persona_id": row["persona_id"],
                "persona_name": row["persona_name"],
                "display_name": row["display_name"],
                "description": row["description"],
                "avatar_url": row["avatar_url"],
                "is_ai": bool(row["is_ai"]),
                "fallback_actor_id": str(row["fallback_actor_id"]) if row["fallback_actor_id"] else None,
                "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                "created_at": row["created_at"]
            }
        return None
    
    def list_personas(self, project_name: str) -> List[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM personas ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        return [
            {
                "persona_id": row["persona_id"],
                "persona_name": row["persona_name"],
                "display_name": row["display_name"],
                "is_ai": bool(row["is_ai"]),
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    
    def delete_persona(self, project_name: str, persona_id: str) -> bool:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM personas WHERE persona_id = ?", (persona_id,))
        conn.commit()
        return cursor.rowcount > 0
    
    # ==================== Chat Session Operations ====================
    
    def create_chat_session(
        self,
        project_name: str,
        actor_id: str,
        persona_id: str,
        session_name: Optional[str] = None
    ) -> str:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        session_id = str(uuid_module.uuid4())
        
        cursor.execute("""
            INSERT INTO chat_sessions (session_id, persona_id, actor_id, title)
            VALUES (?, ?, ?, ?)
        """, (session_id, persona_id, int(actor_id), session_name))
        conn.commit()
        return session_id
    
    def get_chat_session(self, project_name: str, session_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                "session_id": row["session_id"],
                "actor_id": str(row["actor_id"]),
                "persona_id": row["persona_id"],
                "title": row["title"],
                "total_input_tokens": row["total_input_tokens"],
                "total_output_tokens": row["total_output_tokens"],
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
        return None
    
    def list_chat_sessions(
        self, 
        project_name: str,
        persona_id: Optional[str] = None,
        actor_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        sql = "SELECT * FROM chat_sessions WHERE 1=1"
        params = []
        
        if persona_id:
            sql += " AND persona_id = ?"
            params.append(persona_id)
        if actor_id:
            sql += " AND actor_id = ?"
            params.append(int(actor_id))
        
        sql += " ORDER BY updated_at DESC"
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        
        return [
            {
                "session_id": row["session_id"],
                "actor_id": str(row["actor_id"]),
                "persona_id": row["persona_id"],
                "title": row["title"],
                "total_input_tokens": row["total_input_tokens"],
                "total_output_tokens": row["total_output_tokens"],
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }
            for row in rows
        ]
    
    def update_session_tokens(
        self,
        project_name: str,
        session_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> bool:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE chat_sessions 
            SET total_input_tokens = total_input_tokens + ?,
                total_output_tokens = total_output_tokens + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ?
        """, (input_tokens, output_tokens, session_id))
        conn.commit()
        return cursor.rowcount > 0
    
    def delete_chat_session(self, project_name: str, session_id: str) -> bool:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        # Delete messages first (no cascade in SQLite by default)
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0
    
    # ==================== Chat Message Operations ====================
    
    def add_chat_message(
        self,
        project_name: str,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        context_used: Optional[Dict[str, Any]] = None,
        generation_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        message_id = str(uuid_module.uuid4())
        
        cursor.execute("""
            INSERT INTO chat_messages (
                message_id, session_id, role, content, token_count,
                context_metadata, generation_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id,
            session_id,
            role,
            content,
            token_count,
            json.dumps(context_used) if context_used else None,
            json.dumps(generation_metadata) if generation_metadata else None
        ))
        
        # Update session timestamp
        cursor.execute(
            "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
        return message_id
    
    def get_chat_history(
        self,
        project_name: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        if limit:
            # Get last N messages
            cursor.execute("""
                SELECT * FROM chat_messages 
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, limit))
            rows = list(reversed(cursor.fetchall()))
        else:
            cursor.execute("""
                SELECT * FROM chat_messages 
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,))
            rows = cursor.fetchall()
        
        return [
            {
                "message_id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "token_count": row["token_count"],
                "context_metadata": json.loads(row["context_metadata"]) if row["context_metadata"] else None,
                "generation_metadata": json.loads(row["generation_metadata"]) if row["generation_metadata"] else None,
                "created_at": row["created_at"]
            }
            for row in rows
        ]
    
    def get_chat_context_window(
        self,
        project_name: str,
        session_id: str,
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Get most recent messages that fit within token budget."""
        conn = self._get_connection(project_name)
        self._ensure_actor_tables(conn)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chat_messages 
            WHERE session_id = ?
            ORDER BY created_at DESC
        """, (session_id,))
        
        rows = cursor.fetchall()
        
        selected = []
        token_total = 0
        
        for row in rows:
            if token_total + row["token_count"] <= max_tokens:
                selected.append(row)
                token_total += row["token_count"]
            else:
                break
        
        # Reverse to chronological order
        selected = list(reversed(selected))
        
        return [
            {
                "message_id": row["message_id"],
                "role": row["role"],
                "content": row["content"],
                "token_count": row["token_count"]
            }
            for row in selected
        ]