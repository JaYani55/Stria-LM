"""
PostgreSQL + pgvector database backend.
Uses async SQLAlchemy with asyncpg driver.
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, delete, func, text
from sqlalchemy.exc import OperationalError

from .base import DatabaseBackend
from ..models_db import Base, Project, QAPair, ScrapedContent, PromptFile


class PostgreSQLBackend(DatabaseBackend):
    """PostgreSQL + pgvector database backend."""
    
    db_type = "postgresql"
    
    def __init__(self):
        from ..config import DATABASE_URL, ENABLE_WEIGHTS
        
        self.enable_weights = ENABLE_WEIGHTS
        
        # Validate and fix URL
        url = DATABASE_URL
        if not url.startswith("postgresql+asyncpg://"):
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            else:
                raise ValueError(
                    f"Invalid DATABASE_URL scheme: {url}. "
                    "Must start with 'postgresql+asyncpg://' or 'postgresql://'."
                )
        
        self.database_url = url
        self.engine = create_async_engine(url, echo=False)
        self.async_session = async_sessionmaker(self.engine, expire_on_commit=False)
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def _run_async(self, coro):
        """Run an async coroutine in a sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        
        if loop is not None:
            # We're in an async context, create a new loop in a thread
            future = self._executor.submit(asyncio.run, coro)
            return future.result()
        else:
            return asyncio.run(coro)
    
    def close(self):
        """Close the engine and executor."""
        self._run_async(self.engine.dispose())
        self._executor.shutdown(wait=False)
    
    # ==================== Init ====================
    
    def init_db(self, project_name: Optional[str] = None) -> None:
        """Initialize database tables."""
        async def _init():
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        self._run_async(_init())
    
    # ==================== Project Operations ====================
    
    def create_project(self, project_name: str, embedding_model: str, vector_dim: int) -> Dict[str, Any]:
        async def _create():
            async with self.async_session() as session:
                project = Project(
                    name=project_name,
                    embedding_model=embedding_model,
                    vector_dimension=vector_dim
                )
                session.add(project)
                await session.commit()
                await session.refresh(project)
                return {
                    "id": project.id,
                    "name": project.name,
                    "embedding_model": project.embedding_model,
                    "vector_dimension": project.vector_dimension
                }
        return self._run_async(_create())
    
    def get_project(self, project_name: str) -> Optional[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                result = await session.execute(
                    select(Project).where(Project.name == project_name)
                )
                project = result.scalar_one_or_none()
                if project:
                    return {
                        "id": project.id,
                        "name": project.name,
                        "embedding_model": project.embedding_model,
                        "vector_dimension": project.vector_dimension
                    }
                return None
        return self._run_async(_get())
    
    def list_projects(self) -> List[str]:
        async def _list():
            async with self.async_session() as session:
                result = await session.execute(select(Project.name))
                return list(result.scalars().all())
        return self._run_async(_list())
    
    def delete_project(self, project_name: str) -> bool:
        async def _delete():
            async with self.async_session() as session:
                result = await session.execute(
                    select(Project).where(Project.name == project_name)
                )
                project = result.scalar_one_or_none()
                if project:
                    await session.delete(project)
                    await session.commit()
                    return True
                return False
        return self._run_async(_delete())
    
    def get_project_metadata(self, project_name: str) -> Optional[Dict[str, Any]]:
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
        async def _add():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found.")
                
                qa_pair = QAPair(
                    project_id=project.id,
                    prompt_text=prompt,
                    response_text=response,
                    weight=weight,
                    embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                )
                session.add(qa_pair)
                await session.commit()
                await session.refresh(qa_pair)
                return qa_pair.id
        return self._run_async(_add())
    
    def get_qa_pairs(self, project_name: str, limit: int = 200) -> List[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return []
                
                result = await session.execute(
                    select(QAPair)
                    .where(QAPair.project_id == project.id)
                    .order_by(QAPair.id)
                    .limit(limit)
                )
                qa_pairs = result.scalars().all()
                
                return [
                    {
                        "id": qa.id,
                        "prompt_text": qa.prompt_text,
                        "response_text": qa.response_text,
                        "weight": qa.weight
                    }
                    for qa in qa_pairs
                ]
        return self._run_async(_get())
    
    def update_qa_pair(
        self, 
        project_name: str, 
        qa_id: int, 
        updates: Dict[str, Any]
    ) -> bool:
        async def _update():
            async with self.async_session() as session:
                result = await session.execute(
                    select(QAPair).where(QAPair.id == qa_id)
                )
                qa_pair = result.scalar_one_or_none()
                
                if not qa_pair:
                    return False
                
                if 'prompt' in updates:
                    qa_pair.prompt_text = updates['prompt']
                if 'response' in updates:
                    qa_pair.response_text = updates['response']
                if 'weight' in updates:
                    qa_pair.weight = updates['weight']
                if 'embedding' in updates:
                    embedding = updates['embedding']
                    qa_pair.embedding = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                
                await session.commit()
                return True
        return self._run_async(_update())
    
    def delete_qa_pair(self, project_name: str, qa_id: int) -> bool:
        async def _delete():
            async with self.async_session() as session:
                result = await session.execute(delete(QAPair).where(QAPair.id == qa_id))
                await session.commit()
                return result.rowcount > 0
        return self._run_async(_delete())
    
    def find_similar_prompts(
        self, 
        project_name: str, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        async def _find():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found.")
                
                # Use pgvector cosine distance
                distance_col = QAPair.embedding.cosine_distance(query_embedding).label("distance")
                
                stmt = select(QAPair, distance_col).where(QAPair.project_id == project.id)
                
                if self.enable_weights:
                    weighted_sim = (1 - distance_col) * QAPair.weight
                    stmt = stmt.order_by(weighted_sim.desc())
                else:
                    stmt = stmt.order_by(distance_col.asc())
                
                stmt = stmt.limit(top_k)
                
                result = await session.execute(stmt)
                rows = result.all()
                
                return [
                    {
                        "id": row[0].id,
                        "response_text": row[0].response_text,
                        "original_prompt": row[0].prompt_text,
                        "similarity_score": 1 - row[1],
                        "weight": row[0].weight,
                        "weighted_similarity": (1 - row[1]) * row[0].weight if self.enable_weights else None,
                    }
                    for row in rows
                ]
        return self._run_async(_find())
    
    def re_embed_prompts(
        self, 
        project_name: str, 
        ids: str | List[int] = "all",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        from .. import embedding as emb_module
        
        async def _re_embed():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found.")
                
                model_name = project.embedding_model
                
                stmt = select(QAPair).where(QAPair.project_id == project.id)
                if ids != "all":
                    stmt = stmt.where(QAPair.id.in_(ids))
                
                result = await session.execute(stmt)
                qa_pairs = result.scalars().all()
                total = len(qa_pairs)
                
                if total == 0:
                    return 0
                
                for i, qa_pair in enumerate(qa_pairs):
                    new_embedding = emb_module.generate_embedding(qa_pair.prompt_text, model_name)
                    qa_pair.embedding = new_embedding.tolist()
                    
                    if progress_callback:
                        progress_callback(i + 1, total)
                
                await session.commit()
                return total
        return self._run_async(_re_embed())
    
    # ==================== Scraped Content Operations ====================
    
    def add_scraped_content(self, project_name: str, scraped_data: List[Dict[str, Any]]) -> int:
        async def _add():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found.")
                
                for page_data in scraped_data:
                    content = ScrapedContent(
                        project_id=project.id,
                        url=page_data.get('url', ''),
                        title=page_data.get('title', ''),
                        content=page_data.get('content', ''),
                        domain=page_data.get('domain', '')
                    )
                    session.add(content)
                
                await session.commit()
                return len(scraped_data)
        return self._run_async(_add())
    
    def get_scraped_content(self, project_name: str) -> List[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return []
                
                result = await session.execute(
                    select(ScrapedContent).where(ScrapedContent.project_id == project.id)
                )
                rows = result.scalars().all()
                
                return [
                    {
                        "url": row.url,
                        "title": row.title,
                        "content": row.content,
                        "domain": row.domain
                    }
                    for row in rows
                ]
        return self._run_async(_get())
    
    # ==================== Prompt File Operations ====================
    
    def add_prompt_file(self, project_name: str, prompt_data: str, business_context: str) -> int:
        async def _add():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found.")
                
                prompt_file = PromptFile(
                    project_id=project.id,
                    prompt_data=prompt_data,
                    business_context=business_context
                )
                session.add(prompt_file)
                await session.commit()
                await session.refresh(prompt_file)
                return prompt_file.id
        return self._run_async(_add())
    
    def get_latest_prompt_file(self, project_name: str) -> Optional[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return None
                
                stmt = (
                    select(PromptFile)
                    .where(PromptFile.project_id == project.id)
                    .order_by(PromptFile.created_at.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                prompt_file = result.scalar_one_or_none()
                
                if prompt_file:
                    return {
                        "prompt_data": prompt_file.prompt_data,
                        "business_context": prompt_file.business_context
                    }
                return None
        return self._run_async(_get())
    
    # ==================== Weight Operations ====================
    
    def update_qa_weight(self, project_name: str, qa_id: int, new_weight: float) -> bool:
        if not self.enable_weights:
            return False
        if not (0.1 <= new_weight <= 10.0):
            raise ValueError("Weight must be between 0.1 and 10.0")
        return self.update_qa_pair(project_name, qa_id, weight=new_weight)
    
    def get_qa_pairs_with_weights(self, project_name: str) -> List[Dict[str, Any]]:
        return self.get_qa_pairs(project_name, limit=10000)
    
    def reset_all_weights(self, project_name: str) -> int:
        if not self.enable_weights:
            return 0
        
        async def _reset():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return 0
                
                result = await session.execute(
                    update(QAPair)
                    .where(QAPair.project_id == project.id)
                    .values(weight=1.0)
                )
                await session.commit()
                return result.rowcount
        return self._run_async(_reset())
    
    def get_weight_statistics(self, project_name: str) -> Optional[Dict[str, Any]]:
        if not self.enable_weights:
            return None
        
        async def _stats():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return None
                
                stmt = select(
                    func.count(QAPair.id),
                    func.avg(QAPair.weight),
                    func.min(QAPair.weight),
                    func.max(QAPair.weight),
                    func.count(QAPair.id).filter(QAPair.weight != 1.0)
                ).where(QAPair.project_id == project.id)
                
                result = await session.execute(stmt)
                row = result.fetchone()
                
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
        return self._run_async(_stats())
    
    # ==================== Utility Operations ====================
    
    def get_tables(self, project_name: Optional[str] = None) -> List[str]:
        return ["projects", "qa_pairs", "scraped_content", "prompt_files"]
    
    def get_table_data(
        self, 
        project_name: str, 
        table_name: str, 
        limit: int = 200
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        async def _get_data():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return [], []
                
                if table_name == "qa_pairs":
                    result = await session.execute(
                        select(QAPair)
                        .where(QAPair.project_id == project.id)
                        .limit(limit)
                    )
                    rows = result.scalars().all()
                    columns = ["id", "prompt_text", "response_text", "weight"]
                    data = [
                        {"id": r.id, "prompt_text": r.prompt_text, "response_text": r.response_text, "weight": r.weight}
                        for r in rows
                    ]
                elif table_name == "scraped_content":
                    result = await session.execute(
                        select(ScrapedContent)
                        .where(ScrapedContent.project_id == project.id)
                        .limit(limit)
                    )
                    rows = result.scalars().all()
                    columns = ["id", "url", "title", "content", "domain"]
                    data = [
                        {"id": r.id, "url": r.url, "title": r.title, "content": r.content, "domain": r.domain}
                        for r in rows
                    ]
                elif table_name == "prompt_files":
                    result = await session.execute(
                        select(PromptFile)
                        .where(PromptFile.project_id == project.id)
                        .limit(limit)
                    )
                    rows = result.scalars().all()
                    columns = ["id", "prompt_data", "business_context"]
                    data = [
                        {"id": r.id, "prompt_data": r.prompt_data, "business_context": r.business_context}
                        for r in rows
                    ]
                else:
                    return [], []
                
                return columns, data
        return self._run_async(_get_data())
    
    def execute_update(
        self, 
        project_name: str, 
        table_name: str, 
        pk_column: str,
        pk_value: Any,
        column: str, 
        new_value: Any
    ) -> bool:
        async def _update():
            async with self.async_session() as session:
                if table_name == "qa_pairs":
                    stmt = update(QAPair).where(QAPair.id == pk_value).values(**{column: new_value})
                elif table_name == "scraped_content":
                    stmt = update(ScrapedContent).where(ScrapedContent.id == pk_value).values(**{column: new_value})
                elif table_name == "prompt_files":
                    stmt = update(PromptFile).where(PromptFile.id == pk_value).values(**{column: new_value})
                else:
                    return False
                
                result = await session.execute(stmt)
                await session.commit()
                return result.rowcount > 0
        return self._run_async(_update())
    
    # ==================== Export/Import ====================
    
    def export_to_sqlite(self, project_name: str, output_path: str) -> bool:
        """Export a PostgreSQL project to a portable SQLite database."""
        import sqlite3
        import sqlite_vec
        from pathlib import Path
        
        project = self.get_project(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found.")
        
        # Create SQLite database
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(output_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        
        vector_dim = project["vector_dimension"]
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS qa_pairs USING vec0(
                prompt_embedding float[{vector_dim}]
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_text (
                id INTEGER PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                weight REAL DEFAULT 1.0
            )
        """)
        
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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prompt_files (
                id INTEGER PRIMARY KEY,
                prompt_data TEXT NOT NULL,
                business_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert metadata
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                      ("embedding_model", project["embedding_model"]))
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                      ("vector_dimension", str(vector_dim)))
        
        # Export QA pairs
        async def _export_qa():
            async with self.async_session() as session:
                proj = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one()
                
                result = await session.execute(
                    select(QAPair).where(QAPair.project_id == proj.id)
                )
                return result.scalars().all()
        
        qa_pairs = self._run_async(_export_qa())
        for qa in qa_pairs:
            cursor.execute(
                "INSERT INTO qa_text (prompt_text, response_text, weight) VALUES (?, ?, ?)",
                (qa.prompt_text, qa.response_text, qa.weight)
            )
            rowid = cursor.lastrowid
            
            # Convert embedding to bytes
            embedding = qa.embedding
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            embedding_bytes = embedding.astype(np.float32).tobytes()
            
            cursor.execute(
                "INSERT INTO qa_pairs (rowid, prompt_embedding) VALUES (?, ?)",
                (rowid, embedding_bytes)
            )
        
        # Export scraped content
        scraped = self.get_scraped_content(project_name)
        for item in scraped:
            cursor.execute(
                "INSERT INTO scraped_content (url, title, content, domain) VALUES (?, ?, ?, ?)",
                (item["url"], item["title"], item["content"], item["domain"])
            )
        
        # Export prompt files
        prompt_file = self.get_latest_prompt_file(project_name)
        if prompt_file:
            cursor.execute(
                "INSERT INTO prompt_files (prompt_data, business_context) VALUES (?, ?)",
                (prompt_file["prompt_data"], prompt_file["business_context"])
            )
        
        conn.commit()
        conn.close()
        return True
