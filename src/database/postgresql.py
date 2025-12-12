"""
PostgreSQL + pgvector database backend.
Uses async SQLAlchemy with asyncpg driver.
"""

import json
import asyncio
import uuid as uuid_module
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, update, delete, func, text
from sqlalchemy.exc import OperationalError

from .base import DatabaseBackend
from ..models_db import Base, Project, QAPair, ScrapedContent, PromptFile, Actor, Persona, ChatSession, ChatMessage


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

    # ==================== Actor Operations ====================
    
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
        async def _create():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found")
                
                actor = Actor(
                    project_id=project.id,
                    actor_name=actor_name,
                    description=description,
                    prompt_messages=prompt_messages or [],
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    other_generation_parameters=other_generation_parameters
                )
                session.add(actor)
                await session.commit()
                await session.refresh(actor)
                return str(actor.actor_id)
        return self._run_async(_create())
    
    def get_actor(self, project_name: str, actor_id: str) -> Optional[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return None
                
                result = await session.execute(
                    select(Actor).where(
                        Actor.actor_id == int(actor_id),
                        Actor.project_id == project.id
                    )
                )
                actor = result.scalar_one_or_none()
                
                if actor:
                    return {
                        "actor_id": str(actor.actor_id),
                        "actor_name": actor.actor_name,
                        "description": actor.description,
                        "prompt_messages": actor.prompt_messages,
                        "model_name": actor.model_name,
                        "temperature": actor.temperature,
                        "max_tokens": actor.max_tokens,
                        "top_p": actor.top_p,
                        "top_k": actor.top_k,
                        "repetition_penalty": actor.repetition_penalty,
                        "other_generation_parameters": actor.other_generation_parameters,
                        "created_at": actor.created_at.isoformat() if actor.created_at else None,
                        "updated_at": actor.updated_at.isoformat() if actor.updated_at else None
                    }
                return None
        return self._run_async(_get())
    
    def get_actor_by_name(self, project_name: str, actor_name: str) -> Optional[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return None
                
                result = await session.execute(
                    select(Actor).where(
                        Actor.actor_name == actor_name,
                        Actor.project_id == project.id
                    )
                )
                actor = result.scalar_one_or_none()
                
                if actor:
                    return {
                        "actor_id": str(actor.actor_id),
                        "actor_name": actor.actor_name,
                        "description": actor.description,
                        "prompt_messages": actor.prompt_messages,
                        "model_name": actor.model_name,
                        "temperature": actor.temperature,
                        "max_tokens": actor.max_tokens,
                        "top_p": actor.top_p,
                        "top_k": actor.top_k,
                        "repetition_penalty": actor.repetition_penalty,
                        "other_generation_parameters": actor.other_generation_parameters,
                        "created_at": actor.created_at.isoformat() if actor.created_at else None,
                        "updated_at": actor.updated_at.isoformat() if actor.updated_at else None
                    }
                return None
        return self._run_async(_get())
    
    def list_actors(self, project_name: str) -> List[Dict[str, Any]]:
        async def _list():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return []
                
                result = await session.execute(
                    select(Actor).where(Actor.project_id == project.id)
                )
                actors = result.scalars().all()
                
                return [
                    {
                        "actor_id": str(a.actor_id),
                        "actor_name": a.actor_name,
                        "description": a.description,
                        "model_name": a.model_name,
                        "temperature": a.temperature,
                        "max_tokens": a.max_tokens,
                        "created_at": a.created_at.isoformat() if a.created_at else None
                    }
                    for a in actors
                ]
        return self._run_async(_list())
    
    def update_actor(self, project_name: str, actor_id: str, updates: Dict[str, Any]) -> bool:
        async def _update():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return False
                
                # Filter valid columns
                valid_columns = {
                    'actor_name', 'description', 'prompt_messages', 'model_name',
                    'temperature', 'max_tokens', 'top_p', 'top_k', 
                    'repetition_penalty', 'other_generation_parameters'
                }
                filtered_updates = {k: v for k, v in updates.items() if k in valid_columns}
                
                if not filtered_updates:
                    return False
                
                result = await session.execute(
                    update(Actor)
                    .where(Actor.actor_id == int(actor_id), Actor.project_id == project.id)
                    .values(**filtered_updates)
                )
                await session.commit()
                return result.rowcount > 0
        return self._run_async(_update())
    
    def delete_actor(self, project_name: str, actor_id: str) -> bool:
        async def _delete():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return False
                
                result = await session.execute(
                    delete(Actor).where(
                        Actor.actor_id == int(actor_id),
                        Actor.project_id == project.id
                    )
                )
                await session.commit()
                return result.rowcount > 0
        return self._run_async(_delete())
    
    # ==================== Persona Operations ====================
    
    def create_persona(
        self, 
        project_name: str,
        persona_name: str,
        display_name: str,
        is_ai: bool = False,
        fallback_actor_id: Optional[str] = None
    ) -> str:
        async def _create():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found")
                
                persona = Persona(
                    persona_id=str(uuid_module.uuid4()),
                    project_id=project.id,
                    persona_name=persona_name,
                    display_name=display_name,
                    is_ai=is_ai,
                    fallback_actor_id=int(fallback_actor_id) if fallback_actor_id else None
                )
                session.add(persona)
                await session.commit()
                await session.refresh(persona)
                return persona.persona_id
        return self._run_async(_create())
    
    def get_persona(self, project_name: str, persona_id: str) -> Optional[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return None
                
                result = await session.execute(
                    select(Persona).where(
                        Persona.persona_id == persona_id,
                        Persona.project_id == project.id
                    )
                )
                persona = result.scalar_one_or_none()
                
                if persona:
                    return {
                        "persona_id": persona.persona_id,
                        "persona_name": persona.persona_name,
                        "display_name": persona.display_name,
                        "description": persona.description,
                        "avatar_url": persona.avatar_url,
                        "is_ai": persona.is_ai,
                        "fallback_actor_id": str(persona.fallback_actor_id) if persona.fallback_actor_id else None,
                        "extra_data": persona.extra_data,
                        "created_at": persona.created_at.isoformat() if persona.created_at else None
                    }
                return None
        return self._run_async(_get())
    
    def get_persona_by_name(self, project_name: str, persona_name: str) -> Optional[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return None
                
                result = await session.execute(
                    select(Persona).where(
                        Persona.persona_name == persona_name,
                        Persona.project_id == project.id
                    )
                )
                persona = result.scalar_one_or_none()
                
                if persona:
                    return {
                        "persona_id": persona.persona_id,
                        "persona_name": persona.persona_name,
                        "display_name": persona.display_name,
                        "description": persona.description,
                        "avatar_url": persona.avatar_url,
                        "is_ai": persona.is_ai,
                        "fallback_actor_id": str(persona.fallback_actor_id) if persona.fallback_actor_id else None,
                        "extra_data": persona.extra_data,
                        "created_at": persona.created_at.isoformat() if persona.created_at else None
                    }
                return None
        return self._run_async(_get())
    
    def list_personas(self, project_name: str) -> List[Dict[str, Any]]:
        async def _list():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return []
                
                result = await session.execute(
                    select(Persona).where(Persona.project_id == project.id)
                )
                personas = result.scalars().all()
                
                return [
                    {
                        "persona_id": p.persona_id,
                        "persona_name": p.persona_name,
                        "display_name": p.display_name,
                        "is_ai": p.is_ai,
                        "created_at": p.created_at.isoformat() if p.created_at else None
                    }
                    for p in personas
                ]
        return self._run_async(_list())
    
    def delete_persona(self, project_name: str, persona_id: str) -> bool:
        async def _delete():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return False
                
                result = await session.execute(
                    delete(Persona).where(
                        Persona.persona_id == persona_id,
                        Persona.project_id == project.id
                    )
                )
                await session.commit()
                return result.rowcount > 0
        return self._run_async(_delete())
    
    # ==================== Chat Session Operations ====================
    
    def create_chat_session(
        self,
        project_name: str,
        actor_id: str,
        persona_id: str,
        session_name: Optional[str] = None
    ) -> str:
        async def _create():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    raise ValueError(f"Project '{project_name}' not found")
                
                chat_session = ChatSession(
                    session_id=str(uuid_module.uuid4()),
                    project_id=project.id,
                    actor_id=int(actor_id),
                    persona_id=persona_id,
                    title=session_name
                )
                session.add(chat_session)
                await session.commit()
                await session.refresh(chat_session)
                return chat_session.session_id
        return self._run_async(_create())
    
    def get_chat_session(self, project_name: str, session_id: str) -> Optional[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return None
                
                result = await session.execute(
                    select(ChatSession).where(
                        ChatSession.session_id == session_id,
                        ChatSession.project_id == project.id
                    )
                )
                chat_session = result.scalar_one_or_none()
                
                if chat_session:
                    return {
                        "session_id": chat_session.session_id,
                        "actor_id": str(chat_session.actor_id),
                        "persona_id": chat_session.persona_id,
                        "title": chat_session.title,
                        "total_input_tokens": chat_session.total_input_tokens,
                        "total_output_tokens": chat_session.total_output_tokens,
                        "is_active": chat_session.is_active,
                        "created_at": chat_session.created_at.isoformat() if chat_session.created_at else None,
                        "updated_at": chat_session.updated_at.isoformat() if chat_session.updated_at else None
                    }
                return None
        return self._run_async(_get())
    
    def list_chat_sessions(
        self, 
        project_name: str,
        persona_id: Optional[str] = None,
        actor_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        async def _list():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return []
                
                stmt = select(ChatSession).where(ChatSession.project_id == project.id)
                
                if persona_id:
                    stmt = stmt.where(ChatSession.persona_id == persona_id)
                if actor_id:
                    stmt = stmt.where(ChatSession.actor_id == int(actor_id))
                
                result = await session.execute(stmt.order_by(ChatSession.updated_at.desc()))
                sessions = result.scalars().all()
                
                return [
                    {
                        "session_id": s.session_id,
                        "actor_id": str(s.actor_id),
                        "persona_id": s.persona_id,
                        "title": s.title,
                        "total_input_tokens": s.total_input_tokens,
                        "total_output_tokens": s.total_output_tokens,
                        "is_active": s.is_active,
                        "created_at": s.created_at.isoformat() if s.created_at else None,
                        "updated_at": s.updated_at.isoformat() if s.updated_at else None
                    }
                    for s in sessions
                ]
        return self._run_async(_list())
    
    def update_session_tokens(
        self,
        project_name: str,
        session_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> bool:
        async def _update():
            async with self.async_session() as session:
                # Get current values and add
                result = await session.execute(
                    select(ChatSession).where(ChatSession.session_id == session_id)
                )
                chat_session = result.scalar_one_or_none()
                
                if not chat_session:
                    return False
                
                chat_session.total_input_tokens += input_tokens
                chat_session.total_output_tokens += output_tokens
                await session.commit()
                return True
        return self._run_async(_update())
    
    def delete_chat_session(self, project_name: str, session_id: str) -> bool:
        async def _delete():
            async with self.async_session() as session:
                project = (await session.execute(
                    select(Project).where(Project.name == project_name)
                )).scalar_one_or_none()
                
                if not project:
                    return False
                
                result = await session.execute(
                    delete(ChatSession).where(
                        ChatSession.session_id == session_id,
                        ChatSession.project_id == project.id
                    )
                )
                await session.commit()
                return result.rowcount > 0
        return self._run_async(_delete())
    
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
        async def _add():
            async with self.async_session() as session:
                message = ChatMessage(
                    message_id=str(uuid_module.uuid4()),
                    session_id=session_id,
                    role=role,
                    content=content,
                    token_count=token_count,
                    context_metadata=context_used,
                    generation_metadata=generation_metadata
                )
                session.add(message)
                await session.commit()
                await session.refresh(message)
                return message.message_id
        return self._run_async(_add())
    
    def get_chat_history(
        self,
        project_name: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        async def _get():
            async with self.async_session() as session:
                stmt = (
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at)
                )
                
                if limit:
                    # Get last N messages
                    stmt = (
                        select(ChatMessage)
                        .where(ChatMessage.session_id == session_id)
                        .order_by(ChatMessage.created_at.desc())
                        .limit(limit)
                    )
                
                result = await session.execute(stmt)
                messages = result.scalars().all()
                
                # If limited, reverse to get chronological order
                if limit:
                    messages = list(reversed(messages))
                
                return [
                    {
                        "message_id": m.message_id,
                        "role": m.role,
                        "content": m.content,
                        "token_count": m.token_count,
                        "context_metadata": m.context_metadata,
                        "generation_metadata": m.generation_metadata,
                        "created_at": m.created_at.isoformat() if m.created_at else None
                    }
                    for m in messages
                ]
        return self._run_async(_get())
    
    def get_chat_context_window(
        self,
        project_name: str,
        session_id: str,
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """Get most recent messages that fit within token budget."""
        async def _get():
            async with self.async_session() as session:
                # Get messages in reverse order
                stmt = (
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id)
                    .order_by(ChatMessage.created_at.desc())
                )
                
                result = await session.execute(stmt)
                messages = result.scalars().all()
                
                # Accumulate until token limit
                selected = []
                token_total = 0
                
                for m in messages:
                    if token_total + m.token_count <= max_tokens:
                        selected.append(m)
                        token_total += m.token_count
                    else:
                        break
                
                # Reverse to chronological order
                selected = list(reversed(selected))
                
                return [
                    {
                        "message_id": m.message_id,
                        "role": m.role,
                        "content": m.content,
                        "token_count": m.token_count
                    }
                    for m in selected
                ]
        return self._run_async(_get())