import json
import numpy as np
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, delete, func, desc
from sqlalchemy.orm import selectinload
from .config import DATABASE_URL, ENABLE_WEIGHTS
from .models_db import Base, Project, QAPair, ScrapedContent, PromptFile
from . import embedding

# Validate Database URL for Async Engine
if not DATABASE_URL.startswith("postgresql+asyncpg://"):
    if DATABASE_URL.startswith("postgresql://"):
        # Auto-correct common mistake
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        raise ValueError(
            f"Invalid DATABASE_URL scheme: {DATABASE_URL}. "
            "For async SQLAlchemy, the URL must start with 'postgresql+asyncpg://'."
        )

# Create Async Engine
engine = create_async_engine(DATABASE_URL, echo=False)

# Create Async Session Factory
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def init_db():
    """
    Initializes the database tables.
    """
    async with engine.begin() as conn:
        # Create tables
        # Note: pgvector extension must be installed in the database manually or via migration
        # CREATE EXTENSION IF NOT EXISTS vector;
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    """
    Dependency generator for FastAPI.
    """
    async with AsyncSessionLocal() as session:
        yield session

async def create_project(session: AsyncSession, project_name: str, embedding_model: str, vector_dim: int):
    """
    Creates a new project in the database.
    """
    project = Project(name=project_name, embedding_model=embedding_model, vector_dimension=vector_dim)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return project

async def get_project_by_name(session: AsyncSession, project_name: str) -> Optional[Project]:
    result = await session.execute(select(Project).where(Project.name == project_name))
    return result.scalar_one_or_none()

async def list_projects(session: AsyncSession) -> List[str]:
    result = await session.execute(select(Project.name))
    return list(result.scalars().all())

async def add_qa_pair(session: AsyncSession, project_name: str, prompt: str, response: str, embedding_vector: np.ndarray, weight: float = 1.0):
    project = await get_project_by_name(session, project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found.")

    qa_pair = QAPair(
        project_id=project.id,
        prompt_text=prompt,
        response_text=response,
        weight=weight,
        embedding=embedding_vector
    )
    session.add(qa_pair)
    await session.commit()

async def find_similar_prompts(session: AsyncSession, project_name: str, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    project = await get_project_by_name(session, project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found.")

    # Calculate cosine distance
    # Note: pgvector's cosine_distance operator is <=>
    distance_col = QAPair.embedding.cosine_distance(query_embedding).label("distance")
    
    stmt = select(QAPair, distance_col).where(QAPair.project_id == project.id)

    if ENABLE_WEIGHTS:
        # weighted_similarity = (1 - distance) * weight
        # We order by this descending
        weighted_sim = (1 - distance_col) * QAPair.weight
        stmt = stmt.order_by(weighted_sim.desc())
    else:
        # Order by distance ascending (closest first)
        stmt = stmt.order_by(distance_col.asc())

    stmt = stmt.limit(top_k)
    
    result = await session.execute(stmt)
    rows = result.all()

    return [
        {
            "id": row[0].id,
            "response_text": row[0].response_text,
            "original_prompt": row[0].prompt_text,
            "similarity_score": 1 - row[1], # Convert distance back to similarity
            "weight": row[0].weight,
            "weighted_similarity": (1 - row[1]) * row[0].weight if ENABLE_WEIGHTS else None,
        }
        for row in rows
    ]

async def add_scraped_content(session: AsyncSession, project_name: str, scraped_data: list):
    project = await get_project_by_name(session, project_name)
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

async def get_scraped_content(session: AsyncSession, project_name: str) -> List[Dict[str, Any]]:
    project = await get_project_by_name(session, project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found.")

    result = await session.execute(select(ScrapedContent).where(ScrapedContent.project_id == project.id))
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

async def add_prompt_file(session: AsyncSession, project_name: str, prompt_data: str, business_context: str):
    project = await get_project_by_name(session, project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found.")

    prompt_file = PromptFile(
        project_id=project.id,
        prompt_data=prompt_data,
        business_context=business_context
    )
    session.add(prompt_file)
    await session.commit()
    return prompt_file.id

async def get_latest_prompt_file(session: AsyncSession, project_name: str) -> Optional[Dict[str, Any]]:
    project = await get_project_by_name(session, project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found.")

    stmt = select(PromptFile).where(PromptFile.project_id == project.id).order_by(PromptFile.created_at.desc()).limit(1)
    result = await session.execute(stmt)
    prompt_file = result.scalar_one_or_none()

    if prompt_file:
        return {
            "prompt_data": prompt_file.prompt_data,
            "business_context": prompt_file.business_context
        }
    return None

async def get_project_metadata(session: AsyncSession, project_name: str) -> Optional[Dict[str, Any]]:
    project = await get_project_by_name(session, project_name)
    if not project:
        return None
    
    return {
        "embedding_model": project.embedding_model,
        "vector_dimension": project.vector_dimension
    }

async def re_embed_prompts(session: AsyncSession, project_name: str, ids: str | list[int] = "all", progress_callback=None):
    project = await get_project_by_name(session, project_name)
    if not project:
        raise ValueError(f"Project '{project_name}' not found.")
    
    model_name = project.embedding_model

    stmt = select(QAPair).where(QAPair.project_id == project.id)
    if ids != "all":
        stmt = stmt.where(QAPair.id.in_(ids))
    
    result = await session.execute(stmt)
    qa_pairs = result.scalars().all()
    total_prompts = len(qa_pairs)

    if total_prompts == 0:
        return 0

    for i, qa_pair in enumerate(qa_pairs):
        new_embedding = embedding.generate_embedding(qa_pair.prompt_text, model_name)
        qa_pair.embedding = new_embedding
        
        if progress_callback:
            progress_callback(i + 1, total_prompts)
    
    await session.commit()
    return total_prompts
