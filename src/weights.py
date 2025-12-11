"""
Weight management module for QA pairs.
Handles weighting functionality when the feature is enabled.
"""

from typing import List, Dict, Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func
from .config import ENABLE_WEIGHTS
from .models_db import QAPair, Project
from .database import get_project_by_name

def is_weights_enabled() -> bool:
    """Check if weighting feature is enabled."""
    return ENABLE_WEIGHTS

async def update_qa_weight(session: AsyncSession, project_name: str, qa_id: int, new_weight: float) -> bool:
    """
    Update the weight of a specific QA pair.
    Only works if weights are enabled.
    """
    if not ENABLE_WEIGHTS:
        return False
        
    if not (0.1 <= new_weight <= 10.0):
        raise ValueError("Weight must be between 0.1 and 10.0")
    
    project = await get_project_by_name(session, project_name)
    if not project:
        return False

    stmt = update(QAPair).where(QAPair.id == qa_id, QAPair.project_id == project.id).values(weight=new_weight)
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount > 0

async def get_qa_pairs_with_weights(session: AsyncSession, project_name: str) -> List[Dict[str, Any]]:
    """
    Get all QA pairs with their weights for management.
    """
    project = await get_project_by_name(session, project_name)
    if not project:
        return []

    stmt = select(QAPair).where(QAPair.project_id == project.id).order_by(QAPair.id)
    result = await session.execute(stmt)
    qa_pairs = result.scalars().all()
    
    return [
        {
            "id": qa.id,
            "prompt_text": qa.prompt_text,
            "response_text": qa.response_text,
            "weight": qa.weight if ENABLE_WEIGHTS else 1.0
        }
        for qa in qa_pairs
    ]

async def set_qa_weights_bulk(session: AsyncSession, project_name: str, weight_updates: List[Dict[str, Any]]) -> int:
    """
    Update multiple QA pair weights in bulk.
    """
    if not ENABLE_WEIGHTS:
        return 0
    
    project = await get_project_by_name(session, project_name)
    if not project:
        return 0
    
    updated_count = 0
    
    for update_item in weight_updates:
        qa_id = update_item.get('id')
        weight = update_item.get('weight')
        
        if qa_id is None or weight is None:
            continue
            
        if not (0.1 <= weight <= 10.0):
            continue
            
        stmt = update(QAPair).where(QAPair.id == qa_id, QAPair.project_id == project.id).values(weight=weight)
        result = await session.execute(stmt)
        if result.rowcount > 0:
            updated_count += 1
    
    await session.commit()
    return updated_count

async def reset_all_weights(session: AsyncSession, project_name: str) -> int:
    """
    Reset all weights to default value of 1.0.
    """
    if not ENABLE_WEIGHTS:
        return 0
    
    project = await get_project_by_name(session, project_name)
    if not project:
        return 0

    stmt = update(QAPair).where(QAPair.project_id == project.id).values(weight=1.0)
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount

async def get_weight_statistics(session: AsyncSession, project_name: str) -> Optional[Dict[str, Any]]:
    """
    Get statistics about weights in the project.
    """
    if not ENABLE_WEIGHTS:
        return None
    
    project = await get_project_by_name(session, project_name)
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
