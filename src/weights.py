"""
Weight management module for QA pairs.
Handles weighting functionality when the feature is enabled.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Any
from .config import ENABLE_WEIGHTS
from .database import get_db_connection, get_db_path


def is_weights_enabled() -> bool:
    """Check if weighting feature is enabled."""
    return ENABLE_WEIGHTS


def update_qa_weight(project_name: str, qa_id: int, new_weight: float, projects_dir: Path) -> bool:
    """
    Update the weight of a specific QA pair.
    Only works if weights are enabled.
    
    Args:
        project_name: Name of the project
        qa_id: ID of the QA pair to update
        new_weight: New weight value (0.1 to 10.0)
        projects_dir: Directory containing projects
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    if not ENABLE_WEIGHTS:
        return False
        
    if not (0.1 <= new_weight <= 10.0):
        raise ValueError("Weight must be between 0.1 and 10.0")
    
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE qa_text SET weight = ? WHERE id = ?", (new_weight, qa_id))
        conn.commit()
        return cursor.rowcount > 0


def get_qa_pairs_with_weights(project_name: str, projects_dir: Path) -> List[Dict[str, Any]]:
    """
    Get all QA pairs with their weights for management.
    Returns weight information only if weights are enabled.
    
    Args:
        project_name: Name of the project
        projects_dir: Directory containing projects
        
    Returns:
        List of QA pairs with weight information
    """
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        if ENABLE_WEIGHTS:
            cursor.execute("SELECT id, prompt_text, response_text, weight FROM qa_text ORDER BY id")
            results = cursor.fetchall()
            
            return [
                {
                    "id": row[0],
                    "prompt_text": row[1],
                    "response_text": row[2],
                    "weight": row[3]
                }
                for row in results
            ]
        else:
            cursor.execute("SELECT id, prompt_text, response_text FROM qa_text ORDER BY id")
            results = cursor.fetchall()
            
            return [
                {
                    "id": row[0],
                    "prompt_text": row[1],
                    "response_text": row[2],
                    "weight": 1.0  # Default weight when disabled
                }
                for row in results
            ]


def set_qa_weights_bulk(project_name: str, weight_updates: List[Dict[str, Any]], projects_dir: Path) -> int:
    """
    Update multiple QA pair weights in bulk.
    Only works if weights are enabled.
    
    Args:
        project_name: Name of the project
        weight_updates: List of dicts with 'id' and 'weight' keys
        projects_dir: Directory containing projects
        
    Returns:
        int: Number of records updated
    """
    if not ENABLE_WEIGHTS:
        return 0
    
    db_path = get_db_path(project_name, projects_dir)
    updated_count = 0
    
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        for update in weight_updates:
            qa_id = update.get('id')
            weight = update.get('weight')
            
            if qa_id is None or weight is None:
                continue
                
            if not (0.1 <= weight <= 10.0):
                continue
                
            cursor.execute("UPDATE qa_text SET weight = ? WHERE id = ?", (weight, qa_id))
            if cursor.rowcount > 0:
                updated_count += 1
        
        conn.commit()
    
    return updated_count


def reset_all_weights(project_name: str, projects_dir: Path) -> int:
    """
    Reset all weights to default value of 1.0.
    Only works if weights are enabled.
    
    Args:
        project_name: Name of the project
        projects_dir: Directory containing projects
        
    Returns:
        int: Number of records updated
    """
    if not ENABLE_WEIGHTS:
        return 0
    
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE qa_text SET weight = 1.0")
        conn.commit()
        return cursor.rowcount


def get_weight_statistics(project_name: str, projects_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Get statistics about weights in the project.
    Returns None if weights are disabled.
    
    Args:
        project_name: Name of the project
        projects_dir: Directory containing projects
        
    Returns:
        Dict with weight statistics or None if disabled
    """
    if not ENABLE_WEIGHTS:
        return None
    
    db_path = get_db_path(project_name, projects_dir)
    with get_db_connection(db_path) as conn:
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
        result = cursor.fetchone()
        
        if result:
            return {
                "total_pairs": result[0],
                "avg_weight": round(result[1], 3) if result[1] else 1.0,
                "min_weight": result[2] if result[2] else 1.0,
                "max_weight": result[3] if result[3] else 1.0,
                "modified_weights": result[4],
                "weights_enabled": True
            }
    
    return None