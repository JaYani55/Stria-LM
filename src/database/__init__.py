"""
Database abstraction layer for Stria-LM.
Supports both PostgreSQL (remote) and SQLite-vec (local) backends.
"""

from typing import Literal, Optional
from .base import DatabaseBackend
from .postgresql import PostgreSQLBackend
from .sqlite import SQLiteBackend

DatabaseType = Literal["postgresql", "sqlite"]

_current_backend: Optional[DatabaseBackend] = None


def get_database(db_type: Optional[DatabaseType] = None) -> DatabaseBackend:
    """
    Factory function to get the appropriate database backend.
    
    Args:
        db_type: Either "postgresql" or "sqlite". If None, reads from config.
        
    Returns:
        DatabaseBackend instance for the specified type.
    """
    global _current_backend
    
    if db_type is None:
        from ..config import DATABASE_TYPE
        db_type = DATABASE_TYPE
    
    # Return cached backend if same type
    if _current_backend is not None and _current_backend.db_type == db_type:
        return _current_backend
    
    if db_type == "postgresql":
        _current_backend = PostgreSQLBackend()
    elif db_type == "sqlite":
        _current_backend = SQLiteBackend()
    else:
        raise ValueError(f"Unknown database type: {db_type}. Must be 'postgresql' or 'sqlite'.")
    
    return _current_backend


def reset_backend():
    """Reset the cached backend (useful for switching database types)."""
    global _current_backend
    if _current_backend is not None:
        _current_backend.close()
    _current_backend = None


__all__ = [
    "DatabaseBackend",
    "PostgreSQLBackend",
    "SQLiteBackend",
    "get_database",
    "reset_backend",
    "DatabaseType",
]
