"""
Abstract base class for database backends.
Defines the common interface for all database operations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
import numpy as np


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""
    
    db_type: str = "base"
    
    @abstractmethod
    def init_db(self, project_name: Optional[str] = None) -> None:
        """
        Initialize the database/tables.
        For PostgreSQL: Creates all tables in the central database.
        For SQLite: Creates the database file and tables for a specific project.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close any open connections."""
        pass
    
    # ==================== Project Operations ====================
    
    @abstractmethod
    def create_project(self, project_name: str, embedding_model: str, vector_dim: int) -> Dict[str, Any]:
        """
        Create a new project.
        
        Args:
            project_name: Unique name for the project
            embedding_model: Name/path of the embedding model to use
            vector_dim: Dimension of the embedding vectors
            
        Returns:
            Dict with project metadata
        """
        pass
    
    @abstractmethod
    def get_project(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get project metadata by name."""
        pass
    
    @abstractmethod
    def list_projects(self) -> List[str]:
        """List all project names."""
        pass
    
    @abstractmethod
    def delete_project(self, project_name: str) -> bool:
        """Delete a project and all its data."""
        pass
    
    @abstractmethod
    def get_project_metadata(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a project (embedding_model, vector_dimension, etc.)."""
        pass
    
    # ==================== QA Pair Operations ====================
    
    @abstractmethod
    def add_qa_pair(
        self, 
        project_name: str, 
        prompt: str, 
        response: str, 
        embedding: np.ndarray, 
        weight: float = 1.0
    ) -> int:
        """
        Add a QA pair to a project.
        
        Returns:
            The ID of the newly created QA pair.
        """
        pass
    
    @abstractmethod
    def get_qa_pairs(self, project_name: str, limit: int = 200) -> List[Dict[str, Any]]:
        """Get all QA pairs for a project (with optional limit)."""
        pass
    
    @abstractmethod
    def update_qa_pair(
        self, 
        project_name: str, 
        qa_id: int, 
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a QA pair with the provided field values.
        
        Args:
            project_name: Name of the project
            qa_id: ID of the QA pair to update
            updates: Dict of field names to new values (prompt, response, weight, embedding)
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def delete_qa_pair(self, project_name: str, qa_id: int) -> bool:
        """Delete a QA pair by ID."""
        pass
    
    @abstractmethod
    def find_similar_prompts(
        self, 
        project_name: str, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar prompts to a query embedding.
        
        Returns:
            List of dicts with id, response_text, original_prompt, similarity_score, weight, weighted_similarity
        """
        pass
    
    @abstractmethod
    def re_embed_prompts(
        self, 
        project_name: str, 
        ids: str | List[int] = "all",
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """
        Re-generate embeddings for prompts.
        
        Args:
            project_name: Name of the project
            ids: "all" or list of specific IDs to re-embed
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            Number of prompts re-embedded
        """
        pass
    
    # ==================== Scraped Content Operations ====================
    
    @abstractmethod
    def add_scraped_content(self, project_name: str, scraped_data: List[Dict[str, Any]]) -> int:
        """
        Add scraped content to a project.
        
        Args:
            project_name: Name of the project
            scraped_data: List of dicts with url, title, content, domain
            
        Returns:
            Number of items added
        """
        pass
    
    @abstractmethod
    def get_scraped_content(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all scraped content for a project."""
        pass
    
    # ==================== Prompt File Operations ====================
    
    @abstractmethod
    def add_prompt_file(self, project_name: str, prompt_data: str, business_context: str) -> int:
        """
        Add a prompt file to a project.
        
        Args:
            project_name: Name of the project
            prompt_data: JSON string of prompts
            business_context: Business context description
            
        Returns:
            ID of the created prompt file
        """
        pass
    
    @abstractmethod
    def get_latest_prompt_file(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest prompt file for a project."""
        pass
    
    # ==================== Weight Operations ====================
    
    @abstractmethod
    def update_qa_weight(self, project_name: str, qa_id: int, new_weight: float) -> bool:
        """Update the weight of a specific QA pair."""
        pass
    
    @abstractmethod
    def get_qa_pairs_with_weights(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all QA pairs with their weights."""
        pass
    
    @abstractmethod
    def reset_all_weights(self, project_name: str) -> int:
        """Reset all weights to 1.0. Returns number of rows updated."""
        pass
    
    @abstractmethod
    def get_weight_statistics(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics about weights in the project."""
        pass
    
    # ==================== Export/Import Operations ====================
    
    def export_to_sqlite(self, project_name: str, output_path: str) -> bool:
        """
        Export a project to a portable SQLite database file.
        Default implementation - can be overridden by subclasses.
        """
        raise NotImplementedError("Export not implemented for this backend")
    
    def import_from_sqlite(self, project_name: str, input_path: str) -> bool:
        """
        Import a project from a SQLite database file.
        Default implementation - can be overridden by subclasses.
        """
        raise NotImplementedError("Import not implemented for this backend")
    
    # ==================== Utility Operations ====================
    
    @abstractmethod
    def get_tables(self, project_name: Optional[str] = None) -> List[str]:
        """Get list of tables in the database."""
        pass
    
    @abstractmethod
    def get_table_data(
        self, 
        project_name: str, 
        table_name: str, 
        limit: int = 200
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Get data from a specific table.
        
        Returns:
            Tuple of (column_names, rows)
        """
        pass
    
    @abstractmethod
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
        pass
