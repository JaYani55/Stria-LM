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
    
    # ==================== Actor Operations ====================
    
    @abstractmethod
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
        """
        Create a new LLM actor for a project.
        
        Args:
            project_name: Name of the project
            actor_name: Unique name for this actor
            description: Description of the actor's purpose
            prompt_messages: List of system/context messages
            model_name: LLM model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            other_generation_parameters: Additional generation params
            
        Returns:
            UUID string of the created actor
        """
        pass
    
    @abstractmethod
    def get_actor(self, project_name: str, actor_id: str) -> Optional[Dict[str, Any]]:
        """Get an actor by ID."""
        pass
    
    @abstractmethod
    def get_actor_by_name(self, project_name: str, actor_name: str) -> Optional[Dict[str, Any]]:
        """Get an actor by name."""
        pass
    
    @abstractmethod
    def list_actors(self, project_name: str) -> List[Dict[str, Any]]:
        """List all actors for a project."""
        pass
    
    @abstractmethod
    def update_actor(
        self, 
        project_name: str, 
        actor_id: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update an actor's properties."""
        pass
    
    @abstractmethod
    def delete_actor(self, project_name: str, actor_id: str) -> bool:
        """Delete an actor."""
        pass
    
    # ==================== Persona Operations ====================
    
    @abstractmethod
    def create_persona(
        self, 
        project_name: str,
        persona_name: str,
        display_name: str,
        is_ai: bool = False,
        fallback_actor_id: Optional[str] = None
    ) -> str:
        """
        Create a persona for chat sessions.
        
        Args:
            project_name: Name of the project
            persona_name: Unique identifier for this persona
            display_name: Display name shown in chat
            is_ai: Whether this persona represents an AI
            fallback_actor_id: Optional actor ID for AI personas
            
        Returns:
            UUID string of the created persona
        """
        pass
    
    @abstractmethod
    def get_persona(self, project_name: str, persona_id: str) -> Optional[Dict[str, Any]]:
        """Get a persona by ID."""
        pass
    
    @abstractmethod
    def get_persona_by_name(self, project_name: str, persona_name: str) -> Optional[Dict[str, Any]]:
        """Get a persona by name."""
        pass
    
    @abstractmethod
    def list_personas(self, project_name: str) -> List[Dict[str, Any]]:
        """List all personas for a project."""
        pass
    
    @abstractmethod
    def delete_persona(self, project_name: str, persona_id: str) -> bool:
        """Delete a persona."""
        pass
    
    # ==================== Chat Session Operations ====================
    
    @abstractmethod
    def create_chat_session(
        self,
        project_name: str,
        actor_id: str,
        persona_id: str,
        session_name: Optional[str] = None
    ) -> str:
        """
        Create a new chat session.
        
        Args:
            project_name: Name of the project
            actor_id: UUID of the actor for this session
            persona_id: UUID of the persona (user)
            session_name: Optional name for the session
            
        Returns:
            UUID string of the created session
        """
        pass
    
    @abstractmethod
    def get_chat_session(self, project_name: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a chat session by ID."""
        pass
    
    @abstractmethod
    def list_chat_sessions(
        self, 
        project_name: str,
        persona_id: Optional[str] = None,
        actor_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List chat sessions, optionally filtered by persona or actor."""
        pass
    
    @abstractmethod
    def update_session_tokens(
        self,
        project_name: str,
        session_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> bool:
        """Update token counts for a session."""
        pass
    
    @abstractmethod
    def delete_chat_session(self, project_name: str, session_id: str) -> bool:
        """Delete a chat session and all its messages."""
        pass
    
    # ==================== Chat Message Operations ====================
    
    @abstractmethod
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
        """
        Add a message to a chat session.
        
        Args:
            project_name: Name of the project
            session_id: UUID of the chat session
            role: Message role (user, assistant, system)
            content: Message content
            token_count: Number of tokens in the message
            context_used: Metadata about context retrieved for this message
            generation_metadata: LLM generation metadata (tokens, latency, etc.)
            
        Returns:
            ID of the created message
        """
        pass
    
    @abstractmethod
    def get_chat_history(
        self,
        project_name: str,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a session.
        
        Args:
            project_name: Name of the project
            session_id: UUID of the chat session
            limit: Optional limit on number of messages (most recent)
            
        Returns:
            List of messages in chronological order
        """
        pass
    
    @abstractmethod
    def get_chat_context_window(
        self,
        project_name: str,
        session_id: str,
        max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Get chat history that fits within a token budget.
        Returns most recent messages that fit within max_tokens.
        """
        pass

    # ==================== Script Operations ====================
    
    @abstractmethod
    def register_script(
        self,
        project_name: str,
        script_name: str,
        script_type: str,
        file_path: str,
        description: str = "",
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Register a new script for a project.
        
        Args:
            project_name: Name of the project
            script_name: Display name for the script
            script_type: Type of script (scraper, data-manipulation, ai-script, migration)
            file_path: Relative path to script file within project
            description: Description of what the script does
            version: Version string (default: 1.0.0)
            metadata: Additional metadata (dependencies, schedule, etc.)
            
        Returns:
            ID of the registered script
        """
        pass
    
    @abstractmethod
    def get_script(self, project_name: str, script_id: int) -> Optional[Dict[str, Any]]:
        """Get a script by ID."""
        pass
    
    @abstractmethod
    def list_scripts(
        self,
        project_name: str,
        script_type: Optional[str] = None,
        enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List scripts for a project.
        
        Args:
            project_name: Name of the project
            script_type: Optional filter by script type
            enabled_only: Only return enabled scripts (default: True)
        """
        pass
    
    @abstractmethod
    def update_script(
        self,
        project_name: str,
        script_id: int,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a script's properties."""
        pass
    
    @abstractmethod
    def delete_script(self, project_name: str, script_id: int) -> bool:
        """Delete a script registration."""
        pass
    
    @abstractmethod
    def log_script_execution(
        self,
        project_name: str,
        script_id: int,
        status: str,
        exit_code: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log a script execution.
        
        Args:
            project_name: Name of the project
            script_id: ID of the script
            status: Execution status (running, success, failed, cancelled)
            exit_code: Process exit code (if finished)
            stdout: Standard output capture
            stderr: Standard error capture
            metadata: Additional execution metadata
            
        Returns:
            ID of the execution log entry
        """
        pass
    
    @abstractmethod
    def update_script_execution(
        self,
        project_name: str,
        execution_id: int,
        status: str,
        exit_code: Optional[int] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None
    ) -> bool:
        """Update an existing execution log entry."""
        pass
    
    @abstractmethod
    def get_script_executions(
        self,
        project_name: str,
        script_id: Optional[int] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get execution logs for scripts.
        
        Args:
            project_name: Name of the project
            script_id: Optional filter by script ID
            limit: Maximum number of entries to return
        """
        pass
    
    # ==================== Migration Operations ====================
    
    @abstractmethod
    def get_applied_migrations(self, project_name: str) -> List[Dict[str, Any]]:
        """Get list of applied migration versions."""
        pass
    
    @abstractmethod
    def record_migration(
        self,
        project_name: str,
        version: str,
        script_name: str,
        checksum: Optional[str] = None
    ) -> int:
        """
        Record that a migration has been applied.
        
        Args:
            project_name: Name of the project
            version: Migration version string
            script_name: Name of the migration script
            checksum: SHA-256 hash of script content
            
        Returns:
            ID of the recorded migration
        """
        pass
    
    @abstractmethod
    def get_scripts_directory(self, project_name: str) -> Optional[str]:
        """Get the absolute path to the project's scripts directory."""
        pass
