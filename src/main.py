from fastapi import FastAPI, HTTPException, Body
from typing import List
import os
import sys
import json

from . import models
from . import embedding
from . import config
from .database import get_database, DatabaseBackend

app = FastAPI(
    title="Stria-LM",
    description="A framework for portable, file-based retrieval models.",
    version="0.2.0",
)

# Global database backend instance
db: DatabaseBackend = None


def get_db() -> DatabaseBackend:
    """Get the database backend instance."""
    global db
    if db is None:
        db = get_database()
    return db


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global db
    print("Stria-LM server started.")
    print(f"Database type: {config.DATABASE_TYPE}")
    
    try:
        db = get_database()
        db.init_db()
        print(f"Database initialized successfully ({config.DATABASE_TYPE})")
    except OSError as e:
        print("\n" + "="*60)
        print("CRITICAL ERROR: Could not connect to the database.")
        
        if config.DATABASE_TYPE == "postgresql":
            print(f"Connection URL: {config.DATABASE_URL}")
            print("Please ensure PostgreSQL is running and the configuration is correct.")
            
            if sys.platform == "win32":
                print("\n[Windows] PostgreSQL setup:")
                print("1. Download from: https://www.postgresql.org/download/windows/")
                print("2. Start service in 'Services.msc'")
                print("3. Create database: CREATE DATABASE strialm;")
                print("4. Enable extension: CREATE EXTENSION vector;")
            elif sys.platform.startswith("linux"):
                print("\n[Linux] PostgreSQL setup:")
                print("1. Install: sudo apt install postgresql postgresql-contrib")
                print("2. Start: sudo systemctl start postgresql")
                print("3. Create database and extension as shown above")
        else:
            print("SQLite initialization failed.")
            print(f"Projects directory: {config.PROJECTS_DIR}")
        
        print(f"\nDetails: {e}")
        print("="*60 + "\n")
        raise e
    except Exception as e:
        print(f"Database initialization error: {e}")
        raise e


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global db
    if db is not None:
        db.close()


# =============================================================================
# PROJECT ENDPOINTS
# =============================================================================

@app.post("/projects", status_code=201)
def create_project(project: models.ProjectCreate):
    """
    Creates a new project, which corresponds to a new chatbot model.
    """
    database = get_db()
    
    existing = database.get_project(project.project_name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Project '{project.project_name}' already exists.")

    try:
        vector_dim = embedding.get_vector_dimension(project.embedding_model)
        database.create_project(project.project_name, project.embedding_model, vector_dim)
        return {"message": f"Project '{project.project_name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {e}")


@app.get("/projects", response_model=List[str])
def list_projects():
    """
    Returns a list of all available projects.
    """
    return get_db().list_projects()


@app.get("/projects/{project_name}")
def get_project(project_name: str):
    """
    Get details about a specific project.
    """
    project = get_db().get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    return project


@app.delete("/projects/{project_name}")
def delete_project(project_name: str):
    """
    Delete a project and all its data.
    """
    database = get_db()
    if not database.get_project(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    if database.delete_project(project_name):
        return {"message": f"Project '{project_name}' deleted successfully."}
    raise HTTPException(status_code=500, detail="Failed to delete project.")


# =============================================================================
# QA PAIR ENDPOINTS
# =============================================================================

@app.post("/projects/{project_name}/add")
def add_data_to_project(project_name: str, data: models.AddData):
    """
    Adds a new prompt-response pair to the specified project's knowledge base.
    """
    database = get_db()
    metadata = database.get_project_metadata(project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    try:
        model_name = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        prompt_embedding = embedding.generate_embedding(data.prompt, model_name)
        
        qa_id = database.add_qa_pair(project_name, data.prompt, data.response, prompt_embedding, data.weight)
        
        return {"message": "Data added successfully.", "id": qa_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add data: {e}")


@app.get("/projects/{project_name}/qa-pairs")
def get_qa_pairs(project_name: str, limit: int = 200):
    """
    Get all QA pairs for a project.
    """
    database = get_db()
    if not database.get_project(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    return database.get_qa_pairs(project_name, limit)


@app.post("/chat/{project_name}", response_model=List[models.ChatResponseItem])
def chat_with_project(project_name: str, request: models.ChatRequest):
    """
    Performs a semantic search against the project's knowledge base
    and returns the most relevant responses.
    """
    database = get_db()
    metadata = database.get_project_metadata(project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    try:
        model_name = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embedding.generate_embedding(request.prompt, model_name)
        
        results = database.find_similar_prompts(project_name, query_embedding, request.top_k)
        
        if not results:
            return []
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {e}")


# =============================================================================
# AGENT ENDPOINTS
# =============================================================================

@app.post("/agents/business-website-scraper", response_model=models.AgentResultResponse)
def run_business_website_scraper_agent(request: models.BusinessWebsiteScraperRequest):
    """
    Business Website Scraper Agent
    
    An agentic workflow that:
    1. Validates the target project
    2. Scrapes the provided website
    3. Stores scraped content in the database
    4. Generates relevant prompts/questions from content
    5. Generates answers for each prompt using LLM
    6. Creates embeddings and stores Q&A pairs in the knowledge base
    
    Returns detailed step-by-step execution results.
    """
    database = get_db()
    
    from .Agents.business_website_scraper import run_business_website_scraper
    
    result = run_business_website_scraper(
        db_backend=database,
        project_name=request.project_name,
        url=request.url,
        max_pages=request.max_pages,
        business_context=request.business_context,
        default_weight=request.default_weight
    )
    
    if not result.success:
        # Return 200 with success=false for agent failures (not HTTP errors)
        pass
    
    return result.to_dict()


@app.post("/auto-generate-lm-content", deprecated=True)
def auto_generate_lm_content(request: models.AutoGenerateRequest):
    """
    [DEPRECATED] Use /agents/business-website-scraper instead.
    
    Auto-generates LM content by scraping a website and generating Q&A pairs.
    This endpoint is maintained for backward compatibility.
    """
    database = get_db()
    
    from .Agents.business_website_scraper import run_business_website_scraper
    
    result = run_business_website_scraper(
        db_backend=database,
        project_name=request.project_name,
        url=request.url,
        max_pages=request.max_pages,
        business_context=request.business_context,
        default_weight=request.default_weight
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=result.message)
    
    # Return legacy format for backward compatibility
    return {
        "message": result.message,
        "pages_scraped": result.data.get("pages_scraped", 0),
        "prompts_generated": result.data.get("prompts_generated", 0),
        "qa_pairs_added": result.data.get("qa_pairs_added", 0),
        "scraped_domains": result.data.get("domains", [])
    }


# =============================================================================
# CONTENT ENDPOINTS
# =============================================================================

@app.get("/projects/{project_name}/scraped-content")
def get_scraped_content(project_name: str):
    """
    Retrieve scraped content for a project.
    """
    database = get_db()
    if not database.get_project(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    try:
        content = database.get_scraped_content(project_name)
        return {
            "project_name": project_name,
            "content_count": len(content),
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scraped content: {e}")


@app.get("/projects/{project_name}/prompt-file")
def get_prompt_file(project_name: str):
    """
    Retrieve the latest generated prompt file for a project.
    """
    database = get_db()
    if not database.get_project(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    try:
        prompt_file = database.get_latest_prompt_file(project_name)
        if not prompt_file:
            raise HTTPException(status_code=404, detail="No prompt file found for this project.")
        
        return {
            "project_name": project_name,
            "prompt_data": json.loads(prompt_file["prompt_data"]),
            "business_context": prompt_file["business_context"]
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid prompt file data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve prompt file: {e}")


# =============================================================================
# EXPORT/IMPORT ENDPOINTS
# =============================================================================

@app.post("/projects/{project_name}/export")
def export_project(project_name: str, output_path: str = None):
    """
    Export a project to a portable SQLite database file.
    """
    database = get_db()
    if not database.get_project(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    if output_path is None:
        output_path = f"exports/{project_name}/{project_name}.db"
    
    try:
        database.export_to_sqlite(project_name, output_path)
        return {"message": f"Project exported to {output_path}", "path": output_path}
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Export not supported for this database backend.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")


@app.post("/projects/{project_name}/import")
def import_project(project_name: str, input_path: str):
    """
    Import a project from a SQLite database file.
    """
    database = get_db()
    if database.get_project(project_name):
        raise HTTPException(status_code=409, detail=f"Project '{project_name}' already exists.")
    
    try:
        database.import_from_sqlite(project_name, input_path)
        return {"message": f"Project imported from {input_path}"}
    except NotImplementedError:
        raise HTTPException(status_code=501, detail="Import not supported for this database backend.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/config/database")
def get_database_config():
    """
    Get current database configuration.
    """
    return {
        "type": config.DATABASE_TYPE,
        "projects_dir": str(config.PROJECTS_DIR) if config.DATABASE_TYPE == "sqlite" else None,
        "url_configured": bool(config.DATABASE_URL) if config.DATABASE_TYPE == "postgresql" else None
    }


@app.get("/config/embedding-models")
def get_embedding_models():
    """
    Get available embedding models.
    """
    return config.EMBEDDING_MODELS


# =============================================================================
# ACTOR ENDPOINTS
# =============================================================================

@app.post("/projects/{project_name}/actors", response_model=models.ActorResponse)
def create_actor(project_name: str, actor: models.ActorCreate):
    """Create a new LLM actor for a project."""
    database = get_db()
    project = database.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    # Check if actor name already exists
    existing = database.get_actor_by_name(project_name, actor.actor_name)
    if existing:
        raise HTTPException(status_code=409, detail=f"Actor '{actor.actor_name}' already exists.")
    
    try:
        actor_id = database.create_actor(
            project_name=project_name,
            actor_name=actor.actor_name,
            description=actor.description or "",
            prompt_messages=[m.model_dump() for m in actor.prompt_messages],
            model_name=actor.model_name,
            temperature=actor.temperature,
            max_tokens=actor.max_tokens,
            top_p=actor.top_p,
            top_k=actor.top_k,
            repetition_penalty=actor.repetition_penalty,
            other_generation_parameters=actor.other_generation_parameters
        )
        
        # Return the created actor
        return database.get_actor(project_name, actor_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_name}/actors", response_model=List[models.ActorResponse])
def list_actors(project_name: str):
    """List all actors for a project."""
    database = get_db()
    project = database.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    return database.list_actors(project_name)


@app.get("/projects/{project_name}/actors/{actor_id}", response_model=models.ActorResponse)
def get_actor(project_name: str, actor_id: str):
    """Get a specific actor by ID."""
    database = get_db()
    actor = database.get_actor(project_name, actor_id)
    if not actor:
        raise HTTPException(status_code=404, detail=f"Actor '{actor_id}' not found.")
    return actor


@app.put("/projects/{project_name}/actors/{actor_id}", response_model=models.ActorResponse)
def update_actor(project_name: str, actor_id: str, updates: models.ActorUpdate):
    """Update an actor's properties."""
    database = get_db()
    actor = database.get_actor(project_name, actor_id)
    if not actor:
        raise HTTPException(status_code=404, detail=f"Actor '{actor_id}' not found.")
    
    # Build updates dict
    update_dict = updates.model_dump(exclude_unset=True)
    if 'prompt_messages' in update_dict and update_dict['prompt_messages']:
        update_dict['prompt_messages'] = [m.model_dump() if hasattr(m, 'model_dump') else m for m in update_dict['prompt_messages']]
    
    if not update_dict:
        return actor
    
    success = database.update_actor(project_name, actor_id, update_dict)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update actor.")
    
    return database.get_actor(project_name, actor_id)


@app.delete("/projects/{project_name}/actors/{actor_id}")
def delete_actor(project_name: str, actor_id: str):
    """Delete an actor."""
    database = get_db()
    actor = database.get_actor(project_name, actor_id)
    if not actor:
        raise HTTPException(status_code=404, detail=f"Actor '{actor_id}' not found.")
    
    success = database.delete_actor(project_name, actor_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete actor.")
    
    return {"message": f"Actor '{actor_id}' deleted."}


# =============================================================================
# PERSONA ENDPOINTS
# =============================================================================

@app.post("/projects/{project_name}/personas", response_model=models.PersonaResponse)
def create_persona(project_name: str, persona: models.PersonaCreate):
    """Create a new persona for a project."""
    database = get_db()
    project = database.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    try:
        persona_id = database.create_persona(
            project_name=project_name,
            persona_name=persona.persona_name,
            display_name=persona.display_name,
            is_ai=persona.is_ai,
            fallback_actor_id=persona.fallback_actor_id
        )
        
        return database.get_persona(project_name, persona_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_name}/personas", response_model=List[models.PersonaResponse])
def list_personas(project_name: str):
    """List all personas for a project."""
    database = get_db()
    project = database.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    return database.list_personas(project_name)


@app.get("/projects/{project_name}/personas/{persona_id}", response_model=models.PersonaResponse)
def get_persona(project_name: str, persona_id: str):
    """Get a specific persona by ID."""
    database = get_db()
    persona = database.get_persona(project_name, persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")
    return persona


@app.delete("/projects/{project_name}/personas/{persona_id}")
def delete_persona(project_name: str, persona_id: str):
    """Delete a persona."""
    database = get_db()
    persona = database.get_persona(project_name, persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found.")
    
    success = database.delete_persona(project_name, persona_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete persona.")
    
    return {"message": f"Persona '{persona_id}' deleted."}


# =============================================================================
# CHAT SESSION ENDPOINTS
# =============================================================================

@app.post("/projects/{project_name}/sessions", response_model=models.ChatSessionResponse)
def create_chat_session(project_name: str, session: models.ChatSessionCreate):
    """Create a new chat session."""
    database = get_db()
    project = database.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    # Validate actor exists
    actor = database.get_actor(project_name, session.actor_id)
    if not actor:
        raise HTTPException(status_code=404, detail=f"Actor '{session.actor_id}' not found.")
    
    # Validate persona exists
    persona = database.get_persona(project_name, session.persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail=f"Persona '{session.persona_id}' not found.")
    
    try:
        session_id = database.create_chat_session(
            project_name=project_name,
            actor_id=session.actor_id,
            persona_id=session.persona_id,
            session_name=session.session_name
        )
        
        return database.get_chat_session(project_name, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_name}/sessions", response_model=List[models.ChatSessionResponse])
def list_chat_sessions(
    project_name: str, 
    persona_id: str = None, 
    actor_id: str = None
):
    """List chat sessions, optionally filtered by persona or actor."""
    database = get_db()
    project = database.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    return database.list_chat_sessions(project_name, persona_id, actor_id)


@app.get("/projects/{project_name}/sessions/{session_id}", response_model=models.ChatSessionResponse)
def get_chat_session(project_name: str, session_id: str):
    """Get a specific chat session."""
    database = get_db()
    session = database.get_chat_session(project_name, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session


@app.delete("/projects/{project_name}/sessions/{session_id}")
def delete_chat_session(project_name: str, session_id: str):
    """Delete a chat session and all its messages."""
    database = get_db()
    session = database.get_chat_session(project_name, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    success = database.delete_chat_session(project_name, session_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete session.")
    
    return {"message": f"Session '{session_id}' deleted."}


@app.get("/projects/{project_name}/sessions/{session_id}/history", response_model=models.ChatHistoryResponse)
def get_chat_history(project_name: str, session_id: str, limit: int = None):
    """Get chat history for a session."""
    database = get_db()
    session = database.get_chat_session(project_name, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    messages = database.get_chat_history(project_name, session_id, limit)
    
    return {
        "session_id": session_id,
        "messages": messages,
        "total_messages": len(messages)
    }


# =============================================================================
# LLM CHAT ENDPOINTS
# =============================================================================

@app.post("/projects/{project_name}/sessions/{session_id}/chat", response_model=models.LLMChatResponse)
def chat_with_llm(project_name: str, session_id: str, request: models.LLMChatRequest):
    """
    Send a message to the LLM and get a response.
    Uses QA pairs as context and maintains chat history.
    """
    database = get_db()
    session = database.get_chat_session(project_name, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    
    try:
        from .services.llm_chat import create_chat_service
        
        chat_service = create_chat_service(database)
        
        result = chat_service.chat(
            project_name=project_name,
            session_id=session_id,
            user_message=request.message,
            use_context=request.use_context,
            context_top_k=request.context_top_k,
            max_context_tokens=request.max_context_tokens,
            max_history_tokens=request.max_history_tokens
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_name}/chat-quick", response_model=models.LLMChatResponse)
def quick_chat(project_name: str, request: models.LLMChatRequest, actor_name: str = None):
    """
    Quick chat endpoint - creates a temporary session and sends a message.
    Useful for single-turn interactions without session management.
    
    If actor_name is not provided, uses the first available actor.
    """
    database = get_db()
    project = database.get_project(project_name)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    # Get or find actor
    actor = None
    if actor_name:
        actor = database.get_actor_by_name(project_name, actor_name)
        if not actor:
            raise HTTPException(status_code=404, detail=f"Actor '{actor_name}' not found.")
    else:
        actors = database.list_actors(project_name)
        if not actors:
            raise HTTPException(status_code=404, detail="No actors found. Create an actor first.")
        actor = actors[0]
    
    # Get or create a default persona
    default_persona = database.get_persona_by_name(project_name, "default_user")
    if not default_persona:
        persona_id = database.create_persona(
            project_name=project_name,
            persona_name="default_user",
            display_name="User",
            is_ai=False
        )
        default_persona = database.get_persona(project_name, persona_id)
    
    # Create a temporary session
    session_id = database.create_chat_session(
        project_name=project_name,
        actor_id=actor["actor_id"],
        persona_id=default_persona["persona_id"],
        session_name=f"Quick chat - {request.message[:50]}"
    )
    
    try:
        from .services.llm_chat import create_chat_service
        
        chat_service = create_chat_service(database)
        
        result = chat_service.chat(
            project_name=project_name,
            session_id=session_id,
            user_message=request.message,
            use_context=request.use_context,
            context_top_k=request.context_top_k,
            max_context_tokens=request.max_context_tokens,
            max_history_tokens=request.max_history_tokens
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INFERENCE CONFIG ENDPOINTS
# =============================================================================

@app.get("/config/inference")
def get_inference_config():
    """Get current inference configuration."""
    return {
        "base_url": config.get_config_value("inference", "base_url", "http://localhost:8080/v1"),
        "default_model": config.get_config_value("inference", "default_model", "gpt-4"),
        "api_key_configured": bool(config.get_config_value("inference", "api_key", None))
    }


@app.get("/tokenizer/info")
def get_tokenizer_info():
    """Get information about the token counter."""
    from .services.token_counter import get_tokenizer_info
    return get_tokenizer_info()