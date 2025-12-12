from fastapi import FastAPI, HTTPException, Body
from typing import List
import os
import sys
import json

from . import models
from . import embedding
from . import config
from .database import get_database, DatabaseBackend
from .Scraper.scraper_universal import scrape_website
from .Agents.promptfile_agent import generate_prompts_from_content
from .Agents.answer_agent_universal import generate_answers_for_business

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
# AUTO-GENERATION ENDPOINTS
# =============================================================================

@app.post("/auto-generate-lm-content")
def auto_generate_lm_content(request: models.AutoGenerateRequest):
    """
    Auto-generates LM content by:
    1. Scraping the provided website
    2. Generating relevant prompts based on content
    3. Creating answers for those prompts
    4. Populating the project database
    """
    database = get_db()
    metadata = database.get_project_metadata(request.project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")
    
    try:
        # Step 1: Scrape the website
        print(f"Scraping website: {request.url}")
        scraped_data = scrape_website(request.url, request.max_pages)
        
        if not scraped_data:
            raise HTTPException(status_code=400, detail="No content could be scraped from the provided URL.")
        
        # Step 2: Save scraped content to database
        print(f"Saving {len(scraped_data)} pages to database")
        database.add_scraped_content(request.project_name, scraped_data)
        
        # Step 3: Generate prompts based on content
        print("Generating prompts based on scraped content")
        prompt_file_data = generate_prompts_from_content(scraped_data, request.business_context)
        
        # Step 4: Save prompt file to database
        prompt_file_json = json.dumps(prompt_file_data)
        database.add_prompt_file(request.project_name, prompt_file_json, request.business_context or "")
        
        # Step 5: Generate answers for prompts
        print(f"Generating answers for {len(prompt_file_data['prompts'])} prompts")
        qa_pairs = generate_answers_for_business(
            prompt_file_data["prompts"], 
            scraped_data, 
            request.business_context,
            request.default_weight
        )
        
        # Step 6: Add Q&A pairs to the main knowledge base
        model_name = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        
        added_count = 0
        for qa_pair in qa_pairs:
            try:
                prompt_embedding = embedding.generate_embedding(qa_pair["prompt"], model_name)
                database.add_qa_pair(
                    request.project_name, 
                    qa_pair["prompt"], 
                    qa_pair["response"], 
                    prompt_embedding, 
                    qa_pair.get("weight", 1.0)
                )
                added_count += 1
            except Exception as e:
                print(f"Failed to add Q&A pair: {e}")
                continue
        
        return {
            "message": "Auto-generation completed successfully",
            "pages_scraped": len(scraped_data),
            "prompts_generated": len(prompt_file_data["prompts"]),
            "qa_pairs_added": added_count,
            "scraped_domains": list(set(page.get("domain", "") for page in scraped_data))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-generation failed: {str(e)}")


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
