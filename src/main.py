from fastapi import FastAPI, HTTPException, Body, Depends
from typing import List
import os
import sys
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import OperationalError

# Enable weights for testing
os.environ["ENABLE_WEIGHTS"] = "true"

from . import models
from . import database
from . import embedding
from . import config
from .Scraper.scraper_universal import scrape_website
from .Agents.promptfile_agent import generate_prompts_from_content
from .Agents.answer_agent_universal import generate_answers_for_business

app = FastAPI(
    title="Stria-LM",
    description="A framework for portable, file-based retrieval models.",
    version="0.1.1",
)

@app.on_event("startup")
async def startup_event():
    # This is where you could load models, but we do it lazily in embedding.py
    print("Stria-LM server started.")
    # Initialize database tables
    try:
        await database.init_db()
    except (OSError, OperationalError) as e:
        print("\n" + "="*60)
        print("CRITICAL ERROR: Could not connect to the database.")
        print(f"Connection URL: {config.DATABASE_URL}")
        print("Please ensure PostgreSQL is running and the configuration in config.toml is correct.")
        
        if sys.platform == "win32":
            print("\n[Windows] It seems PostgreSQL is not running or not installed.")
            print("1. Download and install from: https://www.postgresql.org/download/windows/")
            print("2. Ensure the service is started in 'Services.msc'.")
            print("3. Create the database and extension:")
            print("   > CREATE DATABASE strialm;")
            print("   > \\c strialm")
            print("   > CREATE EXTENSION vector;")
        elif sys.platform.startswith("linux"):
            print("\n[Linux] It seems PostgreSQL is not running or not installed.")
            print("1. Install: sudo apt install postgresql postgresql-contrib (Debian/Ubuntu)")
            print("2. Start service: sudo systemctl start postgresql")
            print("3. Create the database and extension:")
            print("   $ sudo -u postgres psql")
            print("   postgres=# CREATE DATABASE strialm;")
            print("   postgres=# \\c strialm")
            print("   postgres=# CREATE EXTENSION vector;")
        
        print("\nDetails:", e)
        print("="*60 + "\n")
        raise e
    except Exception as e:
        print(f"An error occurred during database initialization: {e}")
        raise e


@app.post("/projects", status_code=201)
async def create_project(project: models.ProjectCreate, session: AsyncSession = Depends(database.get_db)):
    """
    Creates a new project, which corresponds to a new chatbot model.
    This initializes a dedicated database for the project.
    """
    existing_project = await database.get_project_by_name(session, project.project_name)
    if existing_project:
        raise HTTPException(status_code=409, detail=f"Project '{project.project_name}' already exists.")

    try:
        # Get vector dimension from the model
        vector_dim = embedding.get_vector_dimension(project.embedding_model)
        
        await database.create_project(session, project.project_name, project.embedding_model, vector_dim)
        return {"message": f"Project '{project.project_name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {e}")


@app.get("/projects", response_model=List[str])
async def list_projects(session: AsyncSession = Depends(database.get_db)):
    """
    Returns a list of all available projects.
    """
    return await database.list_projects(session)


@app.post("/projects/{project_name}/add")
async def add_data_to_project(project_name: str, data: models.AddData, session: AsyncSession = Depends(database.get_db)):
    """
    Adds a new prompt-response pair to the specified project's knowledge base.
    """
    metadata = await database.get_project_metadata(session, project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    try:
        model_name = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        prompt_embedding = embedding.generate_embedding(data.prompt, model_name)
        
        await database.add_qa_pair(session, project_name, data.prompt, data.response, prompt_embedding, data.weight)
        
        return {"message": "Data added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add data: {e}")


@app.post("/chat/{project_name}", response_model=List[models.ChatResponseItem])
async def chat_with_project(project_name: str, request: models.ChatRequest, session: AsyncSession = Depends(database.get_db)):
    """
    Performs a semantic search against the project's knowledge base
    and returns the most relevant responses.
    """
    metadata = await database.get_project_metadata(session, project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    try:
        model_name = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embedding.generate_embedding(request.prompt, model_name)
        
        results = await database.find_similar_prompts(session, project_name, query_embedding, request.top_k)
        
        if not results:
            return []
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {e}")


@app.post("/auto-generate-lm-content")
async def auto_generate_lm_content(request: models.AutoGenerateRequest, session: AsyncSession = Depends(database.get_db)):
    """
    Auto-generates LM content by:
    1. Scraping the provided website
    2. Generating relevant prompts based on content
    3. Creating answers for those prompts
    4. Populating the project database
    """
    # Check if project exists
    metadata = await database.get_project_metadata(session, request.project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found.")
    
    try:
        # Step 1: Scrape the website
        print(f"Scraping website: {request.url}")
        # Note: scrape_website is synchronous, might block the event loop. 
        # Ideally should be run in a threadpool or made async.
        scraped_data = scrape_website(request.url, request.max_pages)
        
        if not scraped_data:
            raise HTTPException(status_code=400, detail="No content could be scraped from the provided URL.")
        
        # Step 2: Save scraped content to database
        print(f"Saving {len(scraped_data)} pages to database")
        await database.add_scraped_content(session, request.project_name, scraped_data)
        
        # Step 3: Generate prompts based on content
        print("Generating prompts based on scraped content")
        prompt_file_data = generate_prompts_from_content(scraped_data, request.business_context)
        
        # Step 4: Save prompt file to database
        prompt_file_json = json.dumps(prompt_file_data)
        await database.add_prompt_file(session, request.project_name, prompt_file_json, 
                                request.business_context or "")
        
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
                await database.add_qa_pair(
                    session,
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


@app.get("/projects/{project_name}/scraped-content")
async def get_scraped_content(project_name: str, session: AsyncSession = Depends(database.get_db)):
    """
    Retrieve scraped content for a project
    """
    metadata = await database.get_project_metadata(session, project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    try:
        content = await database.get_scraped_content(session, project_name)
        return {
            "project_name": project_name,
            "content_count": len(content),
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scraped content: {e}")


@app.get("/projects/{project_name}/prompt-file")
async def get_prompt_file(project_name: str, session: AsyncSession = Depends(database.get_db)):
    """
    Retrieve the latest generated prompt file for a project
    """
    metadata = await database.get_project_metadata(session, project_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")
    
    try:
        prompt_file = await database.get_latest_prompt_file(session, project_name)
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
