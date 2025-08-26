from fastapi import FastAPI, HTTPException, Body
from typing import List
import os

from . import models
from . import database
from . import embedding
from . import config

app = FastAPI(
    title="Stria-LM",
    description="A framework for portable, file-based retrieval models.",
    version="0.1.0",
)

@app.on_event("startup")
async def startup_event():
    # This is where you could load models, but we do it lazily in embedding.py
    print("Stria-LM server started.")
    # Ensure the base projects directory exists
    config.PROJECTS_DIR.mkdir(exist_ok=True)


@app.post("/projects", status_code=201)
def create_project(project: models.ProjectCreate):
    """
    Creates a new project, which corresponds to a new chatbot model.
    This initializes a dedicated database for the project.
    """
    project_dir = config.PROJECTS_DIR / project.project_name
    if project_dir.exists():
        raise HTTPException(status_code=409, detail=f"Project '{project.project_name}' already exists.")

    try:
        # Get vector dimension from the model
        temp_model = embedding.get_embedding_model(project.embedding_model)
        vector_dim = temp_model.get_sentence_embedding_dimension()
        
        database.init_db(project.project_name, project.embedding_model, vector_dim, config.PROJECTS_DIR)
        return {"message": f"Project '{project.project_name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {e}")


@app.get("/projects", response_model=List[str])
def list_projects():
    """
    Returns a list of all available projects.
    """
    return [d.name for d in config.PROJECTS_DIR.iterdir() if d.is_dir()]


@app.post("/projects/{project_name}/add")
def add_data_to_project(project_name: str, data: models.AddData):
    """
    Adds a new prompt-response pair to the specified project's knowledge base.
    """
    metadata = database.get_project_metadata(project_name, config.PROJECTS_DIR)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    try:
        model_name = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        prompt_embedding = embedding.generate_embedding(data.prompt, model_name)
        
        database.add_qa_pair(project_name, data.prompt, data.response, prompt_embedding, config.PROJECTS_DIR)
        
        return {"message": "Data added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add data: {e}")


@app.post("/chat/{project_name}", response_model=List[models.ChatResponseItem])
def chat_with_project(project_name: str, request: models.ChatRequest):
    """
    Performs a semantic search against the project's knowledge base
    and returns the most relevant responses.
    """
    metadata = database.get_project_metadata(project_name, config.PROJECTS_DIR)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found.")

    try:
        model_name = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = embedding.generate_embedding(request.prompt, model_name)
        
        results = database.find_similar_prompts(project_name, query_embedding, request.top_k, config.PROJECTS_DIR)
        
        if not results:
            return []
            
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {e}")
