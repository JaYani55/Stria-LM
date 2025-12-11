import asyncio
import sqlite3
import json
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.database import engine, AsyncSessionLocal, create_project, add_qa_pair, add_scraped_content, add_prompt_file
from src.models_db import Base

PROJECTS_DIR = Path("projects")

async def migrate():
    print("Starting migration from SQLite to PostgreSQL...")
    
    # Initialize DB (create tables)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSessionLocal() as session:
        # Iterate over project directories
        if not PROJECTS_DIR.exists():
            print("No projects directory found.")
            return

        for project_dir in PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            
            project_name = project_dir.name
            db_path = project_dir / f"{project_name}.db"
            
            if not db_path.exists():
                print(f"Skipping {project_name}: No database file found.")
                continue
                
            print(f"Migrating project: {project_name}")
            
            try:
                # Connect to SQLite DB
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 1. Get Metadata
                cursor.execute("SELECT key, value FROM metadata")
                metadata = dict(cursor.fetchall())
                embedding_model = metadata.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                vector_dim = int(metadata.get("vector_dimension", 384))
                
                # Create Project in Postgres
                # Check if exists first
                from src.database import get_project_by_name
                existing = await get_project_by_name(session, project_name)
                if not existing:
                    await create_project(session, project_name, embedding_model, vector_dim)
                    print(f"  Created project '{project_name}'")
                else:
                    print(f"  Project '{project_name}' already exists, skipping creation.")

                # 2. Migrate QA Pairs
                # We need to join qa_text and qa_pairs (vec table)
                # Note: In the old schema, qa_pairs was a virtual table using rowid
                # We need to be careful about how we join.
                # qa_text.id matches qa_pairs.rowid
                
                cursor.execute("""
                    SELECT t.prompt_text, t.response_text, t.weight, v.prompt_embedding
                    FROM qa_text t
                    JOIN qa_pairs v ON t.id = v.rowid
                """)
                
                qa_rows = cursor.fetchall()
                print(f"  Migrating {len(qa_rows)} QA pairs...")
                
                for row in qa_rows:
                    prompt, response, weight, embedding_blob = row
                    # Convert binary blob back to numpy array
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                    
                    # Add to Postgres
                    await add_qa_pair(session, project_name, prompt, response, embedding, weight if weight else 1.0)
                
                # 3. Migrate Scraped Content
                try:
                    cursor.execute("SELECT url, title, content, domain FROM scraped_content")
                    scraped_rows = cursor.fetchall()
                    if scraped_rows:
                        print(f"  Migrating {len(scraped_rows)} scraped pages...")
                        scraped_data = [
                            {"url": r[0], "title": r[1], "content": r[2], "domain": r[3]}
                            for r in scraped_rows
                        ]
                        await add_scraped_content(session, project_name, scraped_data)
                except sqlite3.OperationalError:
                    print("  No scraped_content table found.")

                # 4. Migrate Prompt Files
                try:
                    cursor.execute("SELECT prompt_data, business_context FROM prompt_files")
                    prompt_rows = cursor.fetchall()
                    if prompt_rows:
                        print(f"  Migrating {len(prompt_rows)} prompt files...")
                        for r in prompt_rows:
                            await add_prompt_file(session, project_name, r[0], r[1])
                except sqlite3.OperationalError:
                    print("  No prompt_files table found.")

                conn.close()
                print(f"  Successfully migrated {project_name}")

            except Exception as e:
                print(f"  Failed to migrate {project_name}: {e}")
                import traceback
                traceback.print_exc()

    print("Migration completed.")

if __name__ == "__main__":
    asyncio.run(migrate())
