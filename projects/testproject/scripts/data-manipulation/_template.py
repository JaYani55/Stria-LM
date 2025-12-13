"""
Data Manipulation Template
==========================
A template for creating data manipulation scripts.

Usage:
    Customize the data transformations below.
    This script can read from and write to the project database.
"""
import sqlite3
from pathlib import Path


def get_db_connection(project_name: str) -> sqlite3.Connection:
    """Get connection to the project database."""
    db_path = Path(f"projects/{project_name}/project_database.db")
    return sqlite3.connect(db_path)


def transform_data(project_name: str):
    """
    Perform data transformations on the project database.
    
    Customize this function to:
    - Clean and normalize data
    - Merge or split records
    - Calculate derived fields
    - Apply business logic
    """
    conn = get_db_connection(project_name)
    cursor = conn.cursor()
    
    try:
        # Example: Get all QA pairs
        cursor.execute("SELECT id, prompt_text, response_text FROM qa_text")
        rows = cursor.fetchall()
        
        for row in rows:
            id_, prompt, response = row
            # Add your transformation logic here
            # Example: Update a field
            # cursor.execute("UPDATE qa_text SET weight = ? WHERE id = ?", (1.5, id_))
            pass
        
        conn.commit()
        print(f"Processed {len(rows)} records")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    project = sys.argv[1] if len(sys.argv) > 1 else "testproject"
    transform_data(project)
