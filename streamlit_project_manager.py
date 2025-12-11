import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path
from typing import List, Optional
import sys
import os

# Ensure src module can be imported
sys.path.append(os.getcwd())

from src.config import PROJECTS_DIR
from src.database import get_db_connection, get_db_path, re_embed_prompts

ROW_LIMIT = 200

def discover_projects(projects_dir: Path) -> List[str]:
    """Return the list of project names that have an associated SQLite database."""
    project_names: List[str] = []
    if not projects_dir.exists():
        return []
        
    for entry in projects_dir.iterdir():
        if not entry.is_dir():
            continue
        db_path = get_db_path(entry.name, projects_dir)
        if db_path.exists():
            project_names.append(entry.name)
    return sorted(project_names)

def get_tables(conn: sqlite3.Connection) -> List[str]:
    """Get a list of tables in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name")
    return [row[0] for row in cursor.fetchall()]

def get_table_primary_key(conn: sqlite3.Connection, table_name: str) -> Optional[str]:
    """Get the primary key column name for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    # columns is a list of tuples/rows. 
    # structure: (cid, name, type, notnull, dflt_value, pk)
    # pk is the 6th element (index 5).
    for col in columns:
        # sqlite3.Row access by name if row_factory is set, but here we might get tuples if not set.
        # The original code sets row_factory = sqlite3.Row.
        # Let's assume we can access by index or name if we set it.
        if isinstance(col, sqlite3.Row):
            if col['pk']:
                return col['name']
        else:
            if col[5]: # pk flag
                return col[1] # name
    return None

def load_table_data(conn: sqlite3.Connection, table_name: str, limit: int = ROW_LIMIT) -> pd.DataFrame:
    """Load data from a table into a pandas DataFrame."""
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    return pd.read_sql_query(query, conn)

def update_database(conn: sqlite3.Connection, table_name: str, pk_name: str, original_df: pd.DataFrame, edited_df: pd.DataFrame):
    """Compare original and edited DataFrames and update the database."""
    changes_count = 0
    cursor = conn.cursor()
    
    # Iterate over indices to find changes
    # We assume the index hasn't changed and corresponds to the same rows because we are editing in place.
    # However, st.data_editor might reorder if sorting is enabled?
    # If the user sorts in the UI, the index in edited_df might be different?
    # st.data_editor returns a dataframe with the same index as input usually.
    
    # To be safe, we should align by index if possible, or just iterate if we trust the order.
    # Let's iterate by the DataFrame index.
    
    for i in original_df.index:
        if i not in edited_df.index:
            continue # Row deleted? (st.data_editor supports deletion if configured, but we assume fixed rows for now)
            
        row_old = original_df.loc[i]
        row_new = edited_df.loc[i]
        
        if not row_old.equals(row_new):
            # Something changed in this row
            pk_value = row_old[pk_name]
            
            for col in original_df.columns:
                if row_old[col] != row_new[col]:
                    new_value = row_new[col]
                    # Update this cell
                    try:
                        query = f'UPDATE "{table_name}" SET "{col}" = ? WHERE "{pk_name}" = ?'
                        cursor.execute(query, (new_value, pk_value))
                        changes_count += 1
                    except Exception as e:
                        st.error(f"Error updating {col} for ID {pk_value}: {e}")
    
    if changes_count > 0:
        conn.commit()
        st.success(f"Successfully updated {changes_count} cell(s).")
    else:
        st.info("No changes detected.")

def main():
    st.set_page_config(page_title="Stria-LM Project Manager", layout="wide")
    
    st.title("Stria-LM Project Manager")

    # Sidebar
    st.sidebar.header("Project Selection")
    projects = discover_projects(PROJECTS_DIR)
    
    if not projects:
        st.sidebar.warning(f"No projects found in {PROJECTS_DIR}")
        return

    selected_project = st.sidebar.selectbox("Select Project", projects)

    if selected_project:
        db_path = get_db_path(selected_project, PROJECTS_DIR)
        
        if not db_path.exists():
            st.error(f"Database not found at {db_path}")
            return

        # Connect to database
        # We use a context manager or just open/close. 
        # Since Streamlit reruns, we open a new connection each time.
        try:
            conn = get_db_connection(db_path)
            # Set row factory to Row for easier access if needed, though pandas handles reading well.
            conn.row_factory = sqlite3.Row
        except Exception as e:
            st.error(f"Failed to connect to database: {e}")
            return

        # Tabs for different functionalities
        tab_browser, tab_ops = st.tabs(["Data Browser", "Database Operations"])

        with tab_browser:
            st.subheader("Data Browser")
            tables = get_tables(conn)
            
            if not tables:
                st.warning("No tables found in the database.")
            else:
                selected_table = st.selectbox("Select Table", tables)
                
                if selected_table:
                    pk_name = get_table_primary_key(conn, selected_table)
                    
                    if not pk_name:
                        st.warning(f"Table '{selected_table}' has no primary key. Editing is disabled.")
                        df = load_table_data(conn, selected_table)
                        st.dataframe(df)
                    else:
                        # Load data
                        df = load_table_data(conn, selected_table)
                        
                        st.info(f"Showing first {ROW_LIMIT} rows. Primary Key: {pk_name}")
                        
                        # Data Editor
                        edited_df = st.data_editor(df, key=f"editor_{selected_project}_{selected_table}", num_rows="fixed")
                        
                        if st.button("Save Changes"):
                            update_database(conn, selected_table, pk_name, df, edited_df)
                            st.rerun()

        with tab_ops:
            st.subheader("Database Operations")
            
            st.markdown("### Re-embed Prompts")
            st.write("Regenerate embeddings for prompts in the database.")
            
            re_embed_mode = st.radio("Mode", ["All Prompts", "Specific IDs"])
            
            ids_input = ""
            if re_embed_mode == "Specific IDs":
                ids_input = st.text_input("Enter Prompt IDs (comma-separated)", placeholder="1, 2, 3")
            
            if st.button("Start Re-embedding"):
                ids_to_process = "all"
                if re_embed_mode == "Specific IDs":
                    try:
                        ids_to_process = [int(x.strip()) for x in ids_input.split(",") if x.strip()]
                        if not ids_to_process:
                            st.error("Please enter at least one valid ID.")
                            ids_to_process = None
                    except ValueError:
                        st.error("Invalid input. Please enter numbers separated by commas.")
                        ids_to_process = None
                
                if ids_to_process:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total):
                        progress = min(current / total, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {current} of {total}...")
                    
                    try:
                        with st.spinner("Re-embedding..."):
                            count = re_embed_prompts(
                                selected_project, 
                                PROJECTS_DIR, 
                                ids_to_process, 
                                progress_callback=progress_callback
                            )
                        st.success(f"Successfully re-embedded {count} prompt(s).")
                    except Exception as e:
                        st.error(f"Error during re-embedding: {e}")

        conn.close()

if __name__ == "__main__":
    main()
