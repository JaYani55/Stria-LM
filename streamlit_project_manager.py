#!/usr/bin/env python3
"""
Stria-LM Data Manager - Streamlit GUI
Interactive data browsing, editing, and management for Stria-LM projects.
"""
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys
import os
import json

# Ensure src module can be imported
sys.path.append(os.getcwd())

from src.config import PROJECTS_DIR, DATABASE_TYPE, EMBEDDING_MODELS, get_config_value
from src.database import get_database

ROW_LIMIT = 200


def init_session_state():
    """Initialize session state variables."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "current_actor_id" not in st.session_state:
        st.session_state.current_actor_id = None
    if "current_persona_id" not in st.session_state:
        st.session_state.current_persona_id = None


def main():
    st.set_page_config(
        page_title="Stria-LM Data Manager", 
        page_icon="📊",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("📊 Stria-LM Data Manager")
    st.caption(f"Database: {DATABASE_TYPE.upper()}")

    # Initialize database connection
    try:
        db = get_database()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info("Please check your database configuration in the Project Manager.")
        return

    # Sidebar - Project Selection
    st.sidebar.header("Project Selection")
    
    try:
        projects = db.list_projects()
    except Exception as e:
        st.sidebar.error(f"Error loading projects: {e}")
        return
    
    if not projects:
        st.sidebar.warning("No projects found.")
        st.info("Create a project using the Project Manager (Tkinter GUI) first.")
        return

    selected_project = st.sidebar.selectbox("Select Project", projects)
    
    if not selected_project:
        return

    # Display project info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")
    
    try:
        metadata = db.get_project_metadata(selected_project)
        if metadata:
            st.sidebar.write(f"**Model:** {metadata.get('embedding_model', 'N/A')}")
            st.sidebar.write(f"**Dimension:** {metadata.get('vector_dim', 'N/A')}")
            st.sidebar.write(f"**Created:** {metadata.get('created_at', 'N/A')}")
    except Exception as e:
        st.sidebar.warning(f"Could not load metadata: {e}")

    # Main content tabs
    tab_qa, tab_search, tab_chat, tab_actors, tab_ops, tab_add = st.tabs([
        "📝 Q&A Pairs", 
        "🔍 Semantic Search",
        "💬 Chat",
        "🎭 Actors",
        "⚙️ Operations",
        "➕ Add Data"
    ])

    # ==========================================================================
    # Q&A PAIRS TAB
    # ==========================================================================
    with tab_qa:
        st.subheader("Q&A Pairs Browser")
        
        try:
            qa_pairs = db.get_qa_pairs(selected_project, limit=ROW_LIMIT)
            
            if not qa_pairs:
                st.info("No Q&A pairs found in this project.")
            else:
                # Convert to DataFrame for display
                df = pd.DataFrame(qa_pairs)
                
                # Display count
                st.info(f"Showing {len(qa_pairs)} Q&A pairs (limit: {ROW_LIMIT})")
                
                # Column selector
                all_columns = list(df.columns)
                display_columns = st.multiselect(
                    "Columns to display",
                    all_columns,
                    default=[c for c in ['id', 'prompt', 'response', 'weight'] if c in all_columns]
                )
                
                if display_columns:
                    # Display as interactive table
                    edited_df = st.data_editor(
                        df[display_columns],
                        key=f"qa_editor_{selected_project}",
                        num_rows="fixed",
                        width='stretch'
                    )
                    
                    # Save changes button
                    if st.button("💾 Save Changes", key="save_qa"):
                        changes_made = 0
                        for idx in df.index:
                            if idx < len(edited_df):
                                for col in display_columns:
                                    if col in df.columns and df.loc[idx, col] != edited_df.loc[idx, col]:
                                        # Update this value
                                        qa_id = df.loc[idx, 'id'] if 'id' in df.columns else idx
                                        try:
                                            db.update_qa_pair(
                                                selected_project, 
                                                qa_id, 
                                                {col: edited_df.loc[idx, col]}
                                            )
                                            changes_made += 1
                                        except Exception as e:
                                            st.error(f"Error updating row {qa_id}: {e}")
                        
                        if changes_made > 0:
                            st.success(f"Updated {changes_made} values.")
                            st.rerun()
                        else:
                            st.info("No changes detected.")
                else:
                    st.warning("Select at least one column to display.")
                    
        except Exception as e:
            st.error(f"Error loading Q&A pairs: {e}")

    # ==========================================================================
    # SEMANTIC SEARCH TAB
    # ==========================================================================
    with tab_search:
        st.subheader("Semantic Search")
        st.write("Search for similar prompts using vector embeddings.")
        
        query = st.text_area("Enter your query:", height=100, placeholder="Type a question or phrase...")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            top_k = st.number_input("Number of results", min_value=1, max_value=50, value=5)
        
        if st.button("🔍 Search", type="primary"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Searching..."):
                    try:
                        from src import embedding as emb_module
                        
                        # Get embedding model for this project
                        metadata = db.get_project_metadata(selected_project)
                        model_name = metadata.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                        
                        # Generate query embedding
                        query_embedding = emb_module.generate_embedding(query, model_name)
                        
                        # Search
                        results = db.find_similar_prompts(selected_project, query_embedding, top_k)
                        
                        if not results:
                            st.info("No matching results found.")
                        else:
                            st.success(f"Found {len(results)} results:")
                            
                            for i, result in enumerate(results, 1):
                                with st.expander(f"Result {i} (Score: {result.get('score', 'N/A'):.4f})" if 'score' in result else f"Result {i}"):
                                    st.markdown(f"**Prompt:** {result.get('prompt', 'N/A')}")
                                    st.markdown(f"**Response:** {result.get('response', 'N/A')}")
                                    if 'weight' in result:
                                        st.caption(f"Weight: {result['weight']}")
                                        
                    except Exception as e:
                        st.error(f"Search failed: {e}")

    # ==========================================================================
    # CHAT TAB
    # ==========================================================================
    with tab_chat:
        st.subheader("💬 Chat with LLM")
        st.write("Chat with an AI assistant using your Q&A pairs as context.")
        
        # Actor and session selection
        col1, col2 = st.columns(2)
        
        with col1:
            actors = db.list_actors(selected_project)
            if not actors:
                st.warning("No actors found. Create an actor first in the Actors tab.")
            else:
                actor_options = {a["actor_name"]: a["actor_id"] for a in actors}
                selected_actor_name = st.selectbox(
                    "Select Actor",
                    list(actor_options.keys()),
                    key="chat_actor_select"
                )
                if selected_actor_name:
                    st.session_state.current_actor_id = actor_options[selected_actor_name]
        
        with col2:
            personas = db.list_personas(selected_project)
            if not personas:
                # Create default persona
                if st.button("Create Default Persona"):
                    persona_id = db.create_persona(
                        selected_project,
                        "user",
                        "User",
                        is_ai=False
                    )
                    st.success("Default persona created!")
                    st.rerun()
            else:
                persona_options = {p["display_name"] or p["persona_name"]: p["persona_id"] for p in personas}
                selected_persona_name = st.selectbox(
                    "Select Persona",
                    list(persona_options.keys()),
                    key="chat_persona_select"
                )
                if selected_persona_name:
                    st.session_state.current_persona_id = persona_options[selected_persona_name]
        
        # Session management
        st.markdown("---")
        
        if st.session_state.current_actor_id and st.session_state.current_persona_id:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                sessions = db.list_chat_sessions(
                    selected_project,
                    persona_id=st.session_state.current_persona_id,
                    actor_id=st.session_state.current_actor_id
                )
                
                if sessions:
                    session_options = {s.get("title") or f"Session {s['session_id'][:8]}...": s["session_id"] for s in sessions}
                    selected_session = st.selectbox(
                        "Select Session",
                        ["New Session"] + list(session_options.keys()),
                        key="chat_session_select"
                    )
                    
                    if selected_session != "New Session":
                        st.session_state.current_session_id = session_options[selected_session]
                        
                        # Load chat history
                        history = db.get_chat_history(selected_project, st.session_state.current_session_id)
                        st.session_state.chat_messages = [
                            {"role": m["role"], "content": m["content"]}
                            for m in history if m["role"] in ("user", "assistant")
                        ]
                    else:
                        st.session_state.current_session_id = None
                        st.session_state.chat_messages = []
                else:
                    st.info("No sessions found. Start a new conversation below.")
            
            with col2:
                if st.button("🔄 New Session", type="secondary"):
                    st.session_state.current_session_id = None
                    st.session_state.chat_messages = []
                    st.rerun()
            
            # Chat interface
            st.markdown("---")
            
            # Chat settings
            with st.expander("⚙️ Chat Settings"):
                use_context = st.checkbox("Use Q&A context", value=True)
                context_top_k = st.slider("Context items", 1, 20, 5)
                max_context_tokens = st.slider("Max context tokens", 500, 8000, 2000)
            
            # Display chat messages
            chat_container = st.container()
            
            with chat_container:
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Type your message..."):
                # Add user message to display
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Create session if needed
                if not st.session_state.current_session_id:
                    session_id = db.create_chat_session(
                        selected_project,
                        actor_id=st.session_state.current_actor_id,
                        persona_id=st.session_state.current_persona_id,
                        session_name=prompt[:50] + "..." if len(prompt) > 50 else prompt
                    )
                    st.session_state.current_session_id = session_id
                
                # Send to LLM
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            from src.services.llm_chat import create_chat_service
                            
                            chat_service = create_chat_service(db)
                            
                            result = chat_service.chat(
                                project_name=selected_project,
                                session_id=st.session_state.current_session_id,
                                user_message=prompt,
                                use_context=use_context,
                                context_top_k=context_top_k,
                                max_context_tokens=max_context_tokens
                            )
                            
                            response = result["message"]
                            st.write(response)
                            
                            # Show metadata
                            st.caption(f"🎯 Model: {result['model']} | 📊 Tokens: {result['total_tokens']} | ⏱️ {result['latency_ms']:.0f}ms | 📚 Context: {result['context_used']} items")
                            
                            st.session_state.chat_messages.append({"role": "assistant", "content": response})
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.info("Select an actor and persona to start chatting.")

    # ==========================================================================
    # ACTORS TAB
    # ==========================================================================
    with tab_actors:
        st.subheader("🎭 Actor Management")
        st.write("Create and manage LLM actors with custom personas and settings.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Create New Actor")
            
            with st.form("create_actor_form"):
                actor_name = st.text_input("Actor Name", placeholder="e.g., Support Agent")
                description = st.text_area("Description", placeholder="Describe the actor's purpose...")
                
                model_name = st.text_input("Model Name", value="gpt-4", help="OpenAI-compatible model name")
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
                max_tokens = st.number_input("Max Tokens", 100, 16000, 2048)
                
                system_prompt = st.text_area(
                    "System Prompt",
                    placeholder="You are a helpful assistant...",
                    height=150
                )
                
                if st.form_submit_button("Create Actor", type="primary"):
                    if not actor_name:
                        st.error("Actor name is required.")
                    else:
                        try:
                            prompt_messages = []
                            if system_prompt:
                                prompt_messages = [{"role": "system", "content": system_prompt}]
                            
                            actor_id = db.create_actor(
                                project_name=selected_project,
                                actor_name=actor_name,
                                description=description,
                                prompt_messages=prompt_messages,
                                model_name=model_name,
                                temperature=temperature,
                                max_tokens=max_tokens
                            )
                            st.success(f"Actor '{actor_name}' created!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create actor: {e}")
        
        with col2:
            st.markdown("### Existing Actors")
            
            actors = db.list_actors(selected_project)
            
            if not actors:
                st.info("No actors found. Create one to get started.")
            else:
                for actor in actors:
                    with st.expander(f"🎭 {actor['actor_name']}"):
                        st.write(f"**Description:** {actor.get('description', 'N/A')}")
                        st.write(f"**Model:** {actor.get('model_name', 'N/A')}")
                        st.write(f"**Temperature:** {actor.get('temperature', 'N/A')}")
                        st.write(f"**Max Tokens:** {actor.get('max_tokens', 'N/A')}")
                        
                        if st.button("🗑️ Delete", key=f"delete_actor_{actor['actor_id']}"):
                            db.delete_actor(selected_project, actor['actor_id'])
                            st.success("Actor deleted!")
                            st.rerun()
        
        # Personas section
        st.markdown("---")
        st.markdown("### 👤 Personas")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Create Persona")
            
            with st.form("create_persona_form"):
                persona_name = st.text_input("Persona Name", placeholder="e.g., customer")
                display_name = st.text_input("Display Name", placeholder="e.g., Customer")
                is_ai = st.checkbox("Is AI Persona")
                
                if st.form_submit_button("Create Persona"):
                    if not persona_name or not display_name:
                        st.error("Name and display name are required.")
                    else:
                        try:
                            persona_id = db.create_persona(
                                selected_project,
                                persona_name=persona_name,
                                display_name=display_name,
                                is_ai=is_ai
                            )
                            st.success(f"Persona '{display_name}' created!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to create persona: {e}")
        
        with col2:
            st.markdown("#### Existing Personas")
            
            personas = db.list_personas(selected_project)
            
            if not personas:
                st.info("No personas found.")
            else:
                for persona in personas:
                    with st.expander(f"👤 {persona.get('display_name', persona['persona_name'])}"):
                        st.write(f"**Name:** {persona['persona_name']}")
                        st.write(f"**Is AI:** {'Yes' if persona.get('is_ai') else 'No'}")
                        
                        if st.button("🗑️ Delete", key=f"delete_persona_{persona['persona_id']}"):
                            db.delete_persona(selected_project, persona['persona_id'])
                            st.success("Persona deleted!")
                            st.rerun()

    # ==========================================================================
    # OPERATIONS TAB
    # ==========================================================================
    with tab_ops:
        st.subheader("Database Operations")
        
        # Re-embedding section
        st.markdown("### 🔄 Re-embed Prompts")
        st.write("Regenerate vector embeddings for prompts in the database.")
        
        re_embed_mode = st.radio(
            "Mode", 
            ["All Prompts", "Specific IDs"],
            horizontal=True
        )
        
        ids_input = ""
        if re_embed_mode == "Specific IDs":
            ids_input = st.text_input(
                "Enter Prompt IDs (comma-separated)", 
                placeholder="1, 2, 3"
            )
        
        if st.button("🔄 Start Re-embedding"):
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
                    progress = min(current / total, 1.0) if total > 0 else 1.0
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {current} of {total}...")
                
                try:
                    with st.spinner("Re-embedding..."):
                        count = db.re_embed_prompts(
                            selected_project,
                            ids_to_process,
                            progress_callback=progress_callback
                        )
                    st.success(f"Successfully re-embedded {count} prompt(s).")
                except Exception as e:
                    st.error(f"Error during re-embedding: {e}")
        
        st.markdown("---")
        
        # Export section
        st.markdown("### 📤 Export Project")
        st.write("Export this project to a portable SQLite database file.")
        
        export_path = st.text_input(
            "Export path", 
            value=f"exports/{selected_project}/{selected_project}.db"
        )
        
        if st.button("📤 Export"):
            try:
                # Create export directory if needed
                export_dir = Path(export_path).parent
                export_dir.mkdir(parents=True, exist_ok=True)
                
                with st.spinner("Exporting..."):
                    db.export_to_sqlite(selected_project, export_path)
                st.success(f"Project exported to {export_path}")
            except NotImplementedError:
                st.warning("Export is only available for PostgreSQL projects (already SQLite).")
            except Exception as e:
                st.error(f"Export failed: {e}")
        
        st.markdown("---")
        
        # Delete prompts section
        st.markdown("### 🗑️ Delete Q&A Pairs")
        st.write("Delete specific Q&A pairs by ID.")
        
        delete_ids = st.text_input(
            "IDs to delete (comma-separated)",
            placeholder="1, 2, 3",
            key="delete_ids"
        )
        
        if st.button("🗑️ Delete Selected", type="secondary"):
            if not delete_ids.strip():
                st.warning("Please enter at least one ID to delete.")
            else:
                try:
                    ids = [int(x.strip()) for x in delete_ids.split(",") if x.strip()]
                    
                    if st.session_state.get('confirm_delete') != delete_ids:
                        st.session_state['confirm_delete'] = delete_ids
                        st.warning(f"Click again to confirm deletion of {len(ids)} Q&A pair(s).")
                    else:
                        deleted = 0
                        for qa_id in ids:
                            try:
                                db.delete_qa_pair(selected_project, qa_id)
                                deleted += 1
                            except Exception as e:
                                st.error(f"Error deleting ID {qa_id}: {e}")
                        
                        st.success(f"Deleted {deleted} Q&A pair(s).")
                        st.session_state.pop('confirm_delete', None)
                        st.rerun()
                        
                except ValueError:
                    st.error("Invalid input. Please enter numbers separated by commas.")

    # ==========================================================================
    # ADD DATA TAB
    # ==========================================================================
    with tab_add:
        st.subheader("Add New Q&A Pair")
        
        new_prompt = st.text_area(
            "Prompt",
            height=100,
            placeholder="Enter the prompt/question..."
        )
        
        new_response = st.text_area(
            "Response",
            height=200,
            placeholder="Enter the response/answer..."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            new_weight = st.number_input(
                "Weight",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                help="Higher weight = higher priority in search results"
            )
        
        if st.button("➕ Add Q&A Pair", type="primary"):
            if not new_prompt.strip():
                st.warning("Please enter a prompt.")
            elif not new_response.strip():
                st.warning("Please enter a response.")
            else:
                try:
                    from src import embedding as emb_module
                    
                    with st.spinner("Generating embedding..."):
                        # Get embedding model for this project
                        metadata = db.get_project_metadata(selected_project)
                        model_name = metadata.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                        
                        # Generate embedding
                        prompt_embedding = emb_module.generate_embedding(new_prompt, model_name)
                        
                        # Add to database
                        qa_id = db.add_qa_pair(
                            selected_project,
                            new_prompt.strip(),
                            new_response.strip(),
                            prompt_embedding,
                            new_weight
                        )
                    
                    st.success(f"Added Q&A pair with ID: {qa_id}")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Failed to add Q&A pair: {e}")
        
        st.markdown("---")
        
        # Bulk import section
        st.markdown("### 📥 Bulk Import")
        st.write("Import Q&A pairs from a JSON file.")
        
        uploaded_file = st.file_uploader(
            "Upload JSON file",
            type=['json'],
            help="Expected format: [{\"prompt\": \"...\", \"response\": \"...\", \"weight\": 1.0}, ...]"
        )
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                
                if not isinstance(data, list):
                    st.error("JSON must contain a list of Q&A pairs.")
                else:
                    st.info(f"Found {len(data)} Q&A pairs in file.")
                    
                    # Preview
                    if data:
                        st.write("Preview (first 3):")
                        for item in data[:3]:
                            st.json(item)
                    
                    if st.button("📥 Import All"):
                        from src import embedding as emb_module
                        
                        metadata = db.get_project_metadata(selected_project)
                        model_name = metadata.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        added = 0
                        errors = 0
                        
                        for i, item in enumerate(data):
                            try:
                                prompt = item.get('prompt', '')
                                response = item.get('response', '')
                                weight = item.get('weight', 1.0)
                                
                                if prompt and response:
                                    embedding = emb_module.generate_embedding(prompt, model_name)
                                    db.add_qa_pair(selected_project, prompt, response, embedding, weight)
                                    added += 1
                                    
                            except Exception as e:
                                errors += 1
                            
                            progress = (i + 1) / len(data)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {i + 1} of {len(data)}...")
                        
                        st.success(f"Imported {added} Q&A pairs. Errors: {errors}")
                        
            except json.JSONDecodeError:
                st.error("Invalid JSON file.")


if __name__ == "__main__":
    main()
