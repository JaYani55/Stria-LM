#!/usr/bin/env python3
"""
Stria-LM Project Manager GUI - Setup and Configuration
A Tkinter-based GUI for managing projects, environment variables, and database configuration.
"""
from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk, filedialog
from typing import Dict, List, Optional
import threading
import subprocess
import sys

from src.config import (
    PROJECTS_DIR,
    DATABASE_TYPE,
    EMBEDDING_MODELS,
    ENV_PATH,
    get_env_variables,
    set_env_variable,
    create_default_env,
    load_config,
)
from src.database import get_database


class ProjectManagerApp(tk.Tk):
    """Main application window for project setup and configuration."""

    def __init__(self):
        super().__init__()
        self.title("Stria-LM Project Manager")
        self.geometry("900x650")
        self.minsize(700, 500)
        
        # Ensure .env file exists
        if not ENV_PATH.exists():
            create_default_env()
            messagebox.showinfo(
                "Environment Created", 
                f"A new .env file has been created at:\n{ENV_PATH}\n\n"
                "Please configure your environment variables in the Settings tab."
            )
        
        self._build_ui()
        self._refresh_projects()
    
    def _build_ui(self):
        """Build the main UI with notebook tabs."""
        # Menu bar
        self._build_menu()
        
        # Main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._build_projects_tab()
        self._build_env_settings_tab()
        self._build_database_config_tab()
        self._build_about_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def _build_menu(self):
        """Build the menu bar."""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="New Project...", command=self._show_create_project_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Open Data Manager", command=self._open_streamlit)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=False)
        tools_menu.add_command(label="Start API Server", command=self._start_api_server)
        tools_menu.add_command(label="Export Project...", command=self._show_export_dialog)
        tools_menu.add_command(label="Import Project...", command=self._show_import_dialog)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=lambda: self.notebook.select(3))
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
    
    # =========================================================================
    # PROJECTS TAB
    # =========================================================================
    
    def _build_projects_tab(self):
        """Build the Projects management tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Projects")
        
        # Left panel - project list
        left_frame = ttk.LabelFrame(tab, text="Existing Projects", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Project listbox with scrollbar
        list_container = ttk.Frame(left_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.projects_listbox = tk.Listbox(list_container, height=15)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.projects_listbox.yview)
        self.projects_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.projects_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.projects_listbox.bind("<<ListboxSelect>>", self._on_project_select)
        
        # Project action buttons
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="New Project", command=self._show_create_project_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_projects).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Delete", command=self._delete_selected_project).pack(side=tk.LEFT)
        
        # Right panel - project details
        right_frame = ttk.LabelFrame(tab, text="Project Details", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.project_details_text = tk.Text(right_frame, wrap=tk.WORD, height=15, state=tk.DISABLED)
        self.project_details_text.pack(fill=tk.BOTH, expand=True)
        
        # Open in Data Manager button
        ttk.Button(right_frame, text="Open in Data Manager", command=self._open_project_in_streamlit).pack(pady=(10, 0))
    
    def _refresh_projects(self):
        """Refresh the list of projects."""
        self.projects_listbox.delete(0, tk.END)
        
        try:
            db = get_database()
            projects = db.list_projects()
            
            for project in projects:
                self.projects_listbox.insert(tk.END, project)
            
            self.status_var.set(f"Found {len(projects)} project(s) [{DATABASE_TYPE}]")
        except Exception as e:
            self.status_var.set(f"Error loading projects: {e}")
    
    def _on_project_select(self, event):
        """Handle project selection."""
        selection = self.projects_listbox.curselection()
        if not selection:
            return
        
        project_name = self.projects_listbox.get(selection[0])
        self._show_project_details(project_name)
    
    def _show_project_details(self, project_name: str):
        """Display details about a project."""
        self.project_details_text.config(state=tk.NORMAL)
        self.project_details_text.delete("1.0", tk.END)
        
        try:
            db = get_database()
            metadata = db.get_project_metadata(project_name)
            qa_pairs = db.get_qa_pairs(project_name, limit=5)
            
            details = f"Project: {project_name}\n"
            details += "=" * 40 + "\n\n"
            
            if metadata:
                details += f"Embedding Model: {metadata.get('embedding_model', 'N/A')}\n"
                details += f"Vector Dimension: {metadata.get('vector_dim', 'N/A')}\n"
                details += f"Created At: {metadata.get('created_at', 'N/A')}\n\n"
            
            details += f"Database Type: {DATABASE_TYPE}\n\n"
            
            details += f"Sample Q&A Pairs ({len(qa_pairs)} shown):\n"
            details += "-" * 40 + "\n"
            
            for i, qa in enumerate(qa_pairs[:5], 1):
                prompt = qa.get("prompt", "")[:100]
                response = qa.get("response", "")[:100]
                details += f"\n{i}. Prompt: {prompt}...\n"
                details += f"   Response: {response}...\n"
            
            self.project_details_text.insert("1.0", details)
        except Exception as e:
            self.project_details_text.insert("1.0", f"Error loading project details:\n{e}")
        
        self.project_details_text.config(state=tk.DISABLED)
    
    def _show_create_project_dialog(self):
        """Show dialog to create a new project."""
        dialog = CreateProjectDialog(self)
        if dialog.result:
            self._create_project_with_progress(**dialog.result)
    
    def _create_project_with_progress(self, name: str, embedding_model: str):
        """Create a new project with a progress dialog."""
        # Create progress dialog
        progress_dialog = ProjectCreationProgress(self, name)
        
        def create_task():
            try:
                from src import embedding as emb_module
                
                # Step 1: Initialize
                progress_dialog.update_step("Initializing...", 0)
                db = get_database()
                
                # Step 2: Check if project exists
                progress_dialog.update_step("Checking project name...", 20)
                if db.get_project(name):
                    progress_dialog.complete(f"Project '{name}' already exists.", is_error=True)
                    return
                
                # Step 3: Load embedding model
                progress_dialog.update_step("Loading embedding model...", 40)
                
                # Step 4: Get vector dimension (this loads the model)
                progress_dialog.update_step("Calculating vector dimensions...", 60)
                vector_dim = emb_module.get_vector_dimension(embedding_model)
                
                # Step 5: Create the project
                progress_dialog.update_step("Creating project database...", 80)
                db.create_project(name, embedding_model, vector_dim)
                
                # Step 6: Complete
                progress_dialog.update_step("Finalizing...", 95)
                
                # Success
                progress_dialog.complete(f"Project '{name}' created successfully!")
                
                # Refresh the project list (schedule on main thread)
                self.after(100, self._refresh_projects)
                
            except Exception as e:
                progress_dialog.complete(f"Failed to create project: {e}", is_error=True)
        
        # Run in background thread
        threading.Thread(target=create_task, daemon=True).start()
    
    def _create_project(self, name: str, embedding_model: str):
        """Create a new project (legacy method without progress)."""
        try:
            from src import embedding as emb_module
            
            db = get_database()
            
            # Check if project exists
            if db.get_project(name):
                messagebox.showerror("Error", f"Project '{name}' already exists.")
                return
            
            # Get vector dimension
            vector_dim = emb_module.get_vector_dimension(embedding_model)
            
            # Create the project
            db.create_project(name, embedding_model, vector_dim)
            
            self._refresh_projects()
            self.status_var.set(f"Project '{name}' created successfully.")
            messagebox.showinfo("Success", f"Project '{name}' has been created.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create project: {e}")
    
    def _delete_selected_project(self):
        """Delete the selected project."""
        selection = self.projects_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a project to delete.")
            return
        
        project_name = self.projects_listbox.get(selection[0])
        
        if not messagebox.askyesno("Confirm Delete", 
                                   f"Are you sure you want to delete project '{project_name}'?\n\n"
                                   "This action cannot be undone."):
            return
        
        try:
            db = get_database()
            db.delete_project(project_name)
            self._refresh_projects()
            self.status_var.set(f"Project '{project_name}' deleted.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete project: {e}")
    
    def _open_project_in_streamlit(self):
        """Open the selected project in Streamlit data manager."""
        selection = self.projects_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a project first.")
            return
        
        project_name = self.projects_listbox.get(selection[0])
        self._open_streamlit(project_name)
    
    # =========================================================================
    # ENVIRONMENT SETTINGS TAB
    # =========================================================================
    
    def _build_env_settings_tab(self):
        """Build the Environment Variables settings tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Environment Variables")
        
        # Instructions
        ttk.Label(tab, text="Configure environment variables for API keys and database connections.",
                  font=("Segoe UI", 9, "italic")).pack(anchor=tk.W, pady=(0, 10))
        
        # Scrollable frame for env vars
        canvas = tk.Canvas(tab)
        scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
        self.env_frame = ttk.Frame(canvas)
        
        self.env_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.env_frame, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store entry widgets
        self.env_entries: Dict[str, ttk.Entry] = {}
        
        self._populate_env_settings()
        
        # Button frame at bottom
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Save Changes", command=self._save_env_settings).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Reload", command=self._populate_env_settings).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Open .env File", command=self._open_env_file).pack(side=tk.RIGHT)
    
    def _populate_env_settings(self):
        """Populate the environment settings form."""
        # Clear existing widgets
        for widget in self.env_frame.winfo_children():
            widget.destroy()
        self.env_entries.clear()
        
        # Define environment variable groups
        env_groups = {
            "Database Configuration": [
                ("DATABASE_TYPE", "Database type (sqlite or postgresql)"),
                ("DATABASE_URL", "PostgreSQL connection URL"),
            ],
            "API Keys": [
                ("OPENROUTER_API_KEY", "OpenRouter API key for LLM access"),
                ("OPENAI_API_KEY", "OpenAI API key (optional)"),
            ],
            "Experimental Features": [
                ("ENABLE_WEIGHTS", "Enable weighted prompts (true/false)"),
            ]
        }
        
        env_vars = get_env_variables()
        
        row = 0
        for group_name, variables in env_groups.items():
            # Group header
            ttk.Label(self.env_frame, text=group_name, font=("Segoe UI", 10, "bold")).grid(
                row=row, column=0, columnspan=2, sticky=tk.W, pady=(10, 5)
            )
            row += 1
            
            for var_name, description in variables:
                # Label
                ttk.Label(self.env_frame, text=var_name).grid(row=row, column=0, sticky=tk.W, padx=(10, 5), pady=2)
                
                # Entry
                entry = ttk.Entry(self.env_frame, width=50)
                entry.grid(row=row, column=1, sticky=tk.W, pady=2)
                entry.insert(0, env_vars.get(var_name, ""))
                
                # Mask sensitive values
                if "KEY" in var_name or "PASSWORD" in var_name:
                    entry.config(show="*")
                
                self.env_entries[var_name] = entry
                row += 1
                
                # Description
                ttk.Label(self.env_frame, text=description, font=("Segoe UI", 8, "italic"), 
                         foreground="gray").grid(row=row, column=1, sticky=tk.W, pady=(0, 5))
                row += 1
    
    def _save_env_settings(self):
        """Save environment variable changes."""
        try:
            for var_name, entry in self.env_entries.items():
                value = entry.get()
                set_env_variable(var_name, value)
            
            self.status_var.set("Environment variables saved. Restart may be required for changes to take effect.")
            messagebox.showinfo("Saved", "Environment variables have been saved.\n\n"
                               "Some changes may require restarting the application.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save environment variables: {e}")
    
    def _open_env_file(self):
        """Open the .env file in default text editor."""
        if sys.platform == "win32":
            os.startfile(str(ENV_PATH))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(ENV_PATH)])
        else:
            subprocess.run(["xdg-open", str(ENV_PATH)])
    
    # =========================================================================
    # DATABASE CONFIG TAB
    # =========================================================================
    
    def _build_database_config_tab(self):
        """Build the Database Configuration tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Database")
        
        # Current config display
        config_frame = ttk.LabelFrame(tab, text="Current Configuration", padding=10)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.db_type_var = tk.StringVar(value=DATABASE_TYPE)
        ttk.Label(config_frame, text="Database Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(config_frame, textvariable=self.db_type_var, font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(config_frame, text="Projects Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(config_frame, text=str(PROJECTS_DIR)).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Switch database type
        switch_frame = ttk.LabelFrame(tab, text="Switch Database Type", padding=10)
        switch_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(switch_frame, text="To switch database type, update DATABASE_TYPE in the Environment Variables tab.").pack(anchor=tk.W)
        ttk.Label(switch_frame, text="• sqlite - Local file-based databases (one per project)", 
                 font=("Segoe UI", 9)).pack(anchor=tk.W, padx=(10, 0))
        ttk.Label(switch_frame, text="• postgresql - Centralized PostgreSQL server with pgvector", 
                 font=("Segoe UI", 9)).pack(anchor=tk.W, padx=(10, 0))
        
        # Import/Export section
        io_frame = ttk.LabelFrame(tab, text="Import / Export", padding=10)
        io_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(io_frame, text="Export projects to portable SQLite files or import from existing databases.").pack(anchor=tk.W)
        
        io_btn_frame = ttk.Frame(io_frame)
        io_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(io_btn_frame, text="Export Project...", command=self._show_export_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(io_btn_frame, text="Import Project...", command=self._show_import_dialog).pack(side=tk.LEFT)
        
        # PostgreSQL setup guide
        pg_frame = ttk.LabelFrame(tab, text="PostgreSQL Setup Guide", padding=10)
        pg_frame.pack(fill=tk.BOTH, expand=True)
        
        pg_text = tk.Text(pg_frame, wrap=tk.WORD, height=10, state=tk.NORMAL)
        pg_text.pack(fill=tk.BOTH, expand=True)
        
        pg_guide = """PostgreSQL with pgvector Setup Instructions:

1. Install PostgreSQL from https://www.postgresql.org/download/

2. Create a database:
   CREATE DATABASE strialm;

3. Install the pgvector extension:
   CREATE EXTENSION vector;

4. Set your connection URL in Environment Variables:
   DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/strialm

5. Set DATABASE_TYPE to 'postgresql'

6. Restart the application
"""
        pg_text.insert("1.0", pg_guide)
        pg_text.config(state=tk.DISABLED)
    
    def _show_export_dialog(self):
        """Show dialog to export a project."""
        selection = self.projects_listbox.curselection()
        project_name = self.projects_listbox.get(selection[0]) if selection else None
        
        if not project_name:
            # Ask user to select a project
            db = get_database()
            projects = db.list_projects()
            if not projects:
                messagebox.showwarning("No Projects", "No projects available to export.")
                return
            
            dialog = SelectProjectDialog(self, projects, "Export Project")
            project_name = dialog.result
        
        if not project_name:
            return
        
        # Ask for output path
        output_path = filedialog.asksaveasfilename(
            title="Export Project",
            defaultextension=".db",
            initialfile=f"{project_name}.db",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        
        if not output_path:
            return
        
        try:
            db = get_database()
            db.export_to_sqlite(project_name, output_path)
            self.status_var.set(f"Project exported to {output_path}")
            messagebox.showinfo("Export Complete", f"Project '{project_name}' has been exported to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export project: {e}")
    
    def _show_import_dialog(self):
        """Show dialog to import a project."""
        # Ask for input file
        input_path = filedialog.askopenfilename(
            title="Import Project",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        
        if not input_path:
            return
        
        # Ask for project name
        dialog = TextInputDialog(self, "Import Project", "Enter a name for the imported project:")
        project_name = dialog.result
        
        if not project_name:
            return
        
        try:
            db = get_database()
            
            if db.get_project(project_name):
                messagebox.showerror("Error", f"Project '{project_name}' already exists.")
                return
            
            db.import_from_sqlite(project_name, input_path)
            self._refresh_projects()
            self.status_var.set(f"Project imported from {input_path}")
            messagebox.showinfo("Import Complete", f"Project '{project_name}' has been imported successfully.")
        except Exception as e:
            messagebox.showerror("Import Failed", f"Failed to import project: {e}")
    
    # =========================================================================
    # ABOUT TAB
    # =========================================================================
    
    def _build_about_tab(self):
        """Build the About tab."""
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="About")
        
        ttk.Label(tab, text="Stria-LM", font=("Segoe UI", 24, "bold")).pack(pady=(20, 5))
        ttk.Label(tab, text="Portable File-Based Retrieval Models", font=("Segoe UI", 12)).pack(pady=(0, 20))
        
        ttk.Label(tab, text="Version 0.2.0").pack()
        ttk.Label(tab, text="Developed by Pluracon").pack(pady=(5, 20))
        
        info_text = """
Stria-LM is a framework for creating and managing portable, 
file-based retrieval models for semantic search and chatbot applications.

Features:
• Flexible database backends (SQLite or PostgreSQL)
• Semantic search using vector embeddings
• Web scraping and auto-generation of Q&A pairs
• Export/import projects as portable SQLite databases
• RESTful API for integration
        """
        
        ttk.Label(tab, text=info_text, justify=tk.LEFT).pack(pady=10)
        
        link_frame = ttk.Frame(tab)
        link_frame.pack(pady=20)
        
        ttk.Button(link_frame, text="Open Data Manager", command=self._open_streamlit).pack(side=tk.LEFT, padx=5)
        ttk.Button(link_frame, text="Start API Server", command=self._start_api_server).pack(side=tk.LEFT, padx=5)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _open_streamlit(self, project_name: str = None):
        """Open the Streamlit data manager."""
        cmd = [sys.executable, "-m", "streamlit", "run", "streamlit_project_manager.py"]
        if project_name:
            cmd.extend(["--", "--project", project_name])
        
        try:
            subprocess.Popen(cmd, cwd=Path(__file__).parent)
            self.status_var.set("Streamlit data manager launched.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch Streamlit: {e}")
    
    def _start_api_server(self):
        """Start the FastAPI server."""
        cmd = [sys.executable, "-m", "uvicorn", "src.main:app", "--reload"]
        
        try:
            subprocess.Popen(cmd, cwd=Path(__file__).parent)
            self.status_var.set("API server started on http://localhost:8000")
            messagebox.showinfo("Server Started", "API server is starting on http://localhost:8000\n\n"
                               "Check the terminal for logs.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start API server: {e}")


# =============================================================================
# DIALOG CLASSES
# =============================================================================

class ProjectCreationProgress(tk.Toplevel):
    """Progress dialog for project creation."""
    
    def __init__(self, master, project_name: str):
        super().__init__(master)
        self.title("Creating Project")
        self.geometry("400x180")
        self.resizable(False, False)
        self.transient(master)
        
        # Prevent closing during operation
        self.protocol("WM_DELETE_WINDOW", lambda: None)
        
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(
            main_frame, 
            text=f"Creating project: {project_name}",
            font=("Segoe UI", 11, "bold")
        ).pack(pady=(0, 15))
        
        # Status text
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var)
        self.status_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progressbar = ttk.Progressbar(
            main_frame, 
            orient="horizontal", 
            length=350, 
            mode="determinate"
        )
        self.progressbar.pack(pady=(0, 15))
        
        # Close button (initially hidden)
        self.close_btn = ttk.Button(main_frame, text="Close", command=self.destroy)
        
        self.grab_set()
        self.update_idletasks()
        
        # Center on parent
        self.geometry(f"+{master.winfo_x() + 50}+{master.winfo_y() + 100}")
    
    def update_step(self, message: str, progress: int):
        """Update the progress display."""
        if not self.winfo_exists():
            return
        self.status_var.set(message)
        self.progressbar["value"] = progress
        self.update_idletasks()
    
    def complete(self, message: str, is_error: bool = False):
        """Complete the progress dialog."""
        if not self.winfo_exists():
            return
        
        self.status_var.set(message)
        self.progressbar["value"] = 100
        
        # Allow closing now
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        
        # Show close button
        self.close_btn.pack()
        
        # Show message box
        if is_error:
            messagebox.showerror("Error", message, parent=self)
        else:
            messagebox.showinfo("Success", message, parent=self)
        
        self.update_idletasks()


class CreateProjectDialog(tk.Toplevel):
    """Dialog for creating a new project."""
    
    def __init__(self, master):
        super().__init__(master)
        self.title("Create New Project")
        self.geometry("450x250")
        self.resizable(False, False)
        self.transient(master)
        
        self.result = None
        
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Project name
        ttk.Label(main_frame, text="Project Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_entry = ttk.Entry(main_frame, width=40)
        self.name_entry.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Embedding model
        ttk.Label(main_frame, text="Embedding Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(main_frame, textvariable=self.model_var, width=37, state="readonly")
        self.model_combo["values"] = list(EMBEDDING_MODELS.keys())
        if self.model_combo["values"]:
            self.model_combo.current(0)
        self.model_combo.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Model info
        self.model_info_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.model_info_var, font=("Segoe UI", 8, "italic"),
                 foreground="gray").grid(row=2, column=1, sticky=tk.W)
        self.model_combo.bind("<<ComboboxSelected>>", self._update_model_info)
        self._update_model_info()
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(20, 0))
        
        ttk.Button(btn_frame, text="Create", command=self._on_create).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        self.grab_set()
        self.name_entry.focus_set()
        self.wait_window(self)
    
    def _update_model_info(self, event=None):
        model = self.model_var.get()
        if model in EMBEDDING_MODELS:
            info = EMBEDDING_MODELS[model]
            # Get model path/name and category
            model_path = info.get('model', 'Unknown')
            category = info.get('category', 'Unknown')
            self.model_info_var.set(f"Model: {model_path} | Category: {category}")
    
    def _on_create(self):
        name = self.name_entry.get().strip()
        model = self.model_var.get()
        
        if not name:
            messagebox.showwarning("Invalid Name", "Please enter a project name.", parent=self)
            return
        
        # Sanitize name (remove special characters)
        import re
        if not re.match(r'^[\w\-]+$', name):
            messagebox.showwarning("Invalid Name", 
                                   "Project name can only contain letters, numbers, underscores, and hyphens.", 
                                   parent=self)
            return
        
        if not model:
            messagebox.showwarning("No Model", "Please select an embedding model.", parent=self)
            return
        
        self.result = {"name": name, "embedding_model": model}
        self.destroy()


class SelectProjectDialog(tk.Toplevel):
    """Dialog for selecting a project from a list."""
    
    def __init__(self, master, projects: List[str], title: str = "Select Project"):
        super().__init__(master)
        self.title(title)
        self.geometry("300x350")
        self.resizable(False, False)
        self.transient(master)
        
        self.result = None
        
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Select a project:").pack(anchor=tk.W)
        
        self.listbox = tk.Listbox(main_frame, height=12)
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=10)
        
        for project in projects:
            self.listbox.insert(tk.END, project)
        
        if projects:
            self.listbox.selection_set(0)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Select", command=self._on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        self.grab_set()
        self.wait_window(self)
    
    def _on_select(self):
        selection = self.listbox.curselection()
        if selection:
            self.result = self.listbox.get(selection[0])
            self.destroy()


class TextInputDialog(tk.Toplevel):
    """Simple dialog for text input."""
    
    def __init__(self, master, title: str, prompt: str):
        super().__init__(master)
        self.title(title)
        self.geometry("350x120")
        self.resizable(False, False)
        self.transient(master)
        
        self.result = None
        
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text=prompt).pack(anchor=tk.W)
        
        self.entry = ttk.Entry(main_frame, width=40)
        self.entry.pack(fill=tk.X, pady=10)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        self.grab_set()
        self.entry.focus_set()
        self.bind("<Return>", lambda e: self._on_ok())
        self.wait_window(self)
    
    def _on_ok(self):
        self.result = self.entry.get().strip()
        if self.result:
            self.destroy()


if __name__ == "__main__":
    app = ProjectManagerApp()
    app.mainloop()
