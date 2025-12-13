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
    save_config,
    CONFIG_PATH,
    set_current_project,
    set_script_running,
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
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        
        # Create tabs
        self._build_projects_tab()
        self._build_scripts_tab()
        self._build_env_settings_tab()
        self._build_inference_config_tab()
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
        help_menu.add_command(label="About", command=lambda: self.notebook.select(4))
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
            
            # Also refresh scripts tab project combo
            self._refresh_scripts_projects()
        except Exception as e:
            self.status_var.set(f"Error loading projects: {e}")
    
    def _on_tab_changed(self, event):
        """Handle notebook tab changes."""
        tab_id = self.notebook.select()
        tab_text = self.notebook.tab(tab_id, "text")
        
        # Refresh scripts projects when Scripts tab is selected
        if "Scripts" in tab_text:
            self._refresh_scripts_projects()
    
    def _on_project_select(self, event):
        """Handle project selection."""
        selection = self.projects_listbox.curselection()
        if not selection:
            return
        
        project_name = self.projects_listbox.get(selection[0])
        self._show_project_details(project_name)
        
        # Update state file for Streamlit to know which project is selected
        try:
            set_current_project(project_name)
        except Exception:
            pass  # Don't break on state file errors
    
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
    # SCRIPTS TAB
    # =========================================================================
    
    def _build_scripts_tab(self):
        """Build the Scripts management tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="📜 Scripts")
        
        # Top frame - project and script type selection
        top_frame = ttk.Frame(tab)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(top_frame, text="Project:").pack(side=tk.LEFT)
        self.scripts_project_var = tk.StringVar()
        self.scripts_project_combo = ttk.Combobox(
            top_frame, 
            textvariable=self.scripts_project_var,
            state="readonly",
            width=25
        )
        self.scripts_project_combo.pack(side=tk.LEFT, padx=(5, 20))
        self.scripts_project_combo.bind("<<ComboboxSelected>>", self._on_scripts_project_change)
        
        ttk.Label(top_frame, text="Type:").pack(side=tk.LEFT)
        self.script_type_var = tk.StringVar(value="all")
        script_types = ["all", "scraper", "data-manipulation", "ai-script", "migration"]
        self.script_type_combo = ttk.Combobox(
            top_frame,
            textvariable=self.script_type_var,
            values=script_types,
            state="readonly",
            width=18
        )
        self.script_type_combo.pack(side=tk.LEFT, padx=5)
        self.script_type_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_scripts_list())
        
        # Main content - split pane
        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - script list
        left_frame = ttk.LabelFrame(paned, text="Registered Scripts", padding=10)
        paned.add(left_frame, weight=1)
        
        # Scripts listbox
        list_container = ttk.Frame(left_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        self.scripts_listbox = tk.Listbox(list_container, height=15, width=35)
        scripts_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.scripts_listbox.yview)
        self.scripts_listbox.configure(yscrollcommand=scripts_scrollbar.set)
        
        self.scripts_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scripts_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.scripts_listbox.bind("<<ListboxSelect>>", self._on_script_select)
        
        # Script list buttons
        list_btn_frame = ttk.Frame(left_frame)
        list_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(list_btn_frame, text="New Script", command=self._show_new_script_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(list_btn_frame, text="Refresh", command=self._refresh_scripts_list).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(list_btn_frame, text="Delete", command=self._delete_selected_script).pack(side=tk.LEFT)
        
        # Right panel - script editor
        right_frame = ttk.LabelFrame(paned, text="Script Editor", padding=10)
        paned.add(right_frame, weight=2)
        
        # Script info
        info_frame = ttk.Frame(right_frame)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(info_frame, text="Name:").grid(row=0, column=0, sticky=tk.W)
        self.script_name_var = tk.StringVar()
        self.script_name_entry = ttk.Entry(info_frame, textvariable=self.script_name_var, width=30)
        self.script_name_entry.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(info_frame, text="Type:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.edit_script_type_var = tk.StringVar()
        self.edit_script_type_combo = ttk.Combobox(
            info_frame,
            textvariable=self.edit_script_type_var,
            values=["scraper", "data-manipulation", "ai-script", "migration"],
            state="readonly",
            width=15
        )
        self.edit_script_type_combo.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        ttk.Label(info_frame, text="Description:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.script_desc_var = tk.StringVar()
        self.script_desc_entry = ttk.Entry(info_frame, textvariable=self.script_desc_var, width=60)
        self.script_desc_entry.grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=(5, 0))
        
        # Code editor
        editor_label = ttk.Label(right_frame, text="Code:")
        editor_label.pack(anchor=tk.W, pady=(10, 2))
        
        editor_container = ttk.Frame(right_frame)
        editor_container.pack(fill=tk.BOTH, expand=True)
        
        self.script_editor = tk.Text(
            editor_container, 
            wrap=tk.NONE, 
            font=("Consolas", 10),
            undo=True
        )
        editor_vscroll = ttk.Scrollbar(editor_container, orient=tk.VERTICAL, command=self.script_editor.yview)
        editor_hscroll = ttk.Scrollbar(editor_container, orient=tk.HORIZONTAL, command=self.script_editor.xview)
        self.script_editor.configure(yscrollcommand=editor_vscroll.set, xscrollcommand=editor_hscroll.set)
        
        self.script_editor.grid(row=0, column=0, sticky="nsew")
        editor_vscroll.grid(row=0, column=1, sticky="ns")
        editor_hscroll.grid(row=1, column=0, sticky="ew")
        
        editor_container.grid_rowconfigure(0, weight=1)
        editor_container.grid_columnconfigure(0, weight=1)
        
        # Editor buttons
        editor_btn_frame = ttk.Frame(right_frame)
        editor_btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(editor_btn_frame, text="Save Script", command=self._save_current_script).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(editor_btn_frame, text="Run Script", command=self._run_current_script).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(editor_btn_frame, text="Open in Editor", command=self._open_script_external).pack(side=tk.LEFT)
        
        # Store current script info
        self._current_script_id = None
        self._current_script_path = None
    
    def _on_scripts_project_change(self, event=None):
        """Handle project selection change in scripts tab."""
        project_name = self.scripts_project_var.get()
        if project_name:
            try:
                set_current_project(project_name)
            except Exception:
                pass
        self._refresh_scripts_list()
    
    def _refresh_scripts_list(self):
        """Refresh the list of scripts for the selected project."""
        self.scripts_listbox.delete(0, tk.END)
        
        project_name = self.scripts_project_var.get()
        if not project_name:
            return
        
        try:
            db = get_database()
            script_type = self.script_type_var.get()
            script_type = None if script_type == "all" else script_type
            
            scripts = db.list_scripts(project_name, script_type=script_type, enabled_only=False)
            
            self._scripts_cache = {s["id"]: s for s in scripts}
            
            for script in scripts:
                prefix = "✓ " if script["is_enabled"] else "✗ "
                type_abbrev = {"scraper": "SCR", "data-manipulation": "DM", "ai-script": "AI", "migration": "MIG"}.get(script["script_type"], "???")
                self.scripts_listbox.insert(tk.END, f"{prefix}[{type_abbrev}] {script['script_name']}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load scripts: {e}")
    
    def _on_script_select(self, event=None):
        """Handle script selection."""
        selection = self.scripts_listbox.curselection()
        if not selection:
            return
        
        project_name = self.scripts_project_var.get()
        if not project_name:
            return
        
        # Get script ID from cache
        idx = selection[0]
        script_ids = list(self._scripts_cache.keys())
        if idx >= len(script_ids):
            return
        
        script_id = script_ids[idx]
        script = self._scripts_cache.get(script_id)
        
        if not script:
            return
        
        self._current_script_id = script_id
        self.script_name_var.set(script["script_name"])
        self.edit_script_type_var.set(script["script_type"])
        self.script_desc_var.set(script.get("description") or "")
        
        # Load script content
        try:
            db = get_database()
            scripts_dir = db.get_scripts_directory(project_name)
            if scripts_dir:
                script_path = Path(scripts_dir) / script["file_path"]
                self._current_script_path = script_path
                
                if script_path.exists():
                    content = script_path.read_text(encoding="utf-8")
                    self.script_editor.delete("1.0", tk.END)
                    self.script_editor.insert("1.0", content)
                else:
                    self.script_editor.delete("1.0", tk.END)
                    self.script_editor.insert("1.0", f"# Script file not found: {script_path}")
        except Exception as e:
            self.script_editor.delete("1.0", tk.END)
            self.script_editor.insert("1.0", f"# Error loading script: {e}")
    
    def _show_new_script_dialog(self):
        """Show dialog to create a new script."""
        project_name = self.scripts_project_var.get()
        if not project_name:
            messagebox.showwarning("No Project", "Please select a project first.")
            return
        
        dialog = NewScriptDialog(self, project_name)
        self.wait_window(dialog)
        
        if dialog.result:
            self._refresh_scripts_list()
            self.status_var.set(f"Script '{dialog.result['name']}' created.")
    
    def _delete_selected_script(self):
        """Delete the selected script."""
        if not self._current_script_id:
            messagebox.showwarning("No Selection", "Please select a script first.")
            return
        
        project_name = self.scripts_project_var.get()
        script = self._scripts_cache.get(self._current_script_id)
        
        if not messagebox.askyesno("Confirm Delete", 
                                    f"Delete script '{script['script_name']}'?\n\n"
                                    "This will remove the script registration.\n"
                                    "The script file will NOT be deleted."):
            return
        
        try:
            db = get_database()
            db.delete_script(project_name, self._current_script_id)
            self._current_script_id = None
            self._current_script_path = None
            self.script_editor.delete("1.0", tk.END)
            self._refresh_scripts_list()
            self.status_var.set("Script deleted.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete script: {e}")
    
    def _save_current_script(self):
        """Save the current script content and metadata."""
        if not self._current_script_path:
            messagebox.showwarning("No Script", "No script is currently loaded.")
            return
        
        project_name = self.scripts_project_var.get()
        
        try:
            # Save file content
            content = self.script_editor.get("1.0", tk.END)
            self._current_script_path.write_text(content, encoding="utf-8")
            
            # Update metadata
            if self._current_script_id:
                db = get_database()
                db.update_script(project_name, self._current_script_id, {
                    "script_name": self.script_name_var.get(),
                    "script_type": self.edit_script_type_var.get(),
                    "description": self.script_desc_var.get()
                })
            
            self.status_var.set(f"Script saved: {self._current_script_path.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save script: {e}")
    
    def _run_current_script(self):
        """Run the current script in a subprocess."""
        if not self._current_script_path or not self._current_script_path.exists():
            messagebox.showwarning("No Script", "Please select a valid script first.")
            return
        
        project_name = self.scripts_project_var.get()
        
        # Save before running
        self._save_current_script()
        
        # Create execution dialog
        dialog = ScriptExecutionDialog(
            self, 
            project_name, 
            self._current_script_id,
            self._current_script_path
        )
        self.wait_window(dialog)
    
    def _open_script_external(self):
        """Open the script in the system's default editor."""
        if not self._current_script_path or not self._current_script_path.exists():
            messagebox.showwarning("No Script", "No script file to open.")
            return
        
        try:
            import os
            os.startfile(str(self._current_script_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open script: {e}")
    
    def _refresh_scripts_projects(self):
        """Refresh the projects combo in scripts tab."""
        try:
            db = get_database()
            projects = db.list_projects()
            self.scripts_project_combo["values"] = projects
            
            current = self.scripts_project_var.get()
            # Reset selection if current project doesn't exist anymore
            if current and current not in projects:
                self.scripts_project_var.set("")
                current = ""
            
            # Select first project if none selected
            if projects and not current:
                self.scripts_project_var.set(projects[0])
                self._refresh_scripts_list()
        except Exception as e:
            self.status_var.set(f"Error loading projects: {e}")
    
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
    # INFERENCE CONFIG TAB
    # =========================================================================
    
    def _build_inference_config_tab(self):
        """Build the Inference Server configuration tab."""
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Inference")
        
        # Instructions
        ttk.Label(
            tab, 
            text="Configure LLM inference server connection for chat and content generation.",
            font=("Segoe UI", 9, "italic")
        ).pack(anchor=tk.W, pady=(0, 10))
        
        # Load current config
        current_config = load_config()
        inference_config = current_config.get("inference", {})
        
        # Connection Settings Frame
        conn_frame = ttk.LabelFrame(tab, text="Connection Settings", padding=10)
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Store entry widgets for inference settings
        self.inference_entries = {}
        
        # Base URL
        ttk.Label(conn_frame, text="Inference Server URL:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.inference_entries["base_url"] = ttk.Entry(conn_frame, width=50)
        self.inference_entries["base_url"].grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        self.inference_entries["base_url"].insert(0, inference_config.get("base_url", "http://localhost:8080/v1"))
        
        ttk.Label(
            conn_frame, 
            text="OpenAI-compatible API endpoint (e.g., http://localhost:8080/v1, https://openrouter.ai/api/v1)",
            font=("Segoe UI", 8, "italic"), foreground="gray"
        ).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # Default Model
        ttk.Label(conn_frame, text="Default Model:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.inference_entries["default_model"] = ttk.Entry(conn_frame, width=50)
        self.inference_entries["default_model"].grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        self.inference_entries["default_model"].insert(0, inference_config.get("default_model", "gpt-4"))
        
        ttk.Label(
            conn_frame, 
            text="Model name as expected by the inference server (e.g., gpt-4, openai/gpt-4.1-nano)",
            font=("Segoe UI", 8, "italic"), foreground="gray"
        ).grid(row=3, column=1, sticky=tk.W, padx=(10, 0))
        
        # API Key (from .env)
        ttk.Label(conn_frame, text="API Key:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.inference_entries["api_key"] = ttk.Entry(conn_frame, width=50, show="*")
        self.inference_entries["api_key"].grid(row=4, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Load API key from env
        env_vars = get_env_variables()
        api_key = env_vars.get("INFERENCE_API_KEY", "") or env_vars.get("OPENROUTER_API_KEY", "")
        self.inference_entries["api_key"].insert(0, api_key)
        
        ttk.Label(
            conn_frame, 
            text="API key for the inference server (stored securely in .env file)",
            font=("Segoe UI", 8, "italic"), foreground="gray"
        ).grid(row=5, column=1, sticky=tk.W, padx=(10, 0))
        
        # Show/Hide API key button
        self.show_api_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            conn_frame, 
            text="Show API Key", 
            variable=self.show_api_key_var,
            command=self._toggle_api_key_visibility
        ).grid(row=4, column=2, padx=(5, 0))
        
        # Local Inference Server Settings Frame
        local_frame = ttk.LabelFrame(tab, text="Local Inference Server Settings", padding=10)
        local_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Default Host
        ttk.Label(local_frame, text="Default Host:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.inference_entries["default_host"] = ttk.Entry(local_frame, width=30)
        self.inference_entries["default_host"].grid(row=0, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        self.inference_entries["default_host"].insert(0, inference_config.get("default_host", "127.0.0.1"))
        
        # Default Port
        ttk.Label(local_frame, text="Default Port:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.inference_entries["default_port"] = ttk.Entry(local_frame, width=10)
        self.inference_entries["default_port"].grid(row=1, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        self.inference_entries["default_port"].insert(0, str(inference_config.get("default_port", 8008)))
        
        # Inference Model Name
        ttk.Label(local_frame, text="Model Name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.inference_entries["inference_model_name"] = ttk.Entry(local_frame, width=40)
        self.inference_entries["inference_model_name"].grid(row=2, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        self.inference_entries["inference_model_name"].insert(0, inference_config.get("inference_model_name", "default-model"))
        
        # Inference Model Path
        ttk.Label(local_frame, text="Model Path:").grid(row=3, column=0, sticky=tk.W, pady=5)
        model_path_frame = ttk.Frame(local_frame)
        model_path_frame.grid(row=3, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        self.inference_entries["inference_model_path"] = ttk.Entry(model_path_frame, width=45)
        self.inference_entries["inference_model_path"].pack(side=tk.LEFT)
        self.inference_entries["inference_model_path"].insert(0, inference_config.get("inference_model_path", ""))
        
        ttk.Button(model_path_frame, text="Browse...", command=self._browse_model_path).pack(side=tk.LEFT, padx=(5, 0))
        
        ttk.Label(
            local_frame, 
            text="Path to local GGUF model file (for local inference servers)",
            font=("Segoe UI", 8, "italic"), foreground="gray"
        ).grid(row=4, column=1, sticky=tk.W, padx=(10, 0))
        
        # Preset buttons
        preset_frame = ttk.LabelFrame(tab, text="Quick Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            preset_frame, 
            text="Local (localhost:8080)", 
            command=lambda: self._apply_inference_preset("local")
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            preset_frame, 
            text="OpenRouter", 
            command=lambda: self._apply_inference_preset("openrouter")
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            preset_frame, 
            text="OpenAI", 
            command=lambda: self._apply_inference_preset("openai")
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            preset_frame, 
            text="LM Studio", 
            command=lambda: self._apply_inference_preset("lmstudio")
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Save button frame
        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="Save Configuration", command=self._save_inference_config).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Test Connection", command=self._test_inference_connection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Reload", command=self._reload_inference_config).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Open config.toml", command=self._open_config_file).pack(side=tk.RIGHT)
    
    def _toggle_api_key_visibility(self):
        """Toggle API key visibility."""
        if self.show_api_key_var.get():
            self.inference_entries["api_key"].config(show="")
        else:
            self.inference_entries["api_key"].config(show="*")
    
    def _browse_model_path(self):
        """Browse for a GGUF model file."""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")]
        )
        if file_path:
            self.inference_entries["inference_model_path"].delete(0, tk.END)
            self.inference_entries["inference_model_path"].insert(0, file_path)
    
    def _apply_inference_preset(self, preset: str):
        """Apply a preset inference configuration."""
        presets = {
            "local": {
                "base_url": "http://localhost:8080/v1",
                "default_model": "local-model"
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "default_model": "openai/gpt-4.1-nano"
            },
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "default_model": "gpt-4"
            },
            "lmstudio": {
                "base_url": "http://localhost:1234/v1",
                "default_model": "local-model"
            }
        }
        
        if preset in presets:
            config = presets[preset]
            self.inference_entries["base_url"].delete(0, tk.END)
            self.inference_entries["base_url"].insert(0, config["base_url"])
            self.inference_entries["default_model"].delete(0, tk.END)
            self.inference_entries["default_model"].insert(0, config["default_model"])
            self.status_var.set(f"Applied {preset} preset - remember to save!")
    
    def _save_inference_config(self):
        """Save inference configuration to config.toml and .env."""
        try:
            # Load current config
            current_config = load_config()
            
            # Update inference section
            if "inference" not in current_config:
                current_config["inference"] = {}
            
            # Save config.toml values
            current_config["inference"]["base_url"] = self.inference_entries["base_url"].get()
            current_config["inference"]["default_model"] = self.inference_entries["default_model"].get()
            current_config["inference"]["default_host"] = self.inference_entries["default_host"].get()
            
            try:
                current_config["inference"]["default_port"] = int(self.inference_entries["default_port"].get())
            except ValueError:
                current_config["inference"]["default_port"] = 8008
            
            current_config["inference"]["inference_model_name"] = self.inference_entries["inference_model_name"].get()
            current_config["inference"]["inference_model_path"] = self.inference_entries["inference_model_path"].get()
            
            # Save to config.toml
            save_config(current_config)
            
            # Save API key to .env (more secure)
            api_key = self.inference_entries["api_key"].get()
            if api_key:
                set_env_variable("INFERENCE_API_KEY", api_key)
                # Also set OPENROUTER_API_KEY if using OpenRouter
                if "openrouter" in self.inference_entries["base_url"].get().lower():
                    set_env_variable("OPENROUTER_API_KEY", api_key)
            
            self.status_var.set("Inference configuration saved successfully.")
            messagebox.showinfo(
                "Configuration Saved", 
                "Inference settings have been saved.\n\n"
                "You may need to restart the application or API server for changes to take effect."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def _reload_inference_config(self):
        """Reload inference configuration from files."""
        try:
            current_config = load_config()
            inference_config = current_config.get("inference", {})
            env_vars = get_env_variables()
            
            # Update entries
            for key in ["base_url", "default_model", "default_host", "inference_model_name", "inference_model_path"]:
                if key in self.inference_entries:
                    self.inference_entries[key].delete(0, tk.END)
                    self.inference_entries[key].insert(0, str(inference_config.get(key, "")))
            
            self.inference_entries["default_port"].delete(0, tk.END)
            self.inference_entries["default_port"].insert(0, str(inference_config.get("default_port", 8008)))
            
            # Reload API key from env
            api_key = env_vars.get("INFERENCE_API_KEY", "") or env_vars.get("OPENROUTER_API_KEY", "")
            self.inference_entries["api_key"].delete(0, tk.END)
            self.inference_entries["api_key"].insert(0, api_key)
            
            self.status_var.set("Inference configuration reloaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload configuration: {e}")
    
    def _test_inference_connection(self):
        """Test connection to the inference server."""
        base_url = self.inference_entries["base_url"].get().rstrip("/")
        api_key = self.inference_entries["api_key"].get()
        
        def test_task():
            try:
                import httpx
                
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                # Try to hit the models endpoint
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(f"{base_url}/models", headers=headers)
                    
                    if response.status_code == 200:
                        models = response.json()
                        model_count = len(models.get("data", []))
                        self.after(0, lambda: self._show_connection_success(model_count))
                    else:
                        self.after(0, lambda: self._show_connection_error(
                            f"Server returned status {response.status_code}"
                        ))
            except httpx.ConnectError as e:
                self.after(0, lambda: self._show_connection_error(
                    f"Could not connect to server.\n\nMake sure the inference server is running at:\n{base_url}"
                ))
            except Exception as e:
                self.after(0, lambda: self._show_connection_error(str(e)))
        
        self.status_var.set("Testing connection...")
        threading.Thread(target=test_task, daemon=True).start()
    
    def _show_connection_success(self, model_count: int):
        """Show connection success message."""
        self.status_var.set("Connection successful!")
        messagebox.showinfo(
            "Connection Successful", 
            f"Successfully connected to inference server!\n\n"
            f"Available models: {model_count}"
        )
    
    def _show_connection_error(self, error: str):
        """Show connection error message."""
        self.status_var.set("Connection failed.")
        messagebox.showerror("Connection Failed", f"Failed to connect to inference server:\n\n{error}")
    
    def _open_config_file(self):
        """Open the config.toml file in default text editor."""
        if sys.platform == "win32":
            os.startfile(str(CONFIG_PATH))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(CONFIG_PATH)])
        else:
            subprocess.run(["xdg-open", str(CONFIG_PATH)])
    
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


class NewScriptDialog(tk.Toplevel):
    """Dialog for creating a new script."""
    
    def __init__(self, master, project_name: str):
        super().__init__(master)
        self.title("New Script")
        self.geometry("450x320")
        self.resizable(False, False)
        self.transient(master)
        
        self.project_name = project_name
        self.result = None
        
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Script name
        ttk.Label(main_frame, text="Script Name:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.name_var, width=40).grid(row=0, column=1, pady=5, padx=(10, 0))
        
        # Script type
        ttk.Label(main_frame, text="Script Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.type_var = tk.StringVar(value="scraper")
        type_combo = ttk.Combobox(
            main_frame,
            textvariable=self.type_var,
            values=["scraper", "data-manipulation", "ai-script", "migration"],
            state="readonly",
            width=37
        )
        type_combo.grid(row=1, column=1, pady=5, padx=(10, 0))
        
        # File name
        ttk.Label(main_frame, text="File Name:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.filename_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.filename_var, width=40).grid(row=2, column=1, pady=5, padx=(10, 0))
        ttk.Label(main_frame, text="(e.g., my_scraper.py)", font=("Segoe UI", 8, "italic")).grid(
            row=3, column=1, sticky=tk.W, padx=(10, 0)
        )
        
        # Description
        ttk.Label(main_frame, text="Description:").grid(row=4, column=0, sticky=tk.NW, pady=5)
        self.desc_text = tk.Text(main_frame, height=4, width=38)
        self.desc_text.grid(row=4, column=1, pady=5, padx=(10, 0))
        
        # Template checkbox
        self.use_template_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            main_frame, 
            text="Create with template",
            variable=self.use_template_var
        ).grid(row=5, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=(15, 0))
        
        ttk.Button(btn_frame, text="Create", command=self._on_create).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
        self.grab_set()
    
    def _on_create(self):
        name = self.name_var.get().strip()
        filename = self.filename_var.get().strip()
        script_type = self.type_var.get()
        description = self.desc_text.get("1.0", tk.END).strip()
        
        if not name:
            messagebox.showwarning("Missing Name", "Please enter a script name.")
            return
        
        if not filename:
            # Auto-generate filename from name
            filename = name.lower().replace(" ", "_").replace("-", "_") + ".py"
        
        if not filename.endswith(".py") and script_type != "migration":
            filename += ".py"
        elif script_type == "migration" and not filename.endswith(".sql"):
            filename += ".sql"
        
        try:
            db = get_database()
            scripts_dir = db.get_scripts_directory(self.project_name)
            
            if not scripts_dir:
                messagebox.showerror("Error", f"Could not get scripts directory for project '{self.project_name}'.\n\n"
                                              "Please ensure the project exists and try again.")
                return
            
            # Determine subdirectory based on type
            type_dir = {
                "scraper": "scrapers",
                "data-manipulation": "data-manipulation",
                "ai-script": "ai-scripts",
                "migration": "migrations"
            }.get(script_type, "scrapers")
            
            file_path = Path(scripts_dir) / type_dir / filename
            relative_path = f"{type_dir}/{filename}"
            
            # Create file with template
            if self.use_template_var.get():
                template = self._get_template(script_type, name)
            else:
                template = ""
            
            file_path.write_text(template, encoding="utf-8")
            
            # Register in database
            script_id = db.register_script(
                self.project_name,
                name,
                script_type,
                relative_path,
                description
            )
            
            self.result = {"id": script_id, "name": name, "path": str(file_path)}
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create script: {e}")
    
    def _get_template(self, script_type: str, name: str) -> str:
        """Get a template for the script type."""
        if script_type == "scraper":
            return f'''#!/usr/bin/env python3
"""
Scraper: {name}
Description: [Add description here]
"""

import requests
from bs4 import BeautifulSoup


def scrape(url: str) -> dict:
    """
    Scrape data from the given URL.
    
    Args:
        url: The URL to scrape
        
    Returns:
        Dictionary with scraped data
    """
    response = requests.get(url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # TODO: Implement scraping logic
    data = {{
        "url": url,
        "title": soup.title.string if soup.title else "",
        "content": ""
    }}
    
    return data


if __name__ == "__main__":
    # Test the scraper
    result = scrape("https://example.com")
    print(result)
'''
        
        elif script_type == "data-manipulation":
            return f'''#!/usr/bin/env python3
"""
Data Manipulation: {name}
Description: [Add description here]
"""

import json
import sys
sys.path.insert(0, ".")

from src.database import get_database


def main():
    """Main entry point for data manipulation."""
    db = get_database()
    
    # TODO: Get your project name
    project_name = "your-project"
    
    # TODO: Implement data manipulation logic
    # Example: Get QA pairs
    # qa_pairs = db.get_qa_pairs(project_name)
    
    print("Data manipulation complete.")


if __name__ == "__main__":
    main()
'''
        
        elif script_type == "ai-script":
            return f'''#!/usr/bin/env python3
"""
AI Script: {name}
Description: [Add description here]
"""

import os
import sys
sys.path.insert(0, ".")

from openai import OpenAI


def main():
    """Main entry point for AI script."""
    # Initialize client (uses OPENAI_API_KEY or OPENROUTER_API_KEY)
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # TODO: Implement AI logic
    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {{"role": "system", "content": "You are a helpful assistant."}},
            {{"role": "user", "content": "Hello!"}}
        ]
    )
    
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
'''
        
        elif script_type == "migration":
            return f'''-- Migration: {name}
-- Description: [Add description here]
-- Version: 001

-- Add your SQL migration statements here

-- Example: Add a new column
-- ALTER TABLE qa_text ADD COLUMN category TEXT;

-- Example: Create a new table
-- CREATE TABLE IF NOT EXISTS custom_data (
--     id INTEGER PRIMARY KEY,
--     name TEXT NOT NULL,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );
'''
        
        return f"# {name}\n# Script type: {script_type}\n"


class ScriptExecutionDialog(tk.Toplevel):
    """Dialog for running a script and viewing output."""
    
    def __init__(self, master, project_name: str, script_id: int, script_path: Path):
        super().__init__(master)
        self.title("Script Execution")
        self.geometry("700x500")
        self.transient(master)
        
        self.project_name = project_name
        self.script_id = script_id
        self.script_path = script_path
        self.process = None
        self.execution_id = None
        
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Info bar
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text=f"Script: {script_path.name}", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(info_frame, textvariable=self.status_var).pack(side=tk.RIGHT)
        
        # Output area
        output_label = ttk.Label(main_frame, text="Output:")
        output_label.pack(anchor=tk.W)
        
        output_container = ttk.Frame(main_frame)
        output_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.output_text = tk.Text(
            output_container,
            wrap=tk.WORD,
            font=("Consolas", 9),
            state=tk.DISABLED,
            bg="#1e1e1e",
            fg="#d4d4d4"
        )
        output_scroll = ttk.Scrollbar(output_container, orient=tk.VERTICAL, command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scroll.set)
        
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure tags for colored output
        self.output_text.tag_configure("stdout", foreground="#d4d4d4")
        self.output_text.tag_configure("stderr", foreground="#f14c4c")
        self.output_text.tag_configure("info", foreground="#3794ff")
        
        # Button bar
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.run_btn = ttk.Button(btn_frame, text="▶ Run", command=self._run_script)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_btn = ttk.Button(btn_frame, text="⬛ Stop", command=self._stop_script, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(btn_frame, text="Clear", command=self._clear_output).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT)
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.grab_set()
    
    def _append_output(self, text: str, tag: str = "stdout"):
        """Append text to the output area."""
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.insert(tk.END, text, tag)
        self.output_text.see(tk.END)
        self.output_text.configure(state=tk.DISABLED)
    
    def _clear_output(self):
        """Clear the output area."""
        self.output_text.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state=tk.DISABLED)
    
    def _run_script(self):
        """Run the script in a subprocess."""
        if self.process:
            return
        
        self._clear_output()
        self._append_output(f"Starting: {self.script_path}\n", "info")
        self._append_output("-" * 50 + "\n", "info")
        
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Running...")
        
        # Log execution start
        try:
            db = get_database()
            self.execution_id = db.log_script_execution(
                self.project_name,
                self.script_id,
                "running"
            )
        except Exception:
            pass
        
        # Update state file to indicate script is running
        try:
            set_script_running(self.script_id, True)
        except Exception:
            pass
        
        # Determine how to run the script
        if self.script_path.suffix == ".py":
            cmd = [sys.executable, str(self.script_path)]
        elif self.script_path.suffix == ".sql":
            # For SQL files, we'd need to handle differently
            self._append_output("SQL migration execution not yet implemented.\n", "stderr")
            self._append_output("Please use the API or run manually.\n", "info")
            self._execution_finished(1, "", "SQL execution not implemented")
            return
        else:
            cmd = [str(self.script_path)]
        
        # Start the process
        def run_process():
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=Path(__file__).parent,
                    bufsize=1
                )
                
                stdout_data = []
                stderr_data = []
                
                # Read stdout and stderr
                import selectors
                sel = selectors.DefaultSelector()
                sel.register(self.process.stdout, selectors.EVENT_READ)
                sel.register(self.process.stderr, selectors.EVENT_READ)
                
                while self.process.poll() is None:
                    for key, _ in sel.select(timeout=0.1):
                        line = key.fileobj.readline()
                        if line:
                            if key.fileobj == self.process.stdout:
                                stdout_data.append(line)
                                self.after(0, lambda l=line: self._append_output(l, "stdout"))
                            else:
                                stderr_data.append(line)
                                self.after(0, lambda l=line: self._append_output(l, "stderr"))
                
                # Read remaining output
                remaining_stdout, remaining_stderr = self.process.communicate()
                if remaining_stdout:
                    stdout_data.append(remaining_stdout)
                    self.after(0, lambda: self._append_output(remaining_stdout, "stdout"))
                if remaining_stderr:
                    stderr_data.append(remaining_stderr)
                    self.after(0, lambda: self._append_output(remaining_stderr, "stderr"))
                
                exit_code = self.process.returncode
                self.after(0, lambda: self._execution_finished(
                    exit_code, 
                    "".join(stdout_data), 
                    "".join(stderr_data)
                ))
                
            except Exception as e:
                self.after(0, lambda: self._append_output(f"\nError: {e}\n", "stderr"))
                self.after(0, lambda: self._execution_finished(1, "", str(e)))
        
        threading.Thread(target=run_process, daemon=True).start()
    
    def _execution_finished(self, exit_code: int, stdout: str, stderr: str):
        """Handle script execution completion."""
        self.process = None
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        
        self._append_output("-" * 50 + "\n", "info")
        
        if exit_code == 0:
            self.status_var.set("Completed successfully")
            self._append_output("✓ Script completed successfully\n", "info")
            status = "success"
        else:
            self.status_var.set(f"Failed (exit code {exit_code})")
            self._append_output(f"✗ Script failed with exit code {exit_code}\n", "stderr")
            status = "failed"
        
        # Update state file to indicate script finished
        try:
            set_script_running(None, False)
        except Exception:
            pass
        
        # Update execution log
        if self.execution_id:
            try:
                db = get_database()
                db.update_script_execution(
                    self.project_name,
                    self.execution_id,
                    status,
                    exit_code,
                    stdout[-10000:] if len(stdout) > 10000 else stdout,  # Limit size
                    stderr[-10000:] if len(stderr) > 10000 else stderr
                )
            except Exception:
                pass
    
    def _stop_script(self):
        """Stop the running script."""
        if self.process:
            self.process.terminate()
            self._append_output("\n⚠ Script terminated by user\n", "stderr")
            self.status_var.set("Terminated")
            
            # Update state file
            try:
                set_script_running(None, False)
            except Exception:
                pass
            
            if self.execution_id:
                try:
                    db = get_database()
                    db.update_script_execution(
                        self.project_name,
                        self.execution_id,
                        "cancelled",
                        -1,
                        "",
                        "Terminated by user"
                    )
                except Exception:
                    pass
    
    def _on_close(self):
        """Handle window close."""
        if self.process:
            if messagebox.askyesno("Script Running", "A script is still running. Stop it?"):
                self._stop_script()
            else:
                return
        self.destroy()


if __name__ == "__main__":
    app = ProjectManagerApp()
    app.mainloop()
