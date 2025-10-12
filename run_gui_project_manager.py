#!/usr/bin/env python3
"""Simple Tkinter GUI for browsing Stria-LM project databases."""
from __future__ import annotations

import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Sequence
import threading

from src.config import PROJECTS_DIR
from src.database import get_db_connection, get_db_path, re_embed_prompts as db_re_embed_prompts

ROW_LIMIT = 200


def discover_projects(projects_dir: Path) -> List[str]:
    """Return the list of project names that have an associated SQLite database."""
    project_names: List[str] = []
    for entry in projects_dir.iterdir():
        if not entry.is_dir():
            continue
        db_path = get_db_path(entry.name, projects_dir)
        if db_path.exists():
            project_names.append(entry.name)
    return sorted(project_names)


class ProjectSelectionDialog(tk.Toplevel):
    """Modal dialog that lets the user pick a project to open."""

    def __init__(self, master: tk.Misc, project_names: Sequence[str]):
        super().__init__(master)
        self.title("Select Stria-LM Project")
        self.resizable(False, False)
        self.transient(master)

        self.selected_project: Optional[str] = None

        ttk.Label(self, text="Choose a project to open:").pack(padx=12, pady=(12, 4), anchor=tk.W)

        self.projects_var = tk.StringVar(value=project_names)
        self.listbox = tk.Listbox(self, listvariable=self.projects_var, height=min(12, len(project_names)))
        self.listbox.pack(padx=12, pady=4, fill=tk.BOTH, expand=True)
        if project_names:
            self.listbox.selection_set(0)

        button_frame = ttk.Frame(self)
        button_frame.pack(padx=12, pady=(8, 12), fill=tk.X)

        ttk.Button(button_frame, text="Open", command=self._on_open).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(button_frame, text="Cancel", command=self._on_cancel).pack(side=tk.RIGHT)

        self.bind("<Return>", lambda *_: self._on_open())
        self.bind("<Escape>", lambda *_: self._on_cancel())

        self.grab_set()
        self.wait_window(self)

    def _on_open(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please pick a project to open.")
            return
        index = selection[0]
        self.selected_project = self.listbox.get(index)
        self.destroy()

    def _on_cancel(self):
        self.selected_project = None
        self.destroy()


class ProjectManagerGUI(tk.Tk):
    """Main GUI application for browsing project databases."""

    def __init__(self):
        super().__init__()
        self.title("Stria-LM Project Browser")
        self.geometry("1000x600")

        self.conn: Optional[sqlite3.Connection] = None
        self.current_project: Optional[str] = None
        self.table_names: List[str] = []
        self.toolbar_visible = tk.BooleanVar(value=True)

        self._build_scaffold()
        self._show_loading_overlay()

        self.after(200, self._initialize_app)

    def _build_scaffold(self):
        """Builds the static UI structure without loading any data."""
        self._build_menu()

        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        self.header = ttk.Frame(root)
        self.header.pack(fill=tk.X)
        ttk.Label(self.header, text="Current project:").pack(side=tk.LEFT)
        self.project_label_var = tk.StringVar(value="None")
        ttk.Label(self.header, textvariable=self.project_label_var, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(4, 0))

        self.toolbar = ttk.Frame(root, padding=(0, 8, 0, 0))
        ttk.Button(self.toolbar, text="Open Project", command=self._select_and_open_project).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(self.toolbar, text="Re-embed Prompts", command=self._show_re_embed_dialog).pack(side=tk.LEFT)
        self.toolbar.pack(fill=tk.X, after=self.header)

        self.body = ttk.Frame(root)
        self.body.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        # Data-independent parts of the layout
        list_frame = ttk.Frame(self.body)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(list_frame, text="Tables").pack(anchor=tk.W)
        self.tables_listbox = tk.Listbox(list_frame, height=20)
        self.tables_listbox.pack(fill=tk.Y, expand=True)
        self.tables_listbox.bind("<<ListboxSelect>>", lambda *_: self._load_selected_table())

        tree_frame = ttk.Frame(self.body)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.tree = ttk.Treeview(tree_frame, columns=(), show="headings")
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = ttk.Scrollbar(self.body, orient=tk.HORIZONTAL, command=self.tree.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.tree.bind("<Double-1>", self._on_double_click)

        self.status_var = tk.StringVar(value="Initializing...")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(12, 0))

    def _show_loading_overlay(self):
        """Displays a loading indicator over the main body."""
        self.loading_frame = ttk.Frame(self.body, style="TFrame")
        self.loading_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        ttk.Label(self.loading_frame, text="Loading projects...").pack(pady=5)
        ttk.Progressbar(self.loading_frame, mode='indeterminate').pack(pady=5)
        self.loading_frame.lift()

    def _hide_loading_overlay(self):
        """Hides the loading indicator."""
        if hasattr(self, "loading_frame"):
            self.loading_frame.destroy()

    def _initialize_app(self):
        """Finalizes initialization and opens the project selection dialog."""
        self._hide_loading_overlay()
        self._initial_project_selection()

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Project...", command=self._select_and_open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        db_menu = tk.Menu(menubar, tearoff=False)
        db_menu.add_command(label="Re-embed Prompts...", command=self._show_re_embed_dialog)
        menubar.add_cascade(label="Database", menu=db_menu)

        view_menu = tk.Menu(menubar, tearoff=False)
        view_menu.add_checkbutton(label="Show Toolbar", variable=self.toolbar_visible, command=self._toggle_toolbar)
        menubar.add_cascade(label="View", menu=view_menu)

        self.config(menu=menubar)

    def _build_layout(self):
        # This method is now effectively replaced by _build_scaffold
        # and the data-loading methods. It can be removed or left empty.
        pass

    def _toggle_toolbar(self):
        if self.toolbar_visible.get():
            self.toolbar.pack(fill=tk.X, after=self.header)
        else:
            self.toolbar.pack_forget()

    def _show_re_embed_dialog(self):
        if not self.current_project:
            messagebox.showwarning("No Project", "Please open a project first.")
            return

        dialog = ReEmbedDialog(self)
        if dialog.result:
            ids_to_re_embed = dialog.result
            self._trigger_re_embedding(ids_to_re_embed)

    def _trigger_re_embedding(self, ids_to_re_embed: str | List[int]):
        if not self.current_project:
            return

        self.status_var.set("Starting re-embedding process...")
        progress_dialog = ProgressDialog(self, "Re-embedding Prompts")

        def re_embed_task():
            try:
                total_prompts = db_re_embed_prompts(
                    self.current_project,
                    PROJECTS_DIR,
                    ids_to_re_embed,
                    progress_callback=progress_dialog.update_progress
                )
                progress_dialog.complete(f"Successfully re-embedded {total_prompts} prompt(s).")
                self._load_selected_table()
            except Exception as e:
                progress_dialog.complete(f"Error during re-embedding: {e}", is_error=True)

        threading.Thread(target=re_embed_task, daemon=True).start()

    def _on_double_click(self, event):
        """Handle double-click on a treeview cell to edit its value."""
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        column_id = self.tree.identify_column(event.x)
        column_index = int(column_id.replace("#", "")) - 1
        selected_iid = self.tree.focus()

        if not selected_iid:
            return

        item_values = self.tree.item(selected_iid)["values"]
        column_name = self.tree.column(column_id, "id")

        selection = self.tables_listbox.curselection()
        if not selection:
            return
        table_name = self.tables_listbox.get(selection[0])

        if not self.conn:
            return

        cursor = self.conn.cursor()
        try:
            cursor.execute(f"PRAGMA table_info({table_name})")
            table_info = cursor.fetchall()
        except sqlite3.Error as e:
            messagebox.showerror("Error", f"Could not get table info: {e}")
            return

        pk_column_info = next((col for col in table_info if col["pk"]), None)

        if not pk_column_info:
            messagebox.showwarning("Cannot Edit", f"Table '{table_name}' has no primary key, so it cannot be edited.")
            return

        pk_name = pk_column_info["name"]
        pk_column_index = -1
        columns = [self.tree.column(c, "id") for c in self.tree["columns"]]
        try:
            pk_column_index = columns.index(pk_name)
        except ValueError:
            messagebox.showerror("Error", f"Primary key column '{pk_name}' not found in the displayed columns.")
            return

        pk_value = item_values[pk_column_index]

        # Fetch the full, untruncated value from the database
        try:
            query = f'SELECT "{column_name}" FROM "{table_name}" WHERE "{pk_name}" = ?'
            cursor.execute(query, (pk_value,))
            result = cursor.fetchone()
            if result is None:
                messagebox.showerror("Error", "Could not fetch the original value to edit.")
                return
            old_value = result[0]
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to fetch original value: {e}")
            return

        self._show_edit_window(table_name, column_name, pk_name, pk_value, old_value)

    def _show_edit_window(self, table_name, column_name, pk_name, pk_value, old_value):
        """Show a window to edit a cell value."""
        edit_window = tk.Toplevel(self)
        edit_window.title(f"Edit {column_name}")
        edit_window.geometry("600x400")
        edit_window.transient(self)

        ttk.Label(edit_window, text=f"Editing '{column_name}' for row where {pk_name} = {pk_value}").pack(padx=12, pady=(12, 4))

        text_widget = tk.Text(edit_window, wrap=tk.WORD, height=15, width=70)
        text_widget.pack(padx=12, pady=4, fill=tk.BOTH, expand=True)
        text_widget.insert("1.0", old_value)

        button_frame = ttk.Frame(edit_window)
        button_frame.pack(padx=12, pady=(8, 12), fill=tk.X)

        def on_save():
            new_value = text_widget.get("1.0", tk.END).strip()
            self._update_value_in_db(table_name, column_name, new_value, pk_name, pk_value)
            edit_window.destroy()
            self._load_selected_table()

        def on_cancel():
            edit_window.destroy()

        ttk.Button(button_frame, text="Save", command=on_save).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)

        edit_window.grab_set()
        self.wait_window(edit_window)

    def _update_value_in_db(self, table_name, column_name, new_value, pk_name, pk_value):
        if not self.conn:
            return
        try:
            cursor = self.conn.cursor()
            query = f'UPDATE "{table_name}" SET "{column_name}" = ? WHERE "{pk_name}" = ?'
            cursor.execute(query, (new_value, pk_value))
            self.conn.commit()
            self.status_var.set(f"Successfully updated {column_name} for {pk_name} {pk_value}.")
        except sqlite3.Error as e:
            messagebox.showerror("Update Failed", f"Failed to update the database: {e}")
            self.conn.rollback()

    def _initial_project_selection(self):
        if self.current_project is None:
            self._select_and_open_project(initial=True)

    def _select_and_open_project(self, initial: bool = False):
        project_names = discover_projects(PROJECTS_DIR)
        if not project_names:
            messagebox.showerror("No projects found", f"No project databases were found in {PROJECTS_DIR.resolve()}.")
            if initial:
                self.destroy()
            return

        dialog = ProjectSelectionDialog(self, project_names)
        project_to_open = dialog.selected_project
        if project_to_open:
            self._open_project(project_to_open)
        elif initial:
            self.destroy()

    def _open_project(self, project_name: str):
        try:
            db_path = get_db_path(project_name, PROJECTS_DIR)
            if not db_path.exists():
                messagebox.showerror("Missing database", f"Database file not found: {db_path}")
                return

            self._close_connection()

            self.conn = get_db_connection(db_path)
            self.conn.row_factory = sqlite3.Row
            self.current_project = project_name
            self.project_label_var.set(project_name)
            self.status_var.set(f"Connected to {db_path}")
            self._populate_table_list()
        except Exception as exc:
            messagebox.showerror("Connection error", f"Failed to open project '{project_name}': {exc}")
            self._close_connection()

    def _populate_table_list(self):
        if not self.conn:
            return
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
        )
        self.table_names = [row[0] for row in cursor.fetchall()]

        self.tables_listbox.delete(0, tk.END)
        for name in self.table_names:
            self.tables_listbox.insert(tk.END, name)

        if self.table_names:
            self.tables_listbox.selection_clear(0, tk.END)
            self.tables_listbox.selection_set(0)
            self._load_selected_table()
        else:
            self._clear_tree()
            self.status_var.set("No tables found in this database.")

    def _load_selected_table(self):
        if not self.conn:
            return
        selection = self.tables_listbox.curselection()
        if not selection:
            return
        table_name = self.tables_listbox.get(selection[0])
        self._display_table_data(table_name)

    def _display_table_data(self, table_name: str):
        assert self.conn is not None
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {ROW_LIMIT}")
        except sqlite3.Error as exc:
            messagebox.showerror("Query error", f"Failed to read table '{table_name}': {exc}")
            return

        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description] if cursor.description else []

        self._refresh_tree(columns, rows)

        row_count = len(rows)
        suffix = " (limited to first {ROW_LIMIT} rows)" if row_count == ROW_LIMIT else ""
        self.status_var.set(f"Table '{table_name}' – showing {row_count} row(s){suffix}.")

    def _refresh_tree(self, columns: Sequence[str], rows: Sequence[sqlite3.Row]):
        self._clear_tree()
        if not columns:
            return

        self.tree.configure(columns=columns)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor=tk.W)

        for row in rows:
            formatted = [self._format_cell_value(row[col]) for col in columns]
            self.tree.insert("", tk.END, values=formatted)

    def _format_cell_value(self, value):
        if isinstance(value, bytes):
            return f"<BLOB len={len(value)}>"
        if value is None:
            return "NULL"
        text = str(value)
        if len(text) > 200:
            return text[:197] + "..."
        return text

    def _clear_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree.configure(columns=())

    def _close_connection(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.current_project = None

    def destroy(self):  # type: ignore[override]
        self._close_connection()
        super().destroy()


class ReEmbedDialog(tk.Toplevel):
    """Dialog to get options for re-embedding."""
    def __init__(self, master):
        super().__init__(master)
        self.title("Re-embed Prompts")
        self.transient(master)
        self.result: Optional[str | List[int]] = None

        self.mode = tk.StringVar(value="all")

        ttk.Label(self, text="Choose which prompts to re-embed:").pack(padx=12, pady=(12, 4), anchor=tk.W)

        all_frame = ttk.Frame(self)
        ttk.Radiobutton(all_frame, text="Re-embed ALL prompts", variable=self.mode, value="all", command=self._on_mode_change).pack(side=tk.LEFT)
        all_frame.pack(padx=12, pady=2, anchor=tk.W)

        specific_frame = ttk.Frame(self)
        ttk.Radiobutton(specific_frame, text="Re-embed specific prompt IDs (comma-separated):", variable=self.mode, value="specific", command=self._on_mode_change).pack(side=tk.LEFT)
        specific_frame.pack(padx=12, pady=2, anchor=tk.W)

        self.ids_entry = ttk.Entry(self, width=40)
        self.ids_entry.pack(padx=12, pady=2, anchor=tk.W)

        button_frame = ttk.Frame(self)
        button_frame.pack(padx=12, pady=(8, 12), fill=tk.X)
        ttk.Button(button_frame, text="Start", command=self._on_start).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=(0, 4))

        self._on_mode_change()
        self.grab_set()
        self.wait_window(self)

    def _on_mode_change(self):
        if self.mode.get() == "all":
            self.ids_entry.config(state=tk.DISABLED)
        else:
            self.ids_entry.config(state=tk.NORMAL)

    def _on_start(self):
        if self.mode.get() == "all":
            self.result = "all"
        else:
            try:
                ids_str = self.ids_entry.get()
                if not ids_str:
                    messagebox.showwarning("Input Required", "Please enter at least one prompt ID.", parent=self)
                    return
                self.result = [int(i.strip()) for i in ids_str.split(',') if i.strip()]
                if not self.result:
                    messagebox.showwarning("Input Required", "Please enter valid, comma-separated prompt IDs.", parent=self)
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid list of comma-separated integer IDs.", parent=self)
                return
        self.destroy()


class ProgressDialog(tk.Toplevel):
    """A simple dialog to show the progress of a long-running task."""
    def __init__(self, master, title):
        super().__init__(master)
        self.title(title)
        self.geometry("400x120")
        self.transient(master)
        self.resizable(False, False)

        self.progress_var = tk.StringVar(value="Initializing...")
        ttk.Label(self, textvariable=self.progress_var).pack(padx=20, pady=20)

        self.progressbar = ttk.Progressbar(self, orient="horizontal", length=360, mode="determinate")
        self.progressbar.pack(padx=20, pady=5)

        self.grab_set()

    def update_progress(self, current: int, total: int):
        if not self.winfo_exists():
            return
        percentage = (current / total) * 100
        self.progressbar["value"] = percentage
        self.progress_var.set(f"Processing prompt {current} of {total}...")
        self.update_idletasks()

    def complete(self, message: str, is_error: bool = False):
        if not self.winfo_exists():
            return
        self.progress_var.set(message)
        self.progressbar.pack_forget()
        if is_error:
            messagebox.showerror("Error", message, parent=self)
        else:
            messagebox.showinfo("Complete", message, parent=self)
        self.destroy()


if __name__ == "__main__":
    app = ProjectManagerGUI()
    app.mainloop()
