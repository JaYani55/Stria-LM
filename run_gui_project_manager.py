#!/usr/bin/env python3
"""Simple Tkinter GUI for browsing Stria-LM project databases."""
from __future__ import annotations

import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Sequence

from src.config import PROJECTS_DIR
from src.database import get_db_connection, get_db_path

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

        self._build_menu()
        self._build_layout()

        self.after(100, self._initial_project_selection)

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open Project...", command=self._select_and_open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_layout(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Current project:").pack(side=tk.LEFT)
        self.project_label_var = tk.StringVar(value="None")
        ttk.Label(header, textvariable=self.project_label_var, font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(4, 0))

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        # Table list
        list_frame = ttk.Frame(body)
        list_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(list_frame, text="Tables").pack(anchor=tk.W)
        self.tables_listbox = tk.Listbox(list_frame, height=20)
        self.tables_listbox.pack(fill=tk.Y, expand=True)
        self.tables_listbox.bind("<<ListboxSelect>>", lambda *_: self._load_selected_table())

        # Treeview for data
        tree_frame = ttk.Frame(body)
        tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        self.tree = ttk.Treeview(tree_frame, columns=(), show="headings")
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar_y = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x = ttk.Scrollbar(body, orient=tk.HORIZONTAL, command=self.tree.xview)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Status bar
        self.status_var = tk.StringVar(value="Select a table to view data.")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(12, 0))

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


if __name__ == "__main__":
    app = ProjectManagerGUI()
    app.mainloop()
