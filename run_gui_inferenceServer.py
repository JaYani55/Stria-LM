#!/usr/bin/env python3
"""
Main application for the Stria-LM GUI.
This application provides a graphical user interface for managing and running
AI models, including loading GGUF models and serving them via a FastAPI backend.
"""
import asyncio
import logging
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import uvicorn
from llama_cpp import Llama

# Adjust the import path to match the project structure
from src.inference_api import app_state, inference_app
from src.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    GGUF_EXTENSIONS,
    INFERENCE_MODEL_NAME,
    load_config,
    save_config
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Application Class ---

class AIInferenceGUI(tk.Tk):
    """
    A Tkinter-based GUI for loading AI models and managing the inference server.
    """
    def __init__(self):
        super().__init__()
        self.title("Stria-LM Inference Engine")
        self.geometry("600x450")

        self.model_path = tk.StringVar(value="Not loaded")
        self.server_status = tk.StringVar(value="Not running")
        self.host = tk.StringVar(value=DEFAULT_HOST)
        self.port = tk.StringVar(value=str(DEFAULT_PORT))

        self.server_thread: Optional[threading.Thread] = None
        self.server_instance: Optional[uvicorn.Server] = None

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load last used model path from config
        self.load_initial_config()

    def create_widgets(self):
        """Creates and arranges all the widgets in the main window."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Model Section ---
        model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="10")
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="Selected Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(model_frame, textvariable=self.model_path, relief="sunken", padding=2).grid(row=0, column=1, columnspan=2, sticky=tk.EW, pady=2)
        
        ttk.Button(model_frame, text="Browse...", command=self.browse_model).grid(row=1, column=1, sticky=tk.E, padx=5)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).grid(row=1, column=2, sticky=tk.W)

        # --- Server Section ---
        server_frame = ttk.LabelFrame(main_frame, text="Inference Server", padding="10")
        server_frame.pack(fill=tk.X, pady=5)

        ttk.Label(server_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(server_frame, textvariable=self.server_status, relief="sunken", padding=2).grid(row=0, column=1, sticky=tk.EW, pady=2)

        ttk.Label(server_frame, text="Host:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(server_frame, textvariable=self.host).grid(row=1, column=1, sticky=tk.EW)
        
        ttk.Label(server_frame, text="Port:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(server_frame, textvariable=self.port).grid(row=2, column=1, sticky=tk.EW)

        server_frame.columnconfigure(1, weight=1)

        ttk.Button(server_frame, text="Start Server", command=self.start_server).grid(row=3, column=0, pady=10)
        ttk.Button(server_frame, text="Stop Server", command=self.stop_server).grid(row=3, column=1, pady=10, sticky=tk.W)

        # --- Log Section ---
        log_frame = ttk.LabelFrame(main_frame, text="Logs", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, height=10, state='disabled', wrap='word')
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text['yscrollcommand'] = scrollbar.set
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Redirect stdout/stderr to the log widget
        self.redirect_logging()

    def redirect_logging(self):
        """Redirects logging output to the Tkinter text widget."""
        text_handler = TextHandler(self.log_text)
        logging.getLogger().addHandler(text_handler)

    def log(self, message: str):
        """Logs a message to the GUI and the console."""
        logger.info(message)

    def load_initial_config(self):
        """Loads the model path from the config file on startup."""
        config = load_config()
        model_path = config.get("inference_model_path")
        if model_path and os.path.exists(model_path):
            self.model_path.set(model_path)
            self.log(f"Loaded initial model path from config: {model_path}")

    def browse_model(self):
        """Opens a file dialog to select a GGUF model file."""
        filetypes = [("GGUF models", GGUF_EXTENSIONS), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(
            title="Select a GGUF Model",
            filetypes=filetypes,
            initialdir=os.path.expanduser("~")
        )
        if filepath:
            self.model_path.set(filepath)
            self.log(f"Selected model file: {filepath}")

    def load_model(self):
        """Loads the selected GGUF model into memory using llama-cpp-python."""
        path = self.model_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Invalid model path.")
            return

        self.log(f"Loading model: {path}...")
        try:
            # Unload previous model if it exists
            if app_state.model_instance:
                del app_state.model_instance
                app_state.model_instance = None
                self.log("Unloaded previous model.")

            # Load the new model
            app_state.model_instance = Llama(model_path=path, n_ctx=2048, n_gpu_layers=-1)
            
            # Update the global model name for the API
            global INFERENCE_MODEL_NAME
            INFERENCE_MODEL_NAME = os.path.basename(path)

            self.log(f"Successfully loaded model: {INFERENCE_MODEL_NAME}")
            
            # Save the path to config for next session
            config = load_config()
            config["inference_model_path"] = path
            save_config(config)

        except Exception as e:
            messagebox.showerror("Model Load Error", f"Failed to load model: {e}")
            self.log(f"Error loading model: {e}")
            app_state.model_instance = None

    def start_server(self):
        """Starts the Uvicorn server in a separate thread."""
        if self.server_thread and self.server_thread.is_alive():
            messagebox.showinfo("Info", "Server is already running.")
            return

        if not app_state.model_instance:
            messagebox.showerror("Error", "Cannot start server without a loaded model.")
            return

        host = self.host.get()
        port = int(self.port.get())

        self.log(f"Starting server at http://{host}:{port}")
        
        config = uvicorn.Config(inference_app, host=host, port=port, log_level="info")
        self.server_instance = uvicorn.Server(config)
        
        self.server_thread = threading.Thread(target=self.server_instance.run)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        self.server_status.set(f"Running at http://{host}:{port}")
        self.log("Server started successfully.")

    def stop_server(self):
        """Stops the running Uvicorn server."""
        if not self.server_thread or not self.server_thread.is_alive() or not self.server_instance:
            messagebox.showinfo("Info", "Server is not running.")
            return

        self.log("Stopping server...")
        self.server_instance.should_exit = True
        self.server_thread.join(timeout=5) # Wait for the thread to terminate
        
        if self.server_thread.is_alive():
            self.log("Server did not shut down gracefully. Force closing.")
        else:
            self.log("Server stopped.")
            
        self.server_status.set("Not running")
        self.server_thread = None
        self.server_instance = None

    def on_closing(self):
        """Handles the window closing event."""
        self.log("Application closing...")
        self.stop_server()
        self.destroy()

# --- Text Handler for Logging ---

class TextHandler(logging.Handler):
    """A logging handler that redirects logs to a Tkinter Text widget."""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.configure(state='disabled')
            self.text_widget.yview(tk.END)
        self.text_widget.after(0, append)

# --- Main Execution ---

if __name__ == "__main__":
    app = AIInferenceGUI()
    app.mainloop()
