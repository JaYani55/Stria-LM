import os
import toml
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal

# --- Path Configuration ---
CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
ENV_PATH = Path(__file__).parent.parent / ".env"
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# --- Load Environment Variables ---
# Load .env file if it exists (for secrets like DATABASE_URL, API keys)
load_dotenv(ENV_PATH)

# --- Load Configuration ---
def load_config():
    """Loads the configuration from config.toml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    return toml.load(CONFIG_PATH)

config = load_config()

# --- Database Configuration ---
database_config = config.get("database", {})

# Database type: "postgresql" or "sqlite"
# Priority: Environment variable > config.toml > default
DATABASE_TYPE: Literal["postgresql", "sqlite"] = os.getenv(
    "DATABASE_TYPE", 
    database_config.get("type", "sqlite")
)

# PostgreSQL connection URL (for remote database)
# Priority: Environment variable > config.toml > default
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    database_config.get("url", "postgresql+asyncpg://postgres:password@localhost/strialm")
)

# SQLite projects directory (for local databases)
SQLITE_PROJECTS_DIR = Path(database_config.get("sqlite_path", "projects"))
SQLITE_PROJECTS_DIR.mkdir(exist_ok=True)

# --- Embedding Configuration ---
embedding_config = config.get("embedding", {})
EMBEDDING_MODELS = embedding_config.get("models", {})
DEFAULT_EMBEDDING_MODEL = embedding_config.get("default_model", "local_default")

# Override API keys from environment if available
for model_id, model_config in EMBEDDING_MODELS.items():
    env_key = f"{model_id.upper()}_API_KEY"
    if os.getenv(env_key):
        model_config["api_key"] = os.getenv(env_key)
    # Also check generic OPENAI_API_KEY for OpenAI models
    if model_config.get("category") == "OpenAI API" and os.getenv("OPENAI_API_KEY"):
        model_config["api_key"] = os.getenv("OPENAI_API_KEY")

# --- Inference Configuration ---
inference_config = config.get("inference", {})
DEFAULT_HOST = inference_config.get("default_host", "127.0.0.1")
DEFAULT_PORT = inference_config.get("default_port", 8008)
GGUF_EXTENSIONS = inference_config.get("gguf_extensions", ["*.gguf"])
INFERENCE_MODEL_NAME = inference_config.get("inference_model_name", "default-model")
INFERENCE_MODEL_PATH = inference_config.get("inference_model_path", "")

# OpenRouter API key for LLM agents
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# --- Function to Save Configuration ---
def save_config(data):
    """Saves the provided dictionary to config.toml."""
    with open(CONFIG_PATH, "w") as f:
        toml.dump(data, f)

def reload_config():
    """Reload configuration from files."""
    global config, DATABASE_TYPE, DATABASE_URL, EMBEDDING_MODELS
    load_dotenv(ENV_PATH, override=True)
    config = load_config()

# --- Experimental Features ---
ENABLE_WEIGHTS = os.getenv("ENABLE_WEIGHTS", str(config.get("experimental", {}).get("enable_weights", False))).lower() in ("true", "1", "yes")

# --- Environment Variable Management ---
def get_env_variables() -> dict:
    """Get all environment variables from .env file."""
    env_vars = {}
    if ENV_PATH.exists():
        with open(ENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip().strip('"').strip("'")
    return env_vars

def set_env_variable(key: str, value: str):
    """Set an environment variable in the .env file."""
    env_vars = get_env_variables()
    env_vars[key] = value
    save_env_variables(env_vars)
    os.environ[key] = value

def save_env_variables(env_vars: dict):
    """Save environment variables to .env file."""
    with open(ENV_PATH, 'w') as f:
        f.write("# Stria-LM Environment Configuration\n")
        f.write("# This file contains sensitive credentials - do not commit to version control\n\n")
        for key, value in sorted(env_vars.items()):
            # Quote values that contain spaces or special characters
            if ' ' in value or '#' in value:
                value = f'"{value}"'
            f.write(f"{key}={value}\n")

def create_default_env():
    """Create a default .env file if it doesn't exist."""
    if not ENV_PATH.exists():
        default_vars = {
            "DATABASE_TYPE": "sqlite",
            "DATABASE_URL": "postgresql+asyncpg://postgres:password@localhost/strialm",
            "OPENROUTER_API_KEY": "",
            "OPENAI_API_KEY": "",
            "ENABLE_WEIGHTS": "true"
        }
        save_env_variables(default_vars)
        return True
    return False

