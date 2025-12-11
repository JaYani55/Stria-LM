import toml
from pathlib import Path

# --- Path Configuration ---
CONFIG_PATH = Path(__file__).parent.parent / "config.toml"
PROJECTS_DIR = Path("projects")
PROJECTS_DIR.mkdir(exist_ok=True)

# --- Load Configuration ---
def load_config():
    """Loads the configuration from config.toml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    return toml.load(CONFIG_PATH)

config = load_config()

# --- Database Configuration ---
database_config = config.get("database", {})
DATABASE_URL = database_config.get("url", "postgresql+asyncpg://postgres:password@localhost/strialm")

# --- Embedding Configuration ---
embedding_config = config.get("embedding", {})
EMBEDDING_MODELS = embedding_config.get("models", {})
DEFAULT_EMBEDDING_MODEL = embedding_config.get("default_model", "local_default")

# --- Inference Configuration ---
inference_config = config.get("inference", {})
DEFAULT_HOST = inference_config.get("default_host", "127.0.0.1")
DEFAULT_PORT = inference_config.get("default_port", 8008)
GGUF_EXTENSIONS = inference_config.get("gguf_extensions", ["*.gguf"])
INFERENCE_MODEL_NAME = inference_config.get("inference_model_name", "default-model")
INFERENCE_MODEL_PATH = inference_config.get("inference_model_path", "")

# --- Function to Save Configuration ---
def save_config(data):
    """Saves the provided dictionary to config.toml."""
    with open(CONFIG_PATH, "w") as f:
        toml.dump(data, f)

# --- Experimental Features ---
ENABLE_WEIGHTS = config.get("experimental", {}).get("enable_weights", False)
