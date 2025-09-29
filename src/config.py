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

# --- Embedding Configuration ---
embedding_config = config.get("embedding", {})
EMBEDDING_MODELS = embedding_config.get("models", {})
DEFAULT_EMBEDDING_MODEL = embedding_config.get("default_model", "local_default")

# --- Experimental Features ---
ENABLE_WEIGHTS = config.get("experimental", {}).get("enable_weights", False)
