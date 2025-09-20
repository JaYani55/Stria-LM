import os
from pathlib import Path

PROJECTS_DIR = Path(os.getenv("PROJECTS_DIR", "projects"))
PROJECTS_DIR.mkdir(exist_ok=True)

# Experimental Features
ENABLE_WEIGHTS = os.getenv("ENABLE_WEIGHTS", "false").lower() == "true"
