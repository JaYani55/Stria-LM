import os
from pathlib import Path

PROJECTS_DIR = Path(os.getenv("PROJECTS_DIR", "projects"))
PROJECTS_DIR.mkdir(exist_ok=True)
