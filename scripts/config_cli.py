import typer
import toml
from pathlib import Path
import json

CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

app = typer.Typer()

def load_config():
    """Loads the configuration from config.toml."""
    if not CONFIG_PATH.exists():
        typer.echo(f"Error: Configuration file not found at {CONFIG_PATH}")
        raise typer.Exit(code=1)
    return toml.load(CONFIG_PATH)

def save_config(config):
    """Saves the configuration to config.toml."""
    with open(CONFIG_PATH, "w") as f:
        toml.dump(config, f)

@app.command()
def list_models():
    """
    List all available embedding models from the configuration.
    """
    config = load_config()
    models = config.get("embedding", {}).get("models", {})
    if not models:
        typer.echo("No embedding models found in configuration.")
        return
        
    typer.echo(json.dumps(models, indent=2))

@app.command()
def get_default():
    """
    Get the default embedding model.
    """
    config = load_config()
    default_model = config.get("embedding", {}).get("default_model")
    if not default_model:
        typer.echo("Default embedding model is not set in the configuration.")
        return
    
    typer.echo(f"Default embedding model: {default_model}")

if __name__ == "__main__":
    app()
