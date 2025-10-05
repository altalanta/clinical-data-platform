"""Config CLI commands"""
import typer
from rich.console import Console

console = Console()
config_app = typer.Typer(name="config", help="🔧 Configuration management")

@config_app.command("show")
def show_config():
    console.print("🔧 Current configuration:")

def main():
    config_app()

if __name__ == "__main__":
    main()