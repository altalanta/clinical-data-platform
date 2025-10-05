"""Serve CLI commands"""
import typer
from rich.console import Console

console = Console()
serve_app = typer.Typer(name="serve", help="🚀 Serve applications")

@serve_app.command("docs")
def serve_docs():
    console.print("📚 Starting documentation server...")

def main():
    serve_app()

if __name__ == "__main__":
    main()