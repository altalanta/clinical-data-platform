"""Ingest CLI commands"""
import typer
from rich.console import Console

console = Console()
ingest_app = typer.Typer(name="ingest", help="📥 Data ingestion operations")

@ingest_app.command("run")
def run_ingest():
    console.print("📥 Running data ingestion...")

def main():
    ingest_app()

if __name__ == "__main__":
    main()