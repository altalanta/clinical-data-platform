"""Transform CLI commands"""
import typer
from rich.console import Console

console = Console()
transform_app = typer.Typer(name="transform", help="🔄 Data transformation with dbt")

@transform_app.command("run")
def run_transform():
    console.print("🔄 Running dbt transformations...")

def main():
    transform_app()

if __name__ == "__main__":
    main()