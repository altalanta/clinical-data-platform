"""Admin CLI commands"""
import typer
from rich.console import Console

console = Console()
admin_app = typer.Typer(name="admin", help="⚙️ Administrative operations")

@admin_app.command("status")
def admin_status():
    console.print("⚙️ Checking admin status...")

def main():
    admin_app()

if __name__ == "__main__":
    main()