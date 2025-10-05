"""
Main CLI application for Clinical Data Platform

Provides a comprehensive command-line interface for all platform operations
including data management, API serving, validation, and administration.
"""

import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from clinical_platform.cli import __version__
from clinical_platform.cli.api import api_app
from clinical_platform.cli.ingest import ingest_app
from clinical_platform.cli.validate import validate_app
from clinical_platform.cli.transform import transform_app
from clinical_platform.cli.serve import serve_app
from clinical_platform.cli.demo import demo_app
from clinical_platform.cli.admin import admin_app
from clinical_platform.cli.config import config_app
from clinical_platform.cli.health import health_app

console = Console()

# Main Typer application
app = typer.Typer(
    name="clinical-data-platform",
    help="🏥 Clinical Data Platform - End-to-end clinical data operations",
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    add_completion=True,
)

# Add subcommands
app.add_typer(api_app, name="api", help="🔌 API server operations")
app.add_typer(ingest_app, name="ingest", help="📥 Data ingestion operations")
app.add_typer(validate_app, name="validate", help="✅ Data validation operations")
app.add_typer(transform_app, name="transform", help="🔄 Data transformation with dbt")
app.add_typer(serve_app, name="serve", help="🚀 Serve applications (API, UI, docs)")
app.add_typer(demo_app, name="demo", help="🎲 Demo operations and examples")
app.add_typer(admin_app, name="admin", help="⚙️ Administrative operations")
app.add_typer(config_app, name="config", help="🔧 Configuration management")
app.add_typer(health_app, name="health", help="🩺 Health checks and diagnostics")


def version_callback(value: bool):
    """Print version information."""
    if value:
        console.print(f"[bold green]Clinical Data Platform[/bold green] v{__version__}")
        console.print("🏥 [italic]End-to-end clinical data operations platform[/italic]")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True,
        help="Show version information"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output"),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
):
    """
    🏥 Clinical Data Platform CLI
    
    A comprehensive command-line interface for clinical data operations including
    ingestion, transformation, validation, ML, and API serving.
    
    Examples:
        clinical-data-platform demo public          # Run public demo
        clinical-data-platform api start           # Start API server
        clinical-data-platform validate data/      # Validate data quality
        clinical-data-platform transform run       # Run dbt transformations
    """
    if verbose:
        console.print("[dim]Verbose mode enabled[/dim]")
    
    if config_file:
        console.print(f"[dim]Using config file: {config_file}[/dim]")


@app.command()
def info():
    """📋 Show platform information and status"""
    console.print(Panel.fit("🏥 [bold]Clinical Data Platform[/bold]", style="blue"))
    
    # Create info table
    info_table = Table(title="Platform Information", show_header=True)
    info_table.add_column("Component", style="cyan")
    info_table.add_column("Status", style="green")
    info_table.add_column("Description", style="white")
    
    components = [
        ("API Server", "Available", "FastAPI REST API with OpenAPI docs"),
        ("Data Validation", "Available", "Great Expectations + Pandera validation"),
        ("Transformations", "Available", "dbt models and analytics"),
        ("ML Pipeline", "Available", "Scikit-learn models with MLflow"),
        ("Documentation", "Available", "MkDocs with Material theme"),
        ("Quality Gates", "Available", "Automated testing and validation"),
        ("Demo Mode", "Available", "Synthetic data for testing"),
        ("HIPAA Compliance", "Available", "PHI scrubbing and audit trails")
    ]
    
    for component, status, description in components:
        info_table.add_row(component, status, description)
    
    console.print(info_table)
    
    # Usage examples
    examples_text = Text()
    examples_text.append("🚀 Quick Start Examples:\n", style="bold yellow")
    examples_text.append("  clinical-data-platform demo public\n", style="green")
    examples_text.append("  clinical-data-platform api start --port 8000\n", style="green")
    examples_text.append("  clinical-data-platform health check\n", style="green")
    examples_text.append("  clinical-data-platform validate --data-dir ./data\n", style="green")
    
    console.print(Panel(examples_text, title="Examples", border_style="yellow"))


@app.command()
def quickstart():
    """🚀 Interactive quickstart guide"""
    console.print(Panel.fit("🚀 [bold]Clinical Data Platform Quickstart[/bold]", style="green"))
    
    steps = [
        "1️⃣ Generate synthetic demo data",
        "2️⃣ Run data validation checks", 
        "3️⃣ Execute dbt transformations",
        "4️⃣ Start API server",
        "5️⃣ View documentation"
    ]
    
    console.print("\n[bold yellow]Follow these steps to get started:[/bold yellow]")
    for step in steps:
        console.print(f"  {step}")
    
    console.print("\n[bold green]Run the complete demo:[/bold green]")
    console.print("  [cyan]clinical-data-platform demo public[/cyan]")
    
    if typer.confirm("\nWould you like to run the public demo now?"):
        console.print("🎲 Starting public demo...")
        # Import and run demo
        from clinical_platform.cli.demo import run_public_demo
        run_public_demo()
    else:
        console.print("ℹ️ Run [cyan]clinical-data-platform demo public[/cyan] when ready!")


@app.command()
def status():
    """📊 Show platform status and health"""
    console.print("🩺 [bold]Checking platform health...[/bold]")
    
    # Check components
    checks = [
        ("Python Environment", check_python),
        ("Required Dependencies", check_dependencies),
        ("Database Connection", check_database),
        ("Configuration", check_config),
        ("Data Directory", check_data_dir)
    ]
    
    status_table = Table(title="Health Checks", show_header=True)
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="white")
    status_table.add_column("Details", style="dim")
    
    all_healthy = True
    
    for name, check_func in checks:
        try:
            status, details = check_func()
            status_icon = "✅" if status else "❌"
            if not status:
                all_healthy = False
            status_table.add_row(name, status_icon, details)
        except Exception as e:
            status_table.add_row(name, "❌", f"Error: {str(e)}")
            all_healthy = False
    
    console.print(status_table)
    
    if all_healthy:
        console.print("\n🎉 [bold green]All systems operational![/bold green]")
    else:
        console.print("\n⚠️ [bold yellow]Some issues detected. Run with --verbose for details.[/bold yellow]")


def check_python() -> tuple[bool, str]:
    """Check Python version."""
    import sys
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (requires 3.11+)"


def check_dependencies() -> tuple[bool, str]:
    """Check required dependencies."""
    required = ["pandas", "duckdb", "fastapi", "typer", "rich"]
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, f"All {len(required)} dependencies available"


def check_database() -> tuple[bool, str]:
    """Check database connectivity."""
    try:
        import duckdb
        conn = duckdb.connect(":memory:")
        conn.execute("SELECT 1")
        return True, f"DuckDB {duckdb.__version__}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def check_config() -> tuple[bool, str]:
    """Check configuration."""
    try:
        from clinical_platform.config import get_config
        config = get_config()
        return True, "Configuration loaded successfully"
    except Exception as e:
        return False, f"Config error: {str(e)}"


def check_data_dir() -> tuple[bool, str]:
    """Check data directory."""
    data_dirs = ["./data", "./demo_data", "/tmp/clinical_data"]
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            return True, f"Found: {data_dir}"
    
    return False, "No data directory found"


if __name__ == "__main__":
    app()