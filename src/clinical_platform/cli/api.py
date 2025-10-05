"""
API CLI commands for Clinical Data Platform

Provides commands to start, stop, and manage the FastAPI server.
"""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional
import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel

console = Console()

api_app = typer.Typer(
    name="api",
    help="🔌 API server operations",
    rich_markup_mode="rich"
)


@api_app.command("start")
def start_api(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    log_level: str = typer.Option("info", "--log-level", help="Log level"),
    read_only: bool = typer.Option(False, "--read-only", help="Start in read-only mode"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for authentication"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Configuration file"),
):
    """
    🚀 Start the FastAPI server
    
    Examples:
        clinical-data-platform api start                    # Start on default port
        clinical-data-platform api start --port 8080       # Custom port
        clinical-data-platform api start --reload          # Development mode
        clinical-data-platform api start --read-only       # Read-only mode
    """
    
    console.print(Panel.fit("🔌 [bold]Starting Clinical Data Platform API[/bold]", style="blue"))
    
    # Set environment variables
    if read_only:
        os.environ["READ_ONLY_MODE"] = "1"
        console.print("🔒 [yellow]Read-only mode enabled[/yellow]")
    
    if api_key:
        os.environ["CLINICAL_DATA_PLATFORM_SECURITY__API_KEY"] = api_key
        console.print("🔑 [dim]API key configured[/dim]")
    
    if config_file:
        os.environ["CLINICAL_DATA_PLATFORM_CONFIG_FILE"] = str(config_file)
        console.print(f"⚙️ [dim]Using config: {config_file}[/dim]")
    
    # Display server information
    console.print(f"🌐 Server will start at: [cyan]http://{host}:{port}[/cyan]")
    console.print(f"📚 API docs: [cyan]http://{host}:{port}/docs[/cyan]")
    console.print(f"🔍 Health check: [cyan]http://{host}:{port}/health[/cyan]")
    
    if reload:
        console.print("🔄 [yellow]Auto-reload enabled (development mode)[/yellow]")
    
    console.print("\n[dim]Press Ctrl+C to stop the server[/dim]")
    
    try:
        # Import the FastAPI app
        from clinical_platform.api.main import app
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level=log_level,
            access_log=True
        )
        
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Server stopped by user[/yellow]")
    except ImportError as e:
        console.print(f"❌ [red]Failed to import API module: {e}[/red]")
        console.print("💡 [dim]Make sure you have installed the clinical-data-platform package[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [red]Failed to start server: {e}[/red]")
        sys.exit(1)


@api_app.command("test")
def test_api(
    url: str = typer.Option("http://127.0.0.1:8000", "--url", help="API base URL"),
    timeout: int = typer.Option(30, "--timeout", help="Request timeout in seconds"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for testing"),
):
    """
    🧪 Test API endpoints and connectivity
    
    Examples:
        clinical-data-platform api test                           # Test default URL
        clinical-data-platform api test --url http://api.example.com  # Test custom URL
    """
    
    import requests
    
    console.print(f"🧪 [bold]Testing API at: {url}[/bold]")
    
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    tests = [
        ("Health Check", "GET", "/health", None),
        ("OpenAPI Schema", "GET", "/openapi.json", None),
        ("API Documentation", "GET", "/docs", None),
    ]
    
    if api_key:
        tests.extend([
            ("List Studies", "GET", "/studies", None),
            ("ML Scoring", "POST", "/score", {"AGE": 45, "AE_COUNT": 1, "SEVERE_AE_COUNT": 0}),
        ])
    
    results = []
    
    for test_name, method, endpoint, data in tests:
        try:
            full_url = f"{url.rstrip('/')}{endpoint}"
            
            if method == "GET":
                response = requests.get(full_url, headers=headers, timeout=timeout)
            elif method == "POST":
                response = requests.post(full_url, headers=headers, json=data, timeout=timeout)
            
            success = response.status_code < 400
            status_icon = "✅" if success else "❌"
            
            console.print(f"{status_icon} {test_name}: {response.status_code}")
            
            results.append((test_name, success, response.status_code))
            
        except requests.RequestException as e:
            console.print(f"❌ {test_name}: Connection failed - {e}")
            results.append((test_name, False, str(e)))
    
    # Summary
    successful_tests = sum(1 for _, success, _ in results if success)
    total_tests = len(results)
    
    console.print(f"\n📊 [bold]Test Results: {successful_tests}/{total_tests} passed[/bold]")
    
    if successful_tests == total_tests:
        console.print("🎉 [green]All tests passed![/green]")
    else:
        console.print("⚠️ [yellow]Some tests failed. Check API server status.[/yellow]")
        sys.exit(1)


@api_app.command("schema")
def get_schema(
    url: str = typer.Option("http://127.0.0.1:8000", "--url", help="API base URL"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for schema"),
    format: str = typer.Option("json", "--format", help="Output format: json, yaml")
):
    """
    📋 Get OpenAPI schema from running API server
    
    Examples:
        clinical-data-platform api schema                          # Print to console
        clinical-data-platform api schema -o schema.json          # Save to file
        clinical-data-platform api schema --format yaml           # YAML format
    """
    
    import requests
    import json
    
    console.print(f"📋 [bold]Fetching OpenAPI schema from: {url}[/bold]")
    
    try:
        response = requests.get(f"{url.rstrip('/')}/openapi.json", timeout=30)
        response.raise_for_status()
        
        schema = response.json()
        
        if format.lower() == "yaml":
            import yaml
            schema_content = yaml.dump(schema, default_flow_style=False)
        else:
            schema_content = json.dumps(schema, indent=2)
        
        if output:
            with open(output, "w") as f:
                f.write(schema_content)
            console.print(f"✅ Schema saved to: [cyan]{output}[/cyan]")
        else:
            console.print(schema_content)
            
    except requests.RequestException as e:
        console.print(f"❌ [red]Failed to fetch schema: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"❌ [red]Error processing schema: {e}[/red]")
        sys.exit(1)


@api_app.command("docs")
def open_docs(
    url: str = typer.Option("http://127.0.0.1:8000", "--url", help="API base URL"),
):
    """
    📚 Open API documentation in browser
    """
    
    import webbrowser
    
    docs_url = f"{url.rstrip('/')}/docs"
    console.print(f"📚 Opening API documentation: [cyan]{docs_url}[/cyan]")
    
    try:
        webbrowser.open(docs_url)
        console.print("✅ Documentation opened in browser")
    except Exception as e:
        console.print(f"❌ Failed to open browser: {e}")
        console.print(f"💡 Manually visit: {docs_url}")


def main():
    """Entry point for standalone API CLI."""
    api_app()


if __name__ == "__main__":
    main()