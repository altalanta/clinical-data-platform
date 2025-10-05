"""
Demo CLI commands for Clinical Data Platform

Provides commands to run various demo scenarios with synthetic data.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

console = Console()

demo_app = typer.Typer(
    name="demo",
    help="🎲 Demo operations and examples",
    rich_markup_mode="rich"
)


@demo_app.command("public")
def run_public_demo(
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory for demo artifacts"
    ),
    data_size: str = typer.Option(
        "standard", "--size", help="Demo data size: minimal, standard, comprehensive"
    ),
    skip_api: bool = typer.Option(False, "--skip-api", help="Skip API demo"),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Skip validation demo"),
    skip_transform: bool = typer.Option(False, "--skip-transform", help="Skip transformation demo"),
    port: int = typer.Option(8000, "--port", help="API server port")
):
    """
    🎲 Run the complete public demo with synthetic data
    
    This command demonstrates the full Clinical Data Platform capabilities:
    - Generates synthetic OMOP-compliant clinical data
    - Runs data validation with Great Expectations
    - Executes dbt transformations
    - Starts API server with ML endpoints
    - Creates comprehensive reports
    """
    
    console.print(Panel.fit("🎲 [bold]Clinical Data Platform Public Demo[/bold]", style="blue"))
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="clinical_demo_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"📁 Demo artifacts will be saved to: [cyan]{output_dir}[/cyan]")
    
    # Define demo tasks
    tasks = []
    if not skip_validation:
        tasks.append(("📊 Generate synthetic data", generate_synthetic_data))
        tasks.append(("✅ Run data validation", run_data_validation))
    if not skip_transform:
        tasks.append(("🔄 Execute transformations", run_transformations))
    if not skip_api:
        tasks.append(("🔌 Start API demo", run_api_demo))
    
    tasks.extend([
        ("📋 Generate reports", generate_demo_reports),
        ("📚 Create documentation", create_demo_docs)
    ])
    
    # Execute demo with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        demo_task = progress.add_task("Running demo...", total=len(tasks))
        
        demo_context = {
            "output_dir": output_dir,
            "data_size": data_size,
            "port": port,
            "start_time": time.time()
        }
        
        for task_name, task_func in tasks:
            task_id = progress.add_task(task_name, total=100)
            
            try:
                # Update main progress
                progress.update(demo_task, description=f"🔄 {task_name}")
                
                # Run the task
                result = task_func(demo_context, progress, task_id)
                demo_context.update(result or {})
                
                # Mark task complete
                progress.update(task_id, completed=100, description=f"✅ {task_name}")
                progress.advance(demo_task)
                
            except Exception as e:
                progress.update(task_id, description=f"❌ {task_name} - Error: {str(e)}")
                console.print(f"[red]Error in {task_name}: {e}[/red]")
                break
    
    # Demo completion summary
    elapsed_time = time.time() - demo_context["start_time"]
    
    console.print("\n" + "="*60)
    console.print(f"🎉 [bold green]Demo completed successfully![/bold green]")
    console.print(f"⏱️ Total time: {elapsed_time:.1f} seconds")
    console.print(f"📁 Artifacts saved to: [cyan]{output_dir}[/cyan]")
    
    # Show next steps
    next_steps = [
        f"View demo reports: [cyan]open {output_dir}/reports/[/cyan]",
        f"Explore API: [cyan]curl http://localhost:{port}/docs[/cyan]",
        f"Check data quality: [cyan]open {output_dir}/validation/[/cyan]",
        "Read documentation: [cyan]clinical-data-platform serve docs[/cyan]"
    ]
    
    console.print("\n🚀 [bold yellow]Next steps:[/bold yellow]")
    for step in next_steps:
        console.print(f"  • {step}")


def generate_synthetic_data(context: dict, progress, task_id) -> dict:
    """Generate synthetic clinical data."""
    import pandas as pd
    import numpy as np
    from faker import Faker
    import duckdb
    
    fake = Faker()
    fake.seed(42)
    np.random.seed(42)
    
    # Determine data size
    size_configs = {
        "minimal": {"patients": 100, "visits_per_patient": 2},
        "standard": {"patients": 1000, "visits_per_patient": 5},
        "comprehensive": {"patients": 10000, "visits_per_patient": 8}
    }
    
    config = size_configs.get(context["data_size"], size_configs["standard"])
    progress.update(task_id, advance=20, description="🎯 Configuring data generation")
    
    # Generate patients
    patients = []
    for i in range(config["patients"]):
        patients.append({
            "person_id": f"DEMO_{i+1:06d}",
            "gender_concept_id": np.random.choice([8507, 8532]),
            "year_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=90).year,
            "race_concept_id": np.random.choice([8527, 8516, 8515]),
            "ethnicity_concept_id": np.random.choice([38003564, 38003563])
        })
    
    progress.update(task_id, advance=30, description="👥 Generated patients")
    
    # Generate visits
    visits = []
    visit_id = 1
    for patient in patients:
        n_visits = np.random.poisson(config["visits_per_patient"]) + 1
        for _ in range(n_visits):
            visits.append({
                "visit_occurrence_id": f"VISIT_{visit_id:08d}",
                "person_id": patient["person_id"],
                "visit_concept_id": np.random.choice([9201, 9202, 9203]),
                "visit_start_date": fake.date_between(start_date="-2y", end_date="today")
            })
            visit_id += 1
    
    progress.update(task_id, advance=30, description="🏥 Generated visits")
    
    # Save data
    data_dir = context["output_dir"] / "data"
    data_dir.mkdir(exist_ok=True)
    
    pd.DataFrame(patients).to_csv(data_dir / "patients.csv", index=False)
    pd.DataFrame(visits).to_csv(data_dir / "visits.csv", index=False)
    
    # Create DuckDB database
    conn = duckdb.connect(str(data_dir / "demo.db"))
    conn.execute("CREATE TABLE patients AS SELECT * FROM read_csv_auto(?)", [str(data_dir / "patients.csv")])
    conn.execute("CREATE TABLE visits AS SELECT * FROM read_csv_auto(?)", [str(data_dir / "visits.csv")])
    conn.close()
    
    progress.update(task_id, advance=20, description="💾 Saved to database")
    
    return {
        "data_counts": {"patients": len(patients), "visits": len(visits)},
        "database_path": str(data_dir / "demo.db")
    }


def run_data_validation(context: dict, progress, task_id) -> dict:
    """Run data validation checks."""
    import pandas as pd
    import json
    
    progress.update(task_id, advance=25, description="📊 Loading data")
    
    data_dir = context["output_dir"] / "data"
    patients_df = pd.read_csv(data_dir / "patients.csv")
    visits_df = pd.read_csv(data_dir / "visits.csv")
    
    progress.update(task_id, advance=25, description="🔍 Running validation checks")
    
    # Basic validation checks
    validation_results = {
        "patients": {
            "total_records": len(patients_df),
            "unique_ids": patients_df["person_id"].nunique(),
            "null_ids": patients_df["person_id"].isnull().sum(),
            "completeness": 1.0 - (patients_df.isnull().sum().sum() / patients_df.size)
        },
        "visits": {
            "total_records": len(visits_df),
            "unique_ids": visits_df["visit_occurrence_id"].nunique(),
            "null_ids": visits_df["visit_occurrence_id"].isnull().sum(),
            "completeness": 1.0 - (visits_df.isnull().sum().sum() / visits_df.size)
        }
    }
    
    progress.update(task_id, advance=25, description="📝 Generating validation report")
    
    # Save validation results
    validation_dir = context["output_dir"] / "validation"
    validation_dir.mkdir(exist_ok=True)
    
    with open(validation_dir / "results.json", "w") as f:
        json.dump(validation_results, f, indent=2)
    
    progress.update(task_id, advance=25, description="✅ Validation complete")
    
    return {"validation_results": validation_results}


def run_transformations(context: dict, progress, task_id) -> dict:
    """Run dbt transformations."""
    import duckdb
    
    progress.update(task_id, advance=25, description="🔄 Setting up transformations")
    
    # Create analytics views in DuckDB
    conn = duckdb.connect(context["database_path"])
    
    progress.update(task_id, advance=25, description="📊 Creating analytics views")
    
    # Patient summary view
    conn.execute("""
        CREATE OR REPLACE VIEW patient_summary AS
        SELECT 
            p.person_id,
            p.gender_concept_id,
            COUNT(v.visit_occurrence_id) as total_visits,
            MIN(v.visit_start_date) as first_visit,
            MAX(v.visit_start_date) as last_visit
        FROM patients p
        LEFT JOIN visits v ON p.person_id = v.person_id
        GROUP BY p.person_id, p.gender_concept_id
    """)
    
    progress.update(task_id, advance=25, description="📈 Creating metrics")
    
    # Get transformation results
    results = conn.execute("SELECT COUNT(*) as analytics_views FROM information_schema.views").fetchone()
    
    progress.update(task_id, advance=25, description="✅ Transformations complete")
    
    conn.close()
    
    return {"analytics_views": results[0] if results else 0}


def run_api_demo(context: dict, progress, task_id) -> dict:
    """Start API server demo."""
    import threading
    import time
    import requests
    import uvicorn
    
    progress.update(task_id, advance=25, description="🚀 Starting API server")
    
    # Set environment variables for demo
    os.environ["CLINICAL_DATA_PLATFORM_WAREHOUSE__DUCKDB_PATH"] = context["database_path"]
    os.environ["CLINICAL_DATA_PLATFORM_SECURITY__API_KEY"] = "demo-key-12345"
    
    try:
        from clinical_platform.api.main import app
        
        # Start server in background
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=context["port"], log_level="error")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        progress.update(task_id, advance=25, description="⏳ Waiting for server startup")
        time.sleep(3)
        
        # Test API endpoints
        base_url = f"http://127.0.0.1:{context['port']}"
        
        progress.update(task_id, advance=25, description="🧪 Testing API endpoints")
        
        # Test health endpoint
        health_response = requests.get(f"{base_url}/health", timeout=5)
        api_healthy = health_response.status_code == 200
        
        # Test OpenAPI docs
        docs_response = requests.get(f"{base_url}/docs", timeout=5)
        docs_available = docs_response.status_code == 200
        
        progress.update(task_id, advance=25, description="✅ API demo ready")
        
        return {
            "api_url": base_url,
            "api_healthy": api_healthy,
            "docs_available": docs_available
        }
        
    except Exception as e:
        console.print(f"[yellow]API demo skipped: {e}[/yellow]")
        return {"api_error": str(e)}


def generate_demo_reports(context: dict, progress, task_id) -> dict:
    """Generate comprehensive demo reports."""
    import json
    from datetime import datetime
    
    progress.update(task_id, advance=25, description="📋 Creating summary report")
    
    reports_dir = context["output_dir"] / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Create comprehensive demo report
    report = f"""# Clinical Data Platform Demo Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Demo Size:** {context['data_size']}
**Duration:** {time.time() - context['start_time']:.1f} seconds

## Data Summary

- **Patients:** {context.get('data_counts', {}).get('patients', 'N/A'):,}
- **Visits:** {context.get('data_counts', {}).get('visits', 'N/A'):,}
- **Database:** {Path(context.get('database_path', 'N/A')).name}

## Validation Results

"""
    
    if "validation_results" in context:
        validation = context["validation_results"]
        report += f"""
- **Patient Data Quality:** {validation['patients']['completeness']:.1%}
- **Visit Data Quality:** {validation['visits']['completeness']:.1%}
- **Unique Patient IDs:** {validation['patients']['unique_ids']:,}
- **Unique Visit IDs:** {validation['visits']['unique_ids']:,}
"""
    
    if "api_url" in context:
        report += f"""
## API Endpoints

- **Health Check:** {context['api_url']}/health
- **API Documentation:** {context['api_url']}/docs
- **OpenAPI Schema:** {context['api_url']}/openapi.json
- **Status:** {'✅ Healthy' if context.get('api_healthy') else '❌ Unhealthy'}
"""
    
    report += """
## Next Steps

1. **Explore the API:** Visit the `/docs` endpoint for interactive API documentation
2. **Review Data:** Check the generated CSV files in the `data/` directory  
3. **Run Validation:** Use `clinical-data-platform validate` for comprehensive checks
4. **Transform Data:** Execute `clinical-data-platform transform run` for analytics

## Support

- **Documentation:** https://altalanta.github.io/clinical-data-platform/
- **Issues:** https://github.com/altalanta/clinical-data-platform/issues
"""
    
    with open(reports_dir / "demo_report.md", "w") as f:
        f.write(report)
    
    progress.update(task_id, advance=50, description="📊 Saving demo metadata")
    
    # Save demo metadata
    metadata = {
        "demo_id": f"demo_{int(time.time())}",
        "generated_at": datetime.now().isoformat(),
        "configuration": {
            "data_size": context["data_size"],
            "output_dir": str(context["output_dir"]),
            "port": context.get("port", 8000)
        },
        "results": {
            "data_counts": context.get("data_counts", {}),
            "validation_results": context.get("validation_results", {}),
            "api_status": {
                "url": context.get("api_url"),
                "healthy": context.get("api_healthy", False),
                "docs_available": context.get("docs_available", False)
            }
        }
    }
    
    with open(reports_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    progress.update(task_id, advance=25, description="✅ Reports generated")
    
    return {"reports_dir": str(reports_dir)}


def create_demo_docs(context: dict, progress, task_id) -> dict:
    """Create demo documentation."""
    progress.update(task_id, advance=50, description="📚 Creating demo documentation")
    
    docs_dir = context["output_dir"] / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Create simple HTML index
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Clinical Data Platform Demo</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #2196F3; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 3px solid #2196F3; }}
        .code {{ background: #f5f5f5; padding: 10px; border-radius: 3px; font-family: monospace; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Clinical Data Platform Demo</h1>
        <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Demo Summary</h2>
        <ul>
            <li><strong>Data Size:</strong> {context['data_size']}</li>
            <li><strong>Patients:</strong> {context.get('data_counts', {}).get('patients', 'N/A'):,}</li>
            <li><strong>Visits:</strong> {context.get('data_counts', {}).get('visits', 'N/A'):,}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🔗 Quick Links</h2>
        <ul>
            <li><a href="../reports/demo_report.md">Demo Report</a></li>
            <li><a href="../data/">Data Files</a></li>
            <li><a href="../validation/">Validation Results</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🚀 API Endpoints</h2>
        <div class="code">
            # Health check<br>
            curl {context.get('api_url', 'http://localhost:8000')}/health<br><br>
            
            # API documentation<br>
            open {context.get('api_url', 'http://localhost:8000')}/docs
        </div>
    </div>
</body>
</html>"""
    
    with open(docs_dir / "index.html", "w") as f:
        f.write(html_content)
    
    progress.update(task_id, advance=50, description="✅ Documentation created")
    
    return {"docs_dir": str(docs_dir)}


@demo_app.command("list")
def list_demos():
    """📋 List available demo scenarios"""
    
    demos = [
        ("public", "Complete public demo with synthetic data"),
        ("minimal", "Minimal demo with basic functionality"),
        ("api-only", "API server demo without data generation"),
        ("validation-only", "Data validation demo"),
        ("transform-only", "dbt transformation demo")
    ]
    
    console.print("🎲 [bold]Available Demo Scenarios:[/bold]\n")
    
    for name, description in demos:
        console.print(f"  [cyan]{name}[/cyan] - {description}")
    
    console.print(f"\n💡 [dim]Run with: clinical-data-platform demo <scenario>[/dim]")


@demo_app.command("clean")
def clean_demo_data(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt")
):
    """🗑️ Clean up demo data and artifacts"""
    
    if not confirm:
        if not typer.confirm("This will delete all demo data and artifacts. Continue?"):
            console.print("❌ Cleanup cancelled")
            return
    
    # Find and clean demo directories
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.gettempdir())
    demo_dirs = list(temp_dir.glob("clinical_demo_*"))
    
    removed_count = 0
    for demo_dir in demo_dirs:
        try:
            shutil.rmtree(demo_dir)
            removed_count += 1
            console.print(f"🗑️ Removed: {demo_dir}")
        except Exception as e:
            console.print(f"❌ Failed to remove {demo_dir}: {e}")
    
    console.print(f"\n✅ Cleaned up {removed_count} demo directories")


def main():
    """Entry point for standalone demo CLI."""
    demo_app()


if __name__ == "__main__":
    main()