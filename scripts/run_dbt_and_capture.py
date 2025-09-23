import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DBT_DIR = ROOT / "analytics" / "dbt"
ASSETS = ROOT / "docs" / "assets" / "demo" / "dbt"
ASSETS.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env.setdefault("CDP_DUCKDB_PATH", str((ROOT / "data" / "demo.duckdb").resolve()))
env.setdefault("DBT_PROFILES_DIR", str(DBT_DIR))
bin_dir = Path(sys.executable).parent
env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

data_dir = ROOT / "data"
data_dir.mkdir(parents=True, exist_ok=True)

subprocess.check_call([sys.executable, "-m", "pip", "install", "dbt-duckdb>=1.8"], cwd=str(ROOT))
subprocess.check_call(["dbt", "deps"], cwd=str(DBT_DIR), env=env)
subprocess.check_call(["dbt", "seed", "--full-refresh"], cwd=str(DBT_DIR), env=env)
subprocess.check_call(["dbt", "build", "--no-write-json"], cwd=str(DBT_DIR), env=env)

target = DBT_DIR / "target"
target_files = []
if target.exists():
    for name in ["run_results.json", "manifest.json"]:
        path = target / name
        if path.exists():
            shutil.copy2(path, ASSETS / name)
    target_files = [name for name in os.listdir(target) if name.endswith(".json")]

summary = {
    "duckdb_path": env["CDP_DUCKDB_PATH"],
    "target_files": target_files,
}
with open(ASSETS / "summary.json", "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)

print("dbt artifacts captured ->", ASSETS)
