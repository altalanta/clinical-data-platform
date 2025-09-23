import os
import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
db_path = os.environ.get("CDP_DUCKDB_PATH", str((ROOT / "data" / "demo.duckdb").resolve()))
out_dir = ROOT / "docs" / "assets" / "demo" / "schema"
out_dir.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(db_path)
try:
    tables = [
        row[0]
        for row in con.execute(
            "select table_name from information_schema.tables where table_schema='main'"
        ).fetchall()
    ]
finally:
    con.close()

nodes = [f'"{table}" [shape=box, style=rounded]' for table in tables]
edges = []
if "fact_visits" in tables and "dim_patients" in tables:
    edges.append('"fact_visits" -> "dim_patients" [label="patient_id"]')
if "fact_visits" in tables and "dim_providers" in tables:
    edges.append('"fact_visits" -> "dim_providers" [label="provider_id"]')

dot = "digraph G {\nrankdir=LR;\n" + "\n".join(nodes) + "\n" + "\n".join(edges) + "\n}\n"
dot_path = out_dir / "star_schema.dot"
dot_path.write_text(dot, encoding="utf-8")

png_path = out_dir / "star_schema.png"
try:
    subprocess.check_call(["dot", "-Tpng", str(dot_path), "-o", str(png_path)])
except Exception:
    (out_dir / "star_schema.txt").write_text(dot, encoding="utf-8")
    try:
        from PIL import Image, ImageDraw
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
        from PIL import Image, ImageDraw

    image = Image.new("RGB", (720, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), "fact_visits ↔ dimensions", fill="black")
    image.save(png_path)

print("schema diagram written to", out_dir)
