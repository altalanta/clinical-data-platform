from pathlib import Path

from PIL import Image, ImageDraw
import imageio

out_dir = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo"
out_dir.mkdir(parents=True, exist_ok=True)

frames = []
for text in ("dbt build ✓", "MLflow run ✓", "/health ✓", "/predict ✓"):
    image = Image.new("RGB", (720, 200), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 80), text, fill="black")
    frames.append(image)

imageio.mimsave(out_dir / "demo.gif", frames, duration=0.8)
print("Wrote", out_dir / "demo.gif")
