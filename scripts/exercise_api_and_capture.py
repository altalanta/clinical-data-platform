import json
from pathlib import Path

from fastapi.testclient import TestClient

from clinical_data_platform.api import app

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets" / "demo" / "api"
ASSETS.mkdir(parents=True, exist_ok=True)

client = TestClient(app)

health_response = client.get("/health")
(ASSETS / "curl_health.txt").write_text(
    json.dumps(health_response.json(), indent=2) + "\n",
    encoding="utf-8",
)

payload = {"features": [5.1, 3.5, 1.4, 0.2]}
predict_response = client.post("/predict", json=payload)
(ASSETS / "curl_predict.json").write_text(
    json.dumps(predict_response.json(), indent=2) + "\n",
    encoding="utf-8",
)

commands = """curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"features":[5.1,3.5,1.4,0.2]}'
"""
(ASSETS / "curl_commands.txt").write_text(commands, encoding="utf-8")

print("API outputs written to", ASSETS)
