.PHONY: demo demo.dbt demo.schema demo.ml demo.api demo.gif demo.clean

demo: demo.dbt demo.schema demo.ml demo.api demo.gif
	@echo "Demo artifacts in docs/assets/demo/"

demo.dbt:
	python scripts/run_dbt_and_capture.py

demo.schema:
	python scripts/generate_star_schema_diagram.py

demo.ml:
	python -m pip install 'mlflow>=2.15' 'scikit-learn>=1.5' 'matplotlib>=3.8'
	python scripts/run_demo_mlflow.py

demo.api:
	python -m pip install 'fastapi>=0.111' 'uvicorn>=0.30' 'httpx>=0.27'
	python scripts/exercise_api_and_capture.py

demo.gif:
	python -m pip install pillow imageio
	python scripts/make_demo_gif.py

demo.clean:
	rm -rf data/demo.duckdb mlruns
	rm -rf docs/assets/demo/*
