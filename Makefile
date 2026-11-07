.PHONY: setup run run-prefect serve-prefect mlflow-ui test clean
setup:          ## create venv + install deps
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
run:            ## full pipeline, plain-Python orchestrator
	python run_pipeline.py
run-prefect:    ## full pipeline as a Prefect flow
	python dags/recommendation_flow.py
serve-prefect:  ## schedule the flow daily at 02:00 IST
	python dags/recommendation_flow.py --serve
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
test:
	python -m pytest -q tests
clean:
	rm -rf data/raw/* data/processed/* data/features/* data/errors/* models/*.pkl models/*.json mlflow.db mlruns logs
