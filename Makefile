.PHONY: install test validate train score api monitor docker-up docker-down

install:
	pip install -r requirements.txt

validate:
	PYTHONPATH=. python scripts/validate_data.py

test:
	PYTHONPATH=. pytest -q

train:
	PYTHONPATH=. python scripts/train.py

score:
	PYTHONPATH=. python scripts/score_batch.py --input "data/raw/Dataset Renovacion_prestamo.csv"

api:
	PYTHONPATH=. uvicorn api.main:app --reload --port 8001

monitor:
	PYTHONPATH=. python scripts/monitor.py --month 201509

docker-up:
	docker compose up --build

docker-down:
	docker compose down
