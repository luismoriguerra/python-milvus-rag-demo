.PHONY: install run clean lint format help

install:
	pip install -r requirements.txt

run:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete

lint:
	pip install flake8
	flake8 .

format:
	pip install black
	black .

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies from requirements.txt"
	@echo "  make run      - Run the FastAPI server in development mode"
	@echo "  make clean    - Remove Python cache files"
	@echo "  make lint     - Run flake8 linter"
	@echo "  make format   - Format code using black" 