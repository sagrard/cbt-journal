# Makefile for CBT Journal project
# Provides easy commands for development workflow

.PHONY: help install install-dev format lint type-check security test test-cov clean pre-commit-install pre-commit-run all-checks

# Default target
help:
	@echo "CBT Journal - Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  install              Install production dependencies"
	@echo "  install-dev          Install development dependencies"
	@echo "  pre-commit-install   Install pre-commit hooks"
	@echo ""
	@echo "Code Quality:"
	@echo "  format              Format code with black and isort"
	@echo "  lint                Run flake8 linting"
	@echo "  type-check          Run mypy type checking"
	@echo "  security            Run security checks (bandit + safety)"
	@echo "  all-checks          Run all code quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  test                Run tests with pytest"
	@echo "  test-cov            Run tests with coverage report"
	@echo "  test-integration    Run integration tests only"
	@echo "  test-unit           Run unit tests only"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean               Clean up build artifacts and cache"
	@echo "  pre-commit-run      Run pre-commit hooks on all files"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up           Start Qdrant with Docker"
	@echo "  docker-down         Stop Qdrant Docker containers"
	@echo "  docker-logs         Show Qdrant logs"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

pre-commit-install:
	pre-commit install
	pre-commit install --hook-type pre-push

# Code formatting
format:
	@echo "Running isort..."
	isort cbt_journal/ tools/ tests/
	@echo "Running black..."
	black cbt_journal/ tools/ tests/
	@echo "✅ Code formatting completed"

# Linting
lint:
	@echo "Running flake8..."
	flake8 cbt_journal/ tools/ tests/
	@echo "✅ Linting completed"

# Type checking
type-check:
	@echo "Running mypy..."
	mypy cbt_journal/ tools/
	@echo "✅ Type checking completed"

# Security checks
security:
	@echo "Running bandit..."
	bandit -r cbt_journal/ tools/
	@echo "Running safety..."
	safety check --file requirements.txt
	@echo "✅ Security checks completed"

# All code quality checks
all-checks: format lint type-check security
	@echo "✅ All code quality checks completed"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=cbt_journal --cov-report=html --cov-report=term-missing

test-integration:
	@echo "Running integration tests..."
	pytest tests/ -v -m integration

test-unit:
	@echo "Running unit tests..."
	pytest tests/ -v -m "not integration"

# Docker operations
docker-up:
	@echo "Starting Qdrant with Docker..."
	docker-compose -f docker/docker-compose.yaml up -d

docker-down:
	@echo "Stopping Qdrant Docker containers..."
	docker-compose -f docker/docker-compose.yaml down

docker-logs:
	@echo "Showing Qdrant logs..."
	docker-compose -f docker/docker-compose.yaml logs -f qdrant

# Maintenance
clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .tox/
	@echo "✅ Cleanup completed"

pre-commit-run:
	@echo "Running pre-commit hooks on all files..."
	pre-commit run --all-files

# Setup complete development environment
setup-dev: install-dev pre-commit-install
	@echo "✅ Development environment setup completed"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make docker-up' to start Qdrant"
	@echo "2. Run 'make all-checks' to verify code quality"
	@echo "3. Run 'make test' to run tests"

# Quick development workflow
dev-check: format lint type-check test
	@echo "✅ Development checks completed - ready to commit!"