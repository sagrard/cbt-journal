# CBT Journal - Development Guide

This guide helps you set up and maintain a high-quality development environment for the CBT Journal project.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip
- Docker (optional, for running Qdrant locally)
- Git

### Setup Development Environment

**Linux/macOS:**
```bash
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh
```

**Windows:**
```batch
scripts\setup-dev.bat
```

**Manual Setup:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate.bat  # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install

# Start Qdrant (optional)
docker-compose -f docker/docker-compose.yaml up -d
```

## 🛠️ Development Workflow

### Code Quality Tools

We use several tools to maintain high code quality:

#### 1. **Black** - Code Formatting
```bash
# Format code
black cbt_journal/ tools/ tests/

# Check formatting
black --check cbt_journal/ tools/ tests/
```

#### 2. **isort** - Import Sorting
```bash
# Sort imports
isort cbt_journal/ tools/ tests/

# Check import sorting
isort --check-only cbt_journal/ tools/ tests/
```

#### 3. **flake8** - Linting
```bash
# Run linting
flake8 cbt_journal/ tools/ tests/
```

#### 4. **mypy** - Type Checking
```bash
# Run type checking
mypy cbt_journal/ tools/
```

#### 5. **bandit** - Security Scanning
```bash
# Run security scan
bandit -r cbt_journal/ tools/
```

### Using Make Commands

We provide a Makefile for common development tasks:

```bash
# Show all available commands
make help

# Install dependencies
make install-dev

# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run security checks
make security

# Run all quality checks
make all-checks

# Run tests
make test

# Run tests with coverage
make test-cov

# Quick development check
make dev-check
```

### Pre-commit Hooks

Pre-commit hooks run automatically on every commit to ensure code quality:

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files

# Skip hooks (not recommended)
git commit --no-verify
```

## 📋 Code Standards

### Python Code Style

- **Line Length**: 100 characters
- **Formatting**: Black with default settings
- **Import Sorting**: isort with Black profile
- **Type Hints**: Required for all functions
- **Docstrings**: Google style, required for all public functions

### Example Code

```python
from typing import Dict, List, Optional

from qdrant_client import QdrantClient


def process_session_data(
    session_data: Dict[str, Any],
    client: QdrantClient,
    collection_name: str = "default",
) -> Optional[List[Dict[str, Any]]]:
    """
    Process session data and return formatted results.
    
    Args:
        session_data: Raw session data from CBT journal
        client: Qdrant client instance
        collection_name: Name of the collection to use
        
    Returns:
        Formatted session data or None if processing fails
        
    Raises:
        ProcessingError: If session data is invalid
    """
    # Implementation here
    pass
```

### Testing Standards

- **Test Coverage**: Aim for >90% coverage
- **Test Organization**: Separate unit and integration tests
- **Test Naming**: Descriptive test names
- **Fixtures**: Use pytest fixtures for common setup

```python
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    client = Mock(spec=QdrantClient)
    # Setup mock behavior
    return client

def test_process_session_data_success(mock_qdrant_client):
    """Test successful session data processing."""
    # Test implementation
    pass

def test_process_session_data_invalid_input(mock_qdrant_client):
    """Test handling of invalid input data."""
    # Test implementation
    pass
```

## 🔧 Configuration

### Tool Configuration

All tools are configured in `pyproject.toml`:

- **Black**: Line length 100, Python 3.9+ target
- **isort**: Black profile, first-party imports for `cbt_journal`
- **mypy**: Strict type checking enabled
- **pytest**: Test discovery and coverage settings
- **coverage**: HTML reports, exclude test files

### Environment Configuration

Create a `.env` file for local configuration:

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6334
QDRANT_COLLECTION=cbt_journal_sessions

# Cost Control
MAX_COST_PER_SESSION=0.50
MAX_DAILY_COST=5.00
MAX_MONTHLY_COST=100.00

# Development
DEBUG=true
LOG_LEVEL=INFO
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_vector_store.py -v

# Run specific test
pytest tests/test_vector_store.py::test_store_session -v

# Run integration tests only
pytest tests/ -m integration

# Run unit tests only
pytest tests/ -m "not integration"
```

### Test Organization

```
tests/
├── test_vector_store.py      # Vector store unit tests
├── test_cost_control.py      # Cost control unit tests
├── test_qdrant.py           # Qdrant integration tests
└── conftest.py              # Shared fixtures
```

## 🐛 Debugging

### Common Issues

1. **Pre-commit hooks failing**: Run `pre-commit run --all-files` to see all issues
2. **Type checking errors**: Add type hints or use `# type: ignore` comments
3. **Import errors**: Check import order with `isort --check-only`
4. **Docker issues**: Ensure Docker is running and ports are available

### Debugging Tools

```bash
# Check code formatting
black --check .

# Check import sorting
isort --check-only .

# Run linting with verbose output
flake8 --verbose .

# Run type checking with detailed output
mypy --show-error-codes cbt_journal/
```

## 📚 Documentation

### Adding Documentation

- Update docstrings for all public functions
- Add type hints for all parameters and return values
- Include examples in docstrings when helpful
- Update this guide when adding new tools or workflows

### Documentation Standards

- Use Google-style docstrings
- Include type information in docstrings
- Provide usage examples
- Document exceptions that may be raised

## 🚀 Deployment

### Preparing for Production

1. **Run all quality checks**: `make all-checks`
2. **Ensure tests pass**: `make test-cov`
3. **Update version**: Update version in `pyproject.toml`
4. **Update changelog**: Document changes
5. **Create release**: Tag and push to repository

### CI/CD Integration

The project is ready for CI/CD integration with GitHub Actions:

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: 3.9
      - run: pip install -r requirements-dev.txt
      - run: make all-checks
      - run: make test-cov
```

## 📞 Support

For development questions or issues:

1. Check this documentation
2. Review tool configuration files
3. Run `make help` for available commands
4. Check the project's issue tracker

## 🔄 Maintenance

### Regular Maintenance Tasks

- Update dependencies monthly
- Review and update code quality rules
- Monitor test coverage
- Update documentation
- Review security alerts

### Dependency Updates

```bash
# Update dependencies
pip-compile requirements.in
pip-compile requirements-dev.in

# Install updates
pip install -r requirements-dev.txt

# Run tests after updates
make test
```