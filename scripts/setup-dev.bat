@echo off
REM CBT Journal Development Environment Setup Script (Windows)
REM This script sets up a complete development environment

echo 🚀 CBT Journal - Development Environment Setup
echo ==============================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.9 or higher.
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ pip is not installed. Please install pip.
    pause
    exit /b 1
)

echo ✅ pip found

REM Create virtual environment
echo ℹ️  Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ⚠️  Virtual environment already exists
)

REM Activate virtual environment
echo ℹ️  Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ℹ️  Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo ℹ️  Installing dependencies...
pip install -r requirements-dev.txt

echo ✅ Dependencies installed

REM Setup pre-commit hooks
echo ℹ️  Setting up pre-commit hooks...
pre-commit install
pre-commit install --hook-type pre-push

echo ✅ Pre-commit hooks installed

REM Create necessary directories
echo ℹ️  Creating necessary directories...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "tmp" mkdir tmp

echo ✅ Directories created

REM Format code
echo ℹ️  Formatting code...
black cbt_journal\ tools\ tests\
isort cbt_journal\ tools\ tests\

echo ✅ Code formatted

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Docker is not installed. You'll need Docker to run Qdrant locally.
    goto :skip_docker
)

echo ℹ️  Starting Qdrant with Docker...
cd docker
docker-compose up -d
cd ..

echo ✅ Qdrant started

:skip_docker

echo.
echo ✅ Development environment setup completed!
echo.
echo 📋 Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Run tests: python -m pytest tests\
echo 3. Run formatting: black cbt_journal\ tools\ tests\
echo 4. Start coding! 🎉
echo.
echo 📖 Available commands:
echo   python -m black .           - Format code
echo   python -m flake8 .          - Run linting
echo   python -m mypy cbt_journal\ - Type checking
echo   python -m pytest tests\     - Run tests
echo   pre-commit run --all-files  - Run all checks
echo.

pause