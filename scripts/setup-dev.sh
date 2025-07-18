#!/bin/bash
# CBT Journal Development Environment Setup Script
# This script sets up a complete development environment

set -e  # Exit on any error

echo "🚀 CBT Journal - Development Environment Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if Python is installed
check_python() {
    log_info "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.9 or higher."
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    log_success "Python $python_version found"
}

# Check if pip is installed
check_pip() {
    log_info "Checking pip installation..."
    if ! command -v pip3 &> /dev/null; then
        log_error "pip is not installed. Please install pip."
        exit 1
    fi
    log_success "pip found"
}

# Check if Docker is installed
check_docker() {
    log_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed. You'll need Docker to run Qdrant locally."
        return 1
    fi
    log_success "Docker found"
    return 0
}

# Create virtual environment
setup_venv() {
    log_info "Setting up virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    log_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install development dependencies
    pip install -r requirements-dev.txt
    
    log_success "Dependencies installed"
}

# Setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    pre-commit install
    pre-commit install --hook-type pre-push
    
    log_success "Pre-commit hooks installed"
}

# Run initial code quality checks
run_initial_checks() {
    log_info "Running initial code quality checks..."
    
    # Format code
    log_info "Formatting code..."
    black cbt_journal/ tools/ tests/ || log_warning "Black formatting had issues"
    isort cbt_journal/ tools/ tests/ || log_warning "isort had issues"
    
    # Run linting (but don't fail on errors during setup)
    log_info "Running linting..."
    flake8 cbt_journal/ tools/ tests/ || log_warning "Flake8 found issues (normal during initial setup)"
    
    # Run type checking
    log_info "Running type checking..."
    mypy cbt_journal/ tools/ || log_warning "mypy found issues (normal during initial setup)"
    
    log_success "Initial checks completed"
}

# Setup Docker environment
setup_docker() {
    if check_docker; then
        log_info "Setting up Docker environment..."
        
        # Check if docker-compose is available
        if command -v docker-compose &> /dev/null; then
            docker_compose_cmd="docker-compose"
        elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
            docker_compose_cmd="docker compose"
        else
            log_warning "Docker Compose not found. Please install Docker Compose."
            return 1
        fi
        
        # Start Qdrant
        log_info "Starting Qdrant with Docker..."
        cd docker
        $docker_compose_cmd up -d
        cd ..
        
        log_success "Qdrant started"
        
        # Wait for Qdrant to be ready
        log_info "Waiting for Qdrant to be ready..."
        sleep 10
        
        # Test Qdrant connection
        if curl -s http://localhost:6333/health > /dev/null; then
            log_success "Qdrant is ready"
        else
            log_warning "Qdrant may not be ready yet. Check with 'make docker-logs'"
        fi
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p data
    mkdir -p logs
    mkdir -p tmp
    
    log_success "Directories created"
}

# Main setup function
main() {
    echo
    log_info "Starting development environment setup..."
    
    # Check prerequisites
    check_python
    check_pip
    
    # Setup Python environment
    setup_venv
    install_dependencies
    
    # Setup code quality tools
    setup_precommit
    
    # Create directories
    create_directories
    
    # Run initial checks
    run_initial_checks
    
    # Setup Docker (optional)
    setup_docker
    
    echo
    log_success "Development environment setup completed!"
    echo
    echo "📋 Next steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Run tests: make test"
    echo "3. Run all checks: make all-checks"
    echo "4. Start coding! 🎉"
    echo
    echo "📖 Available commands:"
    echo "  make help         - Show all available commands"
    echo "  make format       - Format code"
    echo "  make lint         - Run linting"
    echo "  make test         - Run tests"
    echo "  make all-checks   - Run all quality checks"
    echo
}

# Run main function
main "$@"