#!/bin/bash

# DICE-Talk API Server Startup Script for Ubuntu
# Author: DICE-Talk Team
# Description: One-click startup script for DICE-Talk API server

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   error "This script should not be run as root for security reasons"
   exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log "Starting DICE-Talk API Server Setup..."
log "Working directory: $SCRIPT_DIR"

# Check Ubuntu version
log "Checking Ubuntu version..."
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "$ID" != "ubuntu" ]]; then
        warning "This script is designed for Ubuntu. Current OS: $ID"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    log "OS: $ID $VERSION_ID"
else
    warning "Cannot detect OS version"
fi

# Check Python version
log "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
    log "Python version: $PYTHON_VERSION"
    
    # Check if Python version is >= 3.8
    if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        success "Python version is compatible"
    else
        error "Python 3.8+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    error "Python3 is not installed"
    log "Please install Python3: sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

# Check pip
log "Checking pip installation..."
if command -v pip3 &> /dev/null; then
    success "pip3 is available"
else
    error "pip3 is not installed"
    log "Installing pip3..."
    sudo apt update && sudo apt install python3-pip
fi

# Check CUDA/GPU
log "Checking CUDA/GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    success "GPU detected: $GPU_INFO"
else
    warning "NVIDIA GPU not detected or nvidia-smi not available"
    warning "The server will run on CPU (much slower)"
fi

## Check if virtual environment exists
#log "Checking Python virtual environment..."
#if [[ -d "venv" ]]; then
#    log "Virtual environment found"
#else
#    log "Creating virtual environment..."
#    python3 -m venv venv
#    success "Virtual environment created"
#fi
#
## Activate virtual environment
#log "Activating virtual environment..."
#source venv/bin/activate

## Upgrade pip
#log "Upgrading pip..."
#pip install --upgrade pip
#
## Install requirements
#log "Installing Python dependencies..."
#if [[ -f "requirements.txt" ]]; then
#    log "Installing main requirements..."
#    pip install -r requirements.txt
#else
#    error "requirements.txt not found"
#    exit 1
#fi
#
#if [[ -f "requirements_api.txt" ]]; then
#    log "Installing API requirements..."
#    pip install -r requirements_api.txt
#else
#    error "requirements_api.txt not found"
#    exit 1
#fi
#
#success "Dependencies installed successfully"

# Create necessary directories
log "Creating necessary directories..."
mkdir -p uploads outputs logs
success "Directories created"

# Check model files
log "Checking model files..."
if [[ ! -d "checkpoints" ]] && [[ ! -d "models" ]]; then
    warning "Model files not found in 'checkpoints' or 'models' directory"
    warning "Please ensure you have downloaded the DICE-Talk model files"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Health check function
health_check() {
    local retries=0
    local max_retries=30
    
    log "Performing health check..."
    while [[ $retries -lt $max_retries ]]; do
        if curl -s -f http://localhost:8000/api/health > /dev/null 2>&1; then
            success "Health check passed"
            return 0
        fi
        ((retries++))
        log "Health check attempt $retries/$max_retries..."
        sleep 5
    done
    
    error "Health check failed after $max_retries attempts"
    return 1
}

# Function to start server
start_server() {
    log "Starting DICE-Talk API server..."
    log "Server will be available at: http://localhost:8000"
    log "API documentation will be available at: http://localhost:8000/docs"
    
    # Start server in background for health check
    python -m api.server &
    SERVER_PID=$!
    
    # Wait longer for DICE-Talk model to load
    log "Waiting for DICE-Talk model to load (this may take a few minutes)..."
    sleep 15
    
    # Perform health check
    if health_check; then
        success "DICE-Talk API server started successfully!"
        success "Server PID: $SERVER_PID"
        
        log "Access points:"
        log "  - API Base URL: http://localhost:8000"
        log "  - Health Check: http://localhost:8000/api/health"
        log "  - API Documentation: http://localhost:8000/docs"
        log "  - ReDoc Documentation: http://localhost:8000/redoc"
        
        log "To stop the server, run: kill $SERVER_PID"
        
        # Keep server running in foreground
        wait $SERVER_PID
    else
        error "Server failed to start properly"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
}

# Function to handle cleanup on exit
cleanup() {
    log "Shutting down server..."
    if [[ -n "$SERVER_PID" ]]; then
        kill $SERVER_PID 2>/dev/null || true
    fi
    deactivate 2>/dev/null || true
    log "Cleanup completed"
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Check if port 8000 is already in use
if lsof -i :8000 >/dev/null 2>&1; then
    error "Port 8000 is already in use"
    error "Please stop the service using port 8000 or change the port in the configuration"
    exit 1
fi

# Parse command line arguments
case "${1:-start}" in
    "start")
        start_server
        ;;
    "install")
        success "Installation completed successfully!"
        log "Run './start_server.sh start' to start the server"
        ;;
    "health")
        health_check
        ;;
    "help")
        echo "Usage: $0 [start|install|health|help]"
        echo "  start   - Install dependencies and start server (default)"
        echo "  install - Only install dependencies"
        echo "  health  - Check if server is running"
        echo "  help    - Show this help message"
        ;;
    *)
        error "Unknown command: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac 