#!/bin/bash

# CutFEMx Documentation Build Script
# This script helps build and serve the documentation locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the docs directory
if [ ! -f "conf.py" ]; then
    print_error "This script must be run from the docs directory"
    exit 1
fi

# Initialize conda for the current shell session
initialize_conda() {
    if [ -z "$CONDA_EXE" ]; then
        # Try to find conda
        if command -v conda &> /dev/null; then
            CONDA_EXE=$(which conda)
        elif [ -f "$HOME/miniconda3/bin/conda" ]; then
            CONDA_EXE="$HOME/miniconda3/bin/conda"
        elif [ -f "$HOME/anaconda3/bin/conda" ]; then
            CONDA_EXE="$HOME/anaconda3/bin/conda"
        else
            print_error "Conda not found. Please install conda or make sure it's in your PATH"
            exit 1
        fi
    fi
    
    # Initialize conda for this shell
    eval "$($CONDA_EXE shell.bash hook)"
}

# Function to activate conda environment
activate_env() {
    print_status "Setting up conda environment: fenicsx-dev"
    
    # Initialize conda
    initialize_conda
    
    # Check if environment exists
    if ! conda env list | grep -q "fenicsx-dev"; then
        print_error "Conda environment 'fenicsx-dev' not found"
        print_status "Available environments:"
        conda env list
        exit 1
    fi
    
    # Activate environment
    conda activate fenicsx-dev
    print_success "Environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"
}

# Function to install dependencies
install_deps() {
    activate_env
    print_status "Installing documentation dependencies..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Function to clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    rm -rf _build/
    rm -rf auto_examples/
    rm -rf generated/
    print_success "Build directory cleaned"
}

# Function to build documentation
build_docs() {
    activate_env
    print_status "Building documentation..."
    sphinx-build -b html . _build/html
    print_success "Documentation built successfully"
}

# Function to serve documentation locally
serve_docs() {
    local port=${1:-8000}
    
    # Check if documentation is built
    if [ ! -d "_build/html" ] || [ ! -f "_build/html/index.html" ]; then
        print_error "Documentation not found. Please build it first with: $0 build"
        exit 1
    fi
    
    print_status "Starting local documentation server..."
    print_status "Documentation will be available at: http://localhost:$port"
    print_warning "Press Ctrl+C to stop the server"
    
    # Check if Python is available (should be after conda activation)
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Make sure conda environment is activated."
        exit 1
    fi
    
    # Start the server in the build directory
    cd _build/html
    print_status "Serving from: $(pwd)"
    python -m http.server $port
}

# Function to build and serve with live reload
live_docs() {
    print_status "Starting live documentation server with auto-reload..."
    print_status "Documentation will be available at: http://localhost:8000"
    print_warning "Press Ctrl+C to stop the server"
    
    # Check if sphinx-autobuild is available
    if ! command -v sphinx-autobuild &> /dev/null; then
        print_error "sphinx-autobuild not found. Installing..."
        pip install sphinx-autobuild
    fi
    
    sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000
}

# Function to check for broken links
check_links() {
    activate_env
    print_status "Checking for broken links..."
    sphinx-build -b linkcheck . _build/linkcheck
    print_success "Link check completed"
}

# Function to build PDF documentation
build_pdf() {
    activate_env
    print_status "Building PDF documentation..."
    sphinx-build -b latex . _build/latex
    cd _build/latex
    make
    cd ../..
    print_success "PDF documentation built successfully"
}

# Function to show help
show_help() {
    echo "CutFEMx Documentation Build Script"
    echo
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  install     Install documentation dependencies (activates fenicsx-dev env)"
    echo "  clean       Clean build directory"
    echo "  build       Build documentation (activates fenicsx-dev env)"
    echo "  serve       Serve documentation locally (after building)"
    echo "  live        Build and serve with live reload (activates fenicsx-dev env)"
    echo "  check       Check for broken links (activates fenicsx-dev env)"
    echo "  pdf         Build PDF documentation (activates fenicsx-dev env)"
    echo "  all         Clean, build, and serve documentation (activates fenicsx-dev env)"
    echo "  help        Show this help message"
    echo
    echo "Prerequisites:"
    echo "  - Conda environment 'fenicsx-dev' must exist"
    echo "  - Run from the docs directory"
    echo
    echo "Examples:"
    echo "  $0 install     # Install dependencies in fenicsx-dev environment"
    echo "  $0 build       # Build documentation"
    echo "  $0 live        # Start live development server"
    echo "  $0 all         # Full build and serve"
}

# Main script logic
case "${1:-help}" in
    install)
        install_deps
        ;;
    clean)
        clean_build
        ;;
    build)
        build_docs
        ;;
    serve)
        # Parse port parameter if provided
        port=8000
        if [ "$2" = "--port" ] && [ -n "$3" ]; then
            port=$3
        fi
        
        if [ ! -d "_build/html" ]; then
            print_warning "Documentation not built yet. Building first..."
            activate_env
            build_docs
            serve_docs $port
        else
            activate_env
            serve_docs $port
        fi
        ;;
    live)
        activate_env
        live_docs
        ;;
    check)
        activate_env
        check_links
        ;;
    pdf)
        build_pdf
        ;;
    all)
        # Parse port parameter if provided
        port=8000
        if [ "$2" = "--port" ] && [ -n "$3" ]; then
            port=$3
        fi
        
        activate_env
        clean_build
        build_docs
        serve_docs $port
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac
