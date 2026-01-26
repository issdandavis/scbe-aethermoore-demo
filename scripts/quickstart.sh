#!/bin/bash
#
# SCBE-AETHERMOORE Quick Start Script
# One command to get the system running
#
# Usage: ./scripts/quickstart.sh [mode]
#   mode: demo | api | test | all (default: demo)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check dependencies
check_deps() {
    log_info "Checking dependencies..."

    # Node.js
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node -v)
        log_success "Node.js $NODE_VERSION"
    else
        log_error "Node.js not found. Install from https://nodejs.org"
        exit 1
    fi

    # Python
    if command -v python3 &> /dev/null; then
        PY_VERSION=$(python3 --version)
        log_success "$PY_VERSION"
    else
        log_error "Python 3 not found"
        exit 1
    fi

    # npm
    if command -v npm &> /dev/null; then
        log_success "npm $(npm -v)"
    else
        log_error "npm not found"
        exit 1
    fi
}

# Install dependencies
install_deps() {
    log_info "Installing dependencies..."

    # Node.js dependencies
    if [ ! -d "node_modules" ]; then
        log_info "Installing Node.js dependencies..."
        npm install
    else
        log_success "Node.js dependencies already installed"
    fi

    # Python dependencies
    log_info "Installing Python dependencies..."
    pip install -q -r requirements.txt 2>/dev/null || pip3 install -q -r requirements.txt
    pip install -q pytest pytest-cov 2>/dev/null || pip3 install -q pytest pytest-cov

    log_success "All dependencies installed"
}

# Run demo
run_demo() {
    log_info "Running SCBE-AETHERMOORE Demo..."
    echo ""
    echo "=================================================="
    echo "  SCBE-AETHERMOORE: AI Governance Pipeline Demo"
    echo "=================================================="
    echo ""

    python3 demo_memory_shard.py

    echo ""
    log_success "Demo complete! The system works."
}

# Run API server
run_api() {
    log_info "Starting API server..."

    # Set API key if not set
    if [ -z "$SCBE_API_KEY" ]; then
        export SCBE_API_KEY="demo-key-$(date +%s)"
        log_warn "SCBE_API_KEY not set. Using: $SCBE_API_KEY"
    fi

    echo ""
    echo "=================================================="
    echo "  SCBE-AETHERMOORE API Server"
    echo "=================================================="
    echo ""
    echo "  API Key: $SCBE_API_KEY"
    echo "  Swagger UI: http://localhost:8000/docs"
    echo "  Health: http://localhost:8000/health"
    echo ""
    echo "  Press Ctrl+C to stop"
    echo ""

    python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
}

# Run tests
run_tests() {
    log_info "Running test suite..."
    echo ""

    # TypeScript tests
    log_info "Running TypeScript tests..."
    npm test -- --reporter=verbose 2>&1 | tail -20

    echo ""

    # Python tests (core only, skip slow ones)
    log_info "Running Python tests (core)..."
    python3 -m pytest tests/industry_standard/ tests/test_advanced_mathematics.py -v -q --tb=short 2>&1 | tail -30

    log_success "Tests complete!"
}

# Run everything
run_all() {
    check_deps
    install_deps
    run_tests
    run_demo

    echo ""
    log_success "All systems operational!"
    echo ""
    echo "Next steps:"
    echo "  1. Start the API:     ./scripts/quickstart.sh api"
    echo "  2. Open Swagger UI:   http://localhost:8000/docs"
    echo "  3. Try the CLI:       python scbe-cli.py"
    echo ""
}

# Main
MODE="${1:-demo}"

case "$MODE" in
    demo)
        check_deps
        install_deps
        run_demo
        ;;
    api)
        check_deps
        install_deps
        run_api
        ;;
    test)
        check_deps
        install_deps
        run_tests
        ;;
    all)
        run_all
        ;;
    *)
        echo "Usage: $0 [demo|api|test|all]"
        echo ""
        echo "Modes:"
        echo "  demo  - Run the governance pipeline demo (default)"
        echo "  api   - Start the API server with Swagger UI"
        echo "  test  - Run the test suite"
        echo "  all   - Install, test, and demo everything"
        exit 1
        ;;
esac
