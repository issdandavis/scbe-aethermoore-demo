#!/bin/bash
#
# SCBE-AETHERMOORE End-to-End Demo
# Demonstrates the complete governance workflow
#
# This script:
# 1. Starts the API server
# 2. Registers agents
# 3. Runs governance scenarios
# 4. Shows audit trail
# 5. Stops the server
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[SCBE]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
API_PORT=8000
API_KEY="demo-e2e-$(date +%s)"
API_URL="http://localhost:$API_PORT"

# Cleanup on exit
cleanup() {
    if [ ! -z "$API_PID" ]; then
        log "Stopping API server..."
        kill $API_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Check if port is available
check_port() {
    if lsof -i:$API_PORT &>/dev/null; then
        error "Port $API_PORT is already in use"
        exit 1
    fi
}

# Start API server in background
start_api() {
    log "Starting SCBE API server on port $API_PORT..."

    export SCBE_API_KEY="$API_KEY"

    python3 -m uvicorn api.main:app --host 0.0.0.0 --port $API_PORT &>/dev/null &
    API_PID=$!

    # Wait for server to be ready
    for i in {1..30}; do
        if curl -s "$API_URL/health" &>/dev/null; then
            success "API server started (PID: $API_PID)"
            return 0
        fi
        sleep 0.5
    done

    error "API server failed to start"
    exit 1
}

# Make API request with pretty output
api_request() {
    local method=$1
    local endpoint=$2
    local data=$3

    if [ "$method" == "GET" ]; then
        curl -s -X GET "$API_URL$endpoint" \
            -H "X-API-Key: $API_KEY" \
            -H "Content-Type: application/json"
    else
        curl -s -X POST "$API_URL$endpoint" \
            -H "X-API-Key: $API_KEY" \
            -H "Content-Type: application/json" \
            -d "$data"
    fi
}

# Demo scenarios
run_demos() {
    echo ""
    echo "=================================================="
    echo "  SCBE-AETHERMOORE End-to-End Demo"
    echo "  API Key: $API_KEY"
    echo "=================================================="

    # 1. Health check
    echo ""
    log "1. Health Check"
    echo "   GET /health"
    HEALTH=$(api_request GET /health)
    echo "   Response: $HEALTH"
    success "API is healthy"

    # 2. Register agents
    echo ""
    log "2. Registering Agents"

    echo "   POST /v1/agents (CodeGen-GPT4)"
    AGENT1=$(api_request POST /v1/agents '{
        "agent_id": "codegen-001",
        "name": "CodeGen-GPT4",
        "provider": "openai",
        "model": "gpt-4o",
        "trust_score": 0.75,
        "capabilities": ["code_generation", "refactoring"]
    }')
    echo "   -> $(echo $AGENT1 | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Registered: {d.get(\"agent_id\", \"error\")}")')"

    echo "   POST /v1/agents (Security-Claude)"
    AGENT2=$(api_request POST /v1/agents '{
        "agent_id": "security-001",
        "name": "Security-Claude",
        "provider": "anthropic",
        "model": "claude-3",
        "trust_score": 0.90,
        "capabilities": ["security_scan", "code_review"]
    }')
    echo "   -> $(echo $AGENT2 | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"Registered: {d.get(\"agent_id\", \"error\")}")')"

    success "2 agents registered"

    # 3. Authorization scenarios
    echo ""
    log "3. Authorization Scenarios"

    # Low-risk action
    echo ""
    echo "   Scenario A: Low-risk READ action"
    echo "   POST /v1/authorize"
    AUTH1=$(api_request POST /v1/authorize '{
        "agent_id": "codegen-001",
        "action": "READ",
        "target": "src/utils.py",
        "context": {"sensitivity": 0.1}
    }')
    echo "   Response:"
    echo "$AUTH1" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"     Decision: {d.get(\"decision\", \"?\")}"); print(f"     Risk Score: {d.get(\"risk_score\", \"?\")}")'

    # Medium-risk action
    echo ""
    echo "   Scenario B: Medium-risk WRITE action"
    AUTH2=$(api_request POST /v1/authorize '{
        "agent_id": "codegen-001",
        "action": "WRITE",
        "target": "src/new_feature.py",
        "context": {"sensitivity": 0.5}
    }')
    echo "   Response:"
    echo "$AUTH2" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"     Decision: {d.get(\"decision\", \"?\")}"); print(f"     Risk Score: {d.get(\"risk_score\", \"?\")}")'

    # High-risk action
    echo ""
    echo "   Scenario C: High-risk DEPLOY action"
    AUTH3=$(api_request POST /v1/authorize '{
        "agent_id": "codegen-001",
        "action": "DEPLOY",
        "target": "production",
        "context": {"sensitivity": 0.9}
    }')
    echo "   Response:"
    echo "$AUTH3" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"     Decision: {d.get(\"decision\", \"?\")}"); print(f"     Risk Score: {d.get(\"risk_score\", \"?\")}")'

    # 4. Fleet scenario
    echo ""
    log "4. Fleet Scenario (Multi-Agent)"
    echo "   POST /v1/fleet/run-scenario"

    FLEET=$(api_request POST /v1/fleet/run-scenario '{
        "scenario_name": "e2e-demo-scenario",
        "agents": [
            {"agent_id": "codegen-001", "role": "developer", "trust_score": 0.75},
            {"agent_id": "security-001", "role": "security", "trust_score": 0.90}
        ],
        "actions": [
            {"agent_id": "codegen-001", "action": "WRITE", "target": "src/main.py", "sensitivity": 0.4},
            {"agent_id": "security-001", "action": "READ", "target": "audit.log", "sensitivity": 0.2}
        ],
        "consensus_threshold": 0.67
    }')

    echo "   Response:"
    echo "$FLEET" | python3 -c '
import sys,json
d=json.load(sys.stdin)
print(f"     Scenario ID: {d.get(\"scenario_id\", \"?\")}")
print(f"     Allowed: {d.get(\"allowed\", \"?\")} | Quarantined: {d.get(\"quarantined\", \"?\")} | Denied: {d.get(\"denied\", \"?\")}")
print(f"     Allow Rate: {d.get(\"allow_rate\", 0)*100:.0f}%")
print(f"     Execution Time: {d.get(\"execution_time_ms\", \"?\")}ms")
'

    success "Fleet scenario executed"

    # 5. Metrics
    echo ""
    log "5. System Metrics"
    echo "   GET /v1/metrics"

    METRICS=$(api_request GET /v1/metrics)
    echo "   Response:"
    echo "$METRICS" | python3 -c '
import sys,json
d=json.load(sys.stdin)
print(f"     Total Decisions: {d.get(\"total_decisions\", 0)}")
print(f"     Allow Rate: {d.get(\"allow_rate\", 0)*100:.1f}%")
print(f"     Avg Latency: {d.get(\"avg_latency_ms\", 0):.2f}ms")
'

    success "Metrics retrieved"

    # Summary
    echo ""
    echo "=================================================="
    echo "  Demo Complete!"
    echo "=================================================="
    echo ""
    echo "  What happened:"
    echo "  1. Started SCBE API server"
    echo "  2. Registered 2 AI agents (CodeGen + Security)"
    echo "  3. Tested authorization at 3 risk levels"
    echo "  4. Ran a multi-agent fleet scenario"
    echo "  5. Retrieved governance metrics"
    echo ""
    echo "  The 14-layer hyperbolic geometry pipeline processed"
    echo "  each request and returned deterministic decisions."
    echo ""
    echo "  Try the Swagger UI: $API_URL/docs"
    echo ""
}

# Main
main() {
    check_port
    start_api
    sleep 1
    run_demos

    echo "Press Ctrl+C to stop the server, or explore the API..."
    echo ""

    # Keep running until interrupted
    wait $API_PID
}

main
