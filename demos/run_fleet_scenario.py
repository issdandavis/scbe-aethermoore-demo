#!/usr/bin/env python3
"""
Fleet Scenario Demo
===================

Runs a complete fleet governance scenario through the SCBE 14-layer pipeline.
This demonstrates the "animal eats food" workflow - the whole system working end-to-end.

Usage:
    # Start the API first:
    SCBE_API_KEY=your-key uvicorn api.main:app --host 0.0.0.0 --port 8080

    # Then run this demo:
    python demos/run_fleet_scenario.py

Or use the one-liner:
    python demos/run_fleet_scenario.py --start-server
"""

import argparse
import json
import os
import subprocess
import sys
import time
from urllib.request import Request, urlopen
from urllib.error import URLError

API_URL = os.environ.get("SCBE_API_URL", "http://localhost:8080")
API_KEY = os.environ.get("SCBE_API_KEY", "test-key-12345")

# =============================================================================
# Sample Fleet Scenarios
# =============================================================================

SCENARIOS = {
    "security-audit": {
        "scenario_name": "security-audit-demo",
        "agents": [
            {"agent_id": "analyst-001", "name": "Data Analyst", "role": "analyst", "trust": 0.8},
            {"agent_id": "intern-001", "name": "New Intern", "role": "intern", "trust": 0.3},
            {"agent_id": "admin-001", "name": "System Admin", "role": "admin", "trust": 0.95}
        ],
        "actions": [
            {"agent_id": "analyst-001", "action": "READ", "target": "user_analytics", "sensitivity": "medium"},
            {"agent_id": "intern-001", "action": "WRITE", "target": "production_db", "sensitivity": "high"},
            {"agent_id": "admin-001", "action": "DELETE", "target": "audit_logs", "sensitivity": "critical"},
            {"agent_id": "analyst-001", "action": "EXECUTE", "target": "data_pipeline", "sensitivity": "low"},
            {"agent_id": "intern-001", "action": "READ", "target": "public_docs", "sensitivity": "low"}
        ]
    },
    "fraud-detection": {
        "scenario_name": "fraud-detection-team",
        "agents": [
            {"agent_id": "fraud-detector-001", "name": "ML Fraud Model", "role": "detector", "trust": 0.85},
            {"agent_id": "fraud-detector-002", "name": "Rule Engine", "role": "detector", "trust": 0.9},
            {"agent_id": "human-reviewer-001", "name": "Human Analyst", "role": "reviewer", "trust": 0.95}
        ],
        "actions": [
            {"agent_id": "fraud-detector-001", "action": "READ", "target": "transaction_stream", "sensitivity": "high"},
            {"agent_id": "fraud-detector-002", "action": "READ", "target": "transaction_stream", "sensitivity": "high"},
            {"agent_id": "fraud-detector-001", "action": "WRITE", "target": "fraud_alerts", "sensitivity": "medium"},
            {"agent_id": "human-reviewer-001", "action": "READ", "target": "customer_pii", "sensitivity": "critical"},
            {"agent_id": "human-reviewer-001", "action": "EXECUTE", "target": "block_account", "sensitivity": "critical"}
        ]
    },
    "data-pipeline": {
        "scenario_name": "etl-pipeline-agents",
        "agents": [
            {"agent_id": "extractor-001", "name": "Data Extractor", "role": "etl", "trust": 0.7},
            {"agent_id": "transformer-001", "name": "Data Transformer", "role": "etl", "trust": 0.75},
            {"agent_id": "loader-001", "name": "Data Loader", "role": "etl", "trust": 0.8},
            {"agent_id": "validator-001", "name": "Quality Checker", "role": "qa", "trust": 0.85}
        ],
        "actions": [
            {"agent_id": "extractor-001", "action": "READ", "target": "source_db", "sensitivity": "medium"},
            {"agent_id": "transformer-001", "action": "EXECUTE", "target": "transform_job", "sensitivity": "low"},
            {"agent_id": "loader-001", "action": "WRITE", "target": "data_warehouse", "sensitivity": "high"},
            {"agent_id": "validator-001", "action": "READ", "target": "data_warehouse", "sensitivity": "medium"},
            {"agent_id": "validator-001", "action": "WRITE", "target": "quality_reports", "sensitivity": "low"}
        ]
    }
}

# =============================================================================
# API Client
# =============================================================================

def api_request(endpoint: str, method: str = "GET", data: dict = None, base_url: str = None) -> dict:
    """Make a request to the SCBE API"""
    url = f"{base_url or API_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    body = json.dumps(data).encode() if data else None
    req = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except URLError as e:
        print(f"Error: {e}")
        return None


def check_health(base_url: str = None) -> bool:
    """Check if the API is healthy"""
    result = api_request("/v1/health", base_url=base_url)
    return result and result.get("status") == "healthy"


def run_scenario(scenario_name: str, base_url: str = None) -> dict:
    """Run a fleet scenario"""
    if scenario_name not in SCENARIOS:
        print(f"Unknown scenario: {scenario_name}")
        print(f"Available: {', '.join(SCENARIOS.keys())}")
        return None

    scenario = SCENARIOS[scenario_name]
    return api_request("/v1/fleet/run-scenario", "POST", scenario, base_url=base_url)


# =============================================================================
# Display Functions
# =============================================================================

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_result(result: dict):
    """Pretty print the scenario result"""
    if not result:
        print("No result received")
        return

    print_header(f"Scenario: {result['scenario_name']}")

    print(f"\nAgents Registered: {result['agents_registered']}")
    print(f"Actions Processed: {result['actions_processed']}")

    print("\n--- Decision Summary ---")
    for decision, count in result['summary'].items():
        bar = "█" * count
        print(f"  {decision:12} {count:3} {bar}")

    print("\n--- Individual Decisions ---")
    for d in result['decisions']:
        emoji = {"ALLOW": "✓", "DENY": "✗", "QUARANTINE": "⚠"}[d['decision']]
        print(f"  {emoji} {d['agent_id']:20} {d['action']:10} → {d['decision']:12} (score: {d['score']:.3f})")
        print(f"      {d['reason']}")

    print("\n--- Metrics ---")
    metrics = result['metrics']
    print(f"  Latency:    {metrics['latency_ms']:.2f} ms")
    print(f"  Avg Score:  {metrics['avg_score']:.3f}")
    print(f"  Allow Rate: {metrics['allow_rate']:.1f}%")
    print(f"  Deny Rate:  {metrics['deny_rate']:.1f}%")
    print(f"  Quarantine: {metrics['quarantine_rate']:.1f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run SCBE Fleet Scenario Demo")
    parser.add_argument("--scenario", "-s", default="security-audit",
                        choices=list(SCENARIOS.keys()),
                        help="Scenario to run")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Run all scenarios")
    parser.add_argument("--start-server", action="store_true",
                        help="Start API server before running")
    parser.add_argument("--api-url", default=API_URL,
                        help="API base URL")
    args = parser.parse_args()

    # Use command line API URL if provided
    api_url = args.api_url

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║           SCBE FLEET SCENARIO DEMO                       ║")
    print("║      AI Agent Governance 14-Layer Pipeline               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Start server if requested
    if args.start_server:
        print("\nStarting API server...")
        subprocess.Popen(
            ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env={**os.environ, "SCBE_API_KEY": API_KEY}
        )
        time.sleep(3)

    # Check health
    print(f"\nChecking API at {api_url}...")
    if not check_health(api_url):
        print("✗ API not available. Start with:")
        print("  SCBE_API_KEY=test-key-12345 uvicorn api.main:app --host 0.0.0.0 --port 8080")
        sys.exit(1)
    print("✓ API is healthy")

    # Run scenarios
    if args.all:
        for name in SCENARIOS:
            result = run_scenario(name, api_url)
            print_result(result)
    else:
        result = run_scenario(args.scenario, api_url)
        print_result(result)

    print("\n" + "=" * 60)
    print("✓ Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
