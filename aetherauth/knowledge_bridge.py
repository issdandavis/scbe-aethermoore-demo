"""
Knowledge Bridge - Notion <-> Perplexity API Bridge

Example implementation using AetherAuth for secure API access.

Flow:
    1. Capture context vector (6D behavioral state)
    2. Validate against trust rings (GeoSeal gate)
    3. Decrypt API keys from vault (context-bound)
    4. Query APIs with decrypted credentials
    5. Record success for historical tracking

Usage:
    # Set master key
    export AETHER_MASTER_KEY="your-secure-passphrase"

    # Store API keys (one-time setup)
    python -m aetherauth.vault_setup

    # Run bridge
    python -m aetherauth.knowledge_bridge --database-id YOUR_DB_ID
"""

import time
import json
import argparse
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# AetherAuth imports
try:
    from .context_capture import capture_context_vector, record_success, ContextVector
    from .geoseal_gate import AetherAuthGate, AccessResult, TrustRing
    from .vault_access import LumoVault
except ImportError:
    from context_capture import capture_context_vector, record_success, ContextVector
    from geoseal_gate import AetherAuthGate, AccessResult, TrustRing
    from vault_access import LumoVault

# Optional API client imports
try:
    from notion_client import Client as NotionClient
    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class BridgeConfig:
    """Configuration for the knowledge bridge."""
    notion_database_id: str
    poll_interval: int = 300  # 5 minutes
    dry_run: bool = False
    verbose: bool = False


@dataclass
class AuditEntry:
    """Audit log entry."""
    timestamp: str
    event: str
    ring: str
    distance: float
    success: bool
    details: Dict[str, Any]


class AuditLogger:
    """Simple audit logger for security events."""

    def __init__(self, log_path: str = ".aether/logs/audit.log"):
        self.log_path = log_path

        # Ensure log directory exists
        import os
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, event: str, access: AccessResult, success: bool, details: Dict = None):
        """Log an audit event."""
        entry = AuditEntry(
            timestamp=datetime.utcnow().isoformat(),
            event=event,
            ring=access.ring.value,
            distance=access.distance,
            success=success,
            details=details or {}
        )

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry.__dict__) + '\n')

        # Alert on security events
        if access.ring in [TrustRing.WALL, TrustRing.EVENT_HORIZON]:
            self._send_alert(entry)

    def _send_alert(self, entry: AuditEntry):
        """Send security alert (placeholder for integration)."""
        print(f"SECURITY ALERT: {entry.event} blocked at {entry.ring}")
        # TODO: Integrate with Slack, PagerDuty, etc.


class KnowledgeBridge:
    """
    Notion <-> Perplexity bridge with AetherAuth security.

    Monitors a Notion database for questions, queries Perplexity
    for answers, and writes results back to Notion.
    """

    def __init__(self, config: BridgeConfig):
        self.config = config
        self.gate = AetherAuthGate()
        self.vault = LumoVault()
        self.audit = AuditLogger()

        # API clients (initialized after auth)
        self.notion = None
        self.pplx_key = None

        # Auth state
        self.access: Optional[AccessResult] = None

    def authenticate(self) -> AccessResult:
        """
        Perform AetherAuth handshake.

        1. Capture context vector
        2. Validate against trust rings
        3. Decrypt API keys from vault
        """
        print("AetherAuth: Capturing context...")

        # Capture current context
        context = capture_context_vector(caller_name="authenticate")

        if self.config.verbose:
            print(f"  Context: {context}")

        # Check trust ring
        access = self.gate.check_access(context)
        print(f"  Trust Ring: {access.ring.value} (distance: {access.distance:.3f})")

        if not access.allowed:
            self.audit.log("AUTH_DENIED", access, False, {"reason": access.reason})
            raise PermissionError(f"Access Denied: {access.reason}")

        # Apply latency penalty
        access = self.gate.enforce_latency(access)

        # Decrypt API keys
        print("Accessing Lumo Vault...")

        notion_key = self.vault.get_key("notion", context, access)
        pplx_key = self.vault.get_key("perplexity", context, access)

        if not notion_key or not pplx_key:
            self.audit.log("VAULT_DECRYPT_FAILED", access, False)
            raise ValueError("Failed to decrypt API keys (context mismatch)")

        # Initialize API clients
        if NOTION_AVAILABLE and not self.config.dry_run:
            self.notion = NotionClient(auth=notion_key)
        else:
            self.notion = None

        self.pplx_key = pplx_key
        self.access = access

        # Log success
        self.audit.log("AUTH_SUCCESS", access, True, {
            "access_level": access.access_level
        })

        print(f"Authenticated (Access Level: {access.access_level})")
        record_success()

        return access

    def query_notion(self) -> List[Dict]:
        """Query Notion for pending questions."""
        if self.config.dry_run or not self.notion:
            print("[DRY RUN] Would query Notion database")
            return [
                {"id": "mock-1", "question": "What is quantum computing?"},
                {"id": "mock-2", "question": "How does SCBE-AETHERMOORE work?"},
            ]

        results = self.notion.databases.query(
            database_id=self.config.notion_database_id,
            filter={
                "property": "Status",
                "select": {"equals": "Pending"}
            }
        )

        questions = []
        for page in results.get('results', []):
            try:
                question = page['properties']['Question']['title'][0]['plain_text']
                questions.append({
                    "id": page['id'],
                    "question": question
                })
            except (KeyError, IndexError):
                continue

        return questions

    def query_perplexity(self, question: str) -> str:
        """Query Perplexity API for an answer."""
        if self.config.dry_run or not REQUESTS_AVAILABLE:
            print(f"[DRY RUN] Would query Perplexity: {question[:50]}...")
            return f"[Mock Answer] This is a simulated response to: {question}"

        url = "https://api.perplexity.ai/chat/completions"

        payload = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [
                {"role": "user", "content": question}
            ]
        }

        headers = {
            "Authorization": f"Bearer {self.pplx_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            record_success()
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code} - {response.text}"

    def write_answer(self, page_id: str, answer: str):
        """Write answer back to Notion."""
        if self.config.dry_run or not self.notion:
            print(f"[DRY RUN] Would write answer to {page_id}")
            return

        # Append answer as paragraph block
        self.notion.blocks.children.append(
            block_id=page_id,
            children=[
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": answer}}]
                    }
                }
            ]
        )

        # Update status
        self.notion.pages.update(
            page_id=page_id,
            properties={
                "Status": {"select": {"name": "Answered"}}
            }
        )

        record_success()

    def run_once(self):
        """Run one iteration of the bridge."""
        # Re-authenticate (context may have changed)
        self.authenticate()

        # Query for pending questions
        print(f"\nQuerying Notion database: {self.config.notion_database_id}")
        questions = self.query_notion()

        if not questions:
            print("No pending questions found")
            return

        print(f"Found {len(questions)} pending questions")

        # Process each question
        for item in questions:
            question = item['question']
            page_id = item['id']

            print(f"\nQuestion: {question[:60]}...")

            # Query Perplexity
            answer = self.query_perplexity(question)
            print(f"Answer: {answer[:100]}...")

            # Write back to Notion
            self.write_answer(page_id, answer)
            print("Answer written to Notion")

            self.audit.log("QUESTION_ANSWERED", self.access, True, {
                "question_id": page_id
            })

    def run_loop(self):
        """Run the bridge in a continuous loop."""
        print(f"Starting Knowledge Bridge (poll interval: {self.config.poll_interval}s)")
        print("Press Ctrl+C to stop")

        while True:
            try:
                self.run_once()
            except PermissionError as e:
                print(f"Auth failed: {e}")
                print("Waiting 60s before retry...")
                time.sleep(60)
                continue
            except Exception as e:
                print(f"Error: {e}")
                self.audit.log("ERROR", self.access or AccessResult(
                    allowed=False, ring=TrustRing.WALL, distance=1.0
                ), False, {"error": str(e)})

            print(f"\nSleeping {self.config.poll_interval}s...")
            time.sleep(self.config.poll_interval)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Notion-Perplexity Knowledge Bridge")
    parser.add_argument("--database-id", required=True, help="Notion database ID")
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds between polls")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without API calls")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config = BridgeConfig(
        notion_database_id=args.database_id,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    bridge = KnowledgeBridge(config)

    if args.once:
        bridge.run_once()
    else:
        bridge.run_loop()


if __name__ == "__main__":
    main()
