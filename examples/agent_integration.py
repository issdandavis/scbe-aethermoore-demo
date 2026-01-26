#!/usr/bin/env python3
"""
SCBE-AETHERMOORE Agent Integration Example
==========================================

This shows how to integrate SCBE governance into your AI agent workflow.

Use Cases:
1. Multi-agent systems (Claude, GPT, etc. working together)
2. Tool-calling APIs (controlling what agents can do)
3. Autonomous workflows (governance gates for actions)

Run with: python examples/agent_integration.py
"""

import sys
import os
import json
import time
import math
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Decision(Enum):
    """Governance decisions"""
    ALLOW = "ALLOW"
    QUARANTINE = "QUARANTINE"
    DENY = "DENY"


@dataclass
class Agent:
    """Represents an AI agent in the fleet"""
    id: str
    name: str
    role: str
    trust_score: float
    capabilities: List[str]

    def to_context_vector(self) -> List[float]:
        """Convert agent properties to 6D context vector for SCBE"""
        # Map agent properties to 6D space:
        # [capability_breadth, trust_level, role_weight, time_factor, sensitivity, audit_level]
        return [
            len(self.capabilities) / 10.0,  # capability breadth (normalized)
            self.trust_score,                # trust level
            0.5 if self.role == "developer" else 0.7 if self.role == "security" else 0.3,  # role weight
            0.5,                             # time factor (neutral)
            0.3,                             # default sensitivity
            0.8,                             # audit level (high)
        ]


@dataclass
class Action:
    """Represents an action an agent wants to take"""
    agent_id: str
    action_type: str  # READ, WRITE, EXECUTE, DEPLOY, DELETE
    target: str
    sensitivity: float = 0.5
    requires_consensus: bool = False


class SCBEGovernanceGate:
    """
    The main integration point for SCBE governance.

    Use this class to wrap your agent actions with governance checks.

    This implements a simplified version of the 14-layer pipeline
    for standalone use. For full pipeline, use the API or TypeScript SDK.
    """

    # Golden ratio for harmonic scaling
    PHI = (1 + math.sqrt(5)) / 2

    # Action risk weights
    ACTION_RISK = {
        "READ": 0.1,
        "WRITE": 0.3,
        "EXECUTE": 0.5,
        "DEPLOY": 0.8,
        "DELETE": 0.95,
    }

    def __init__(self):
        self.audit_log: List[Dict[str, Any]] = []

        # Thresholds (configurable)
        self.allow_threshold = 0.3
        self.quarantine_threshold = 0.7

    def _hyperbolic_distance(self, u: List[float], v: List[float]) -> float:
        """
        Calculate hyperbolic distance in Poincare ball.
        d_H(u,v) = arcosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
        """
        # Clamp to ball (||x|| < 1)
        def clamp(x):
            norm = math.sqrt(sum(xi**2 for xi in x))
            if norm >= 0.999:
                return [xi * 0.999 / norm for xi in x]
            return x

        u = clamp(u)
        v = clamp(v)

        u_norm_sq = sum(xi**2 for xi in u)
        v_norm_sq = sum(xi**2 for xi in v)
        diff_sq = sum((ui - vi)**2 for ui, vi in zip(u, v))

        denom = (1 - u_norm_sq) * (1 - v_norm_sq)
        if denom <= 0:
            denom = 1e-10

        arg = 1 + 2 * diff_sq / denom
        return math.acosh(max(1.0, arg))

    def _harmonic_scaling(self, distance: float, sensitivity: float) -> float:
        """
        Harmonic risk scaling: H(d, s) = phi^d / (1 + e^(-s*10))
        """
        exp_factor = 1 / (1 + math.exp(-sensitivity * 10 + 5))
        return (self.PHI ** distance) * exp_factor

    def _compute_risk(self, agent: Agent, action: Action) -> float:
        """
        Compute risk score using simplified 14-layer pipeline.
        """
        # Get agent context vector
        context = agent.to_context_vector()

        # Ideal trust center (in 6D space)
        ideal = [0.5, 0.8, 0.5, 0.5, 0.3, 0.9]

        # L5: Hyperbolic distance from ideal
        d_hyp = self._hyperbolic_distance(context, ideal)

        # L12: Harmonic scaling with action risk
        action_weight = self.ACTION_RISK.get(action.action_type.upper(), 0.5)
        combined_sensitivity = (action.sensitivity + action_weight) / 2

        risk = self._harmonic_scaling(d_hyp, combined_sensitivity)

        # Normalize to [0, 1]
        risk = min(1.0, max(0.0, risk / 5.0))

        # Adjust by trust score (higher trust = lower risk)
        risk = risk * (1 - agent.trust_score * 0.5)

        return risk

    def authorize(self, agent: Agent, action: Action) -> Dict[str, Any]:
        """
        Main governance check. Call this before any agent action.

        Returns:
            {
                "decision": "ALLOW" | "QUARANTINE" | "DENY",
                "risk_score": float,
                "token": str (if allowed),
                "reason": str (if denied),
                "latency_ms": float
            }
        """
        start_time = time.time()

        # Compute risk using hyperbolic geometry pipeline
        risk_score = self._compute_risk(agent, action)

        # Determine decision based on risk
        if risk_score < self.allow_threshold:
            decision = Decision.ALLOW
        elif risk_score < self.quarantine_threshold:
            decision = Decision.QUARANTINE
        else:
            decision = Decision.DENY

        latency_ms = (time.time() - start_time) * 1000

        # Build response
        response = {
            "decision": decision.value,
            "risk_score": risk_score,
            "latency_ms": round(latency_ms, 2),
            "agent_id": agent.id,
            "action": action.action_type,
            "target": action.target,
        }

        if decision == Decision.ALLOW:
            # Generate authorization token
            response["token"] = f"scbe_{agent.id}_{int(time.time())}_{action.action_type.lower()}"
        elif decision == Decision.DENY:
            response["reason"] = f"Risk score {risk_score:.2f} exceeds threshold {self.quarantine_threshold}"

        # Log for audit
        self.audit_log.append({
            **response,
            "timestamp": time.time(),
        })

        return response

    def authorize_with_consensus(
        self,
        requesting_agent: Agent,
        action: Action,
        voting_agents: List[Agent],
        threshold: float = 0.67
    ) -> Dict[str, Any]:
        """
        For high-risk actions, require consensus from multiple agents.

        Args:
            requesting_agent: The agent requesting the action
            action: The action to authorize
            voting_agents: List of agents who vote on approval
            threshold: Fraction of votes needed to approve (default 67%)

        Returns:
            Response with consensus details
        """
        # First, basic authorization check
        base_result = self.authorize(requesting_agent, action)

        if base_result["decision"] == Decision.DENY.value:
            return base_result

        # Collect votes from voting agents
        votes = []
        for voter in voting_agents:
            # Each voter evaluates based on their perspective
            voter_context = voter.to_context_vector()
            vote_score = 1.0 - voter_context[1]  # Trust inverted as skepticism

            # Voter approves if their trust in the requester is high enough
            approves = requesting_agent.trust_score > 0.5 and voter.trust_score > 0.5

            votes.append({
                "voter_id": voter.id,
                "voter_name": voter.name,
                "approves": approves,
                "confidence": voter.trust_score,
            })

        # Calculate consensus
        approve_count = sum(1 for v in votes if v["approves"])
        total_votes = len(votes)
        approval_rate = approve_count / total_votes if total_votes > 0 else 0

        consensus_reached = approval_rate >= threshold

        return {
            **base_result,
            "consensus": {
                "required_threshold": threshold,
                "approval_rate": round(approval_rate, 2),
                "votes": votes,
                "reached": consensus_reached,
            },
            "decision": Decision.ALLOW.value if consensus_reached else Decision.DENY.value,
        }

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the audit log for compliance review"""
        return self.audit_log


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def demo_single_agent():
    """Demo: Single agent requesting actions"""
    print("\n" + "="*60)
    print(" DEMO 1: Single Agent Authorization")
    print("="*60)

    gate = SCBEGovernanceGate()

    # Create a code-generation agent
    codegen = Agent(
        id="codegen-001",
        name="CodeGen-GPT4",
        role="developer",
        trust_score=0.75,
        capabilities=["code_generation", "refactoring", "testing"]
    )

    # Actions with varying risk levels
    actions = [
        Action("codegen-001", "READ", "src/utils.py", sensitivity=0.1),
        Action("codegen-001", "WRITE", "src/new_feature.py", sensitivity=0.4),
        Action("codegen-001", "EXECUTE", "npm test", sensitivity=0.3),
        Action("codegen-001", "DEPLOY", "production", sensitivity=0.9),
        Action("codegen-001", "DELETE", "database", sensitivity=1.0),
    ]

    print(f"\nAgent: {codegen.name} (trust: {codegen.trust_score})")
    print("-" * 60)

    for action in actions:
        result = gate.authorize(codegen, action)
        icon = {"ALLOW": "[OK]", "QUARANTINE": "[!!]", "DENY": "[XX]"}[result["decision"]]
        print(f"  {icon} {action.action_type:8} {action.target:20} -> {result['decision']:10} (risk: {result['risk_score']:.2f})")


def demo_multi_agent():
    """Demo: Multi-agent fleet with consensus"""
    print("\n" + "="*60)
    print(" DEMO 2: Multi-Agent Fleet with Consensus")
    print("="*60)

    gate = SCBEGovernanceGate()

    # Create a fleet of agents
    fleet = [
        Agent("codegen-001", "CodeGen-GPT4", "developer", 0.75, ["code_generation"]),
        Agent("security-001", "Security-Claude", "security", 0.90, ["security_scan", "code_review"]),
        Agent("deploy-001", "Deploy-Bot", "deployer", 0.65, ["deployment", "infrastructure"]),
        Agent("test-001", "Test-Runner", "tester", 0.80, ["testing", "validation"]),
    ]

    print(f"\nFleet registered: {len(fleet)} agents")
    for agent in fleet:
        print(f"  - {agent.name} ({agent.role}, trust: {agent.trust_score})")

    # High-risk action requiring consensus
    print("\n" + "-" * 60)
    print("High-risk action: Deploy to production")
    print("-" * 60)

    deploy_action = Action(
        agent_id="deploy-001",
        action_type="DEPLOY",
        target="production",
        sensitivity=0.9,
        requires_consensus=True
    )

    deployer = fleet[2]
    voters = [fleet[0], fleet[1], fleet[3]]  # Other agents vote

    result = gate.authorize_with_consensus(deployer, deploy_action, voters, threshold=0.67)

    print(f"\nRequesting agent: {deployer.name}")
    print(f"Action: {deploy_action.action_type} -> {deploy_action.target}")
    print(f"\nVoting results:")

    for vote in result["consensus"]["votes"]:
        status = "APPROVE" if vote["approves"] else "REJECT"
        print(f"  - {vote['voter_name']}: {status} (confidence: {vote['confidence']:.2f})")

    print(f"\nConsensus: {result['consensus']['approval_rate']*100:.0f}% approval")
    print(f"Threshold: {result['consensus']['required_threshold']*100:.0f}%")
    print(f"Decision: {result['decision']}")


def demo_audit_trail():
    """Demo: Audit trail for compliance"""
    print("\n" + "="*60)
    print(" DEMO 3: Audit Trail for Compliance")
    print("="*60)

    gate = SCBEGovernanceGate()

    agent = Agent("audit-test", "AuditTestAgent", "developer", 0.7, ["testing"])

    # Perform several actions
    actions = [
        Action("audit-test", "READ", "config.json", 0.2),
        Action("audit-test", "WRITE", "output.log", 0.3),
        Action("audit-test", "EXECUTE", "python test.py", 0.4),
    ]

    for action in actions:
        gate.authorize(agent, action)

    # Get audit log
    print("\nAudit Log:")
    print("-" * 60)

    for entry in gate.get_audit_log():
        print(f"  [{entry['decision']}] {entry['action']} -> {entry['target']}")
        print(f"       Agent: {entry['agent_id']}, Risk: {entry['risk_score']:.2f}")
        print(f"       Latency: {entry['latency_ms']:.2f}ms")
        print()


def demo_integration_pattern():
    """Demo: How to integrate into your code"""
    print("\n" + "="*60)
    print(" DEMO 4: Integration Pattern")
    print("="*60)

    print("""
    # In your agent code:

    from examples.agent_integration import SCBEGovernanceGate, Agent, Action

    # Initialize once
    gate = SCBEGovernanceGate()

    # Define your agent
    my_agent = Agent(
        id="my-agent-001",
        name="MyAgent",
        role="developer",
        trust_score=0.8,
        capabilities=["code_generation", "testing"]
    )

    # Before any action:
    def do_action(action_type, target):
        action = Action(my_agent.id, action_type, target)
        result = gate.authorize(my_agent, action)

        if result["decision"] == "ALLOW":
            # Proceed with action
            print(f"Authorized: {result['token']}")
            return perform_action(action_type, target)
        elif result["decision"] == "QUARANTINE":
            # Limited/monitored mode
            print("Running in quarantine mode")
            return perform_action_limited(action_type, target)
        else:
            # Denied
            print(f"Denied: {result['reason']}")
            return None
    """)


def main():
    """Run all demos"""
    print("="*60)
    print(" SCBE-AETHERMOORE Agent Integration Demo")
    print(" AI Governance Made Practical")
    print("="*60)

    demo_single_agent()
    demo_multi_agent()
    demo_audit_trail()
    demo_integration_pattern()

    print("\n" + "="*60)
    print(" All demos complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Copy SCBEGovernanceGate to your project")
    print("  2. Define your agents with Agent dataclass")
    print("  3. Call gate.authorize() before any action")
    print("  4. Handle ALLOW/QUARANTINE/DENY decisions")
    print()


if __name__ == "__main__":
    main()
