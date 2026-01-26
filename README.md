# SCBE-AETHERMOORE v4.0

> **Spectral Context-Bound Encryption + Hyperbolic Governance for AI-to-AI Communication**

[![Patent Pending](https://img.shields.io/badge/Patent-USPTO%20%2363%2F961%2C403-blue)](https://github.com/ISDanDavis2/scbe-aethermoore)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4-blue)](https://www.typescriptlang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-692%20passing-brightgreen)]()

---

## TL;DR

| | |
|---|---|
| **Problem** | AI agents talking to each other have no governance layer—any agent can tell another to do anything |
| **Solution** | A 14-layer pipeline that embeds context into hyperbolic space and gates actions with ALLOW/QUARANTINE/DENY |
| **Use Case** | Multi-agent systems, tool-calling AI, autonomous workflows needing auditable authorization |
| **Try It** | `git clone` → `npm install` → `npm test` (692 tests pass) |

---

## Get Started in 2 Minutes

```bash
# Clone the repo
git clone https://github.com/ISDanDavis2/scbe-aethermoore-demo.git
cd scbe-aethermoore-demo

# Install dependencies
npm install

# Run all tests (692 tests)
npm test

# Run the Python demo (memory shard governance)
pip install numpy scipy argon2-cffi pycryptodome
python demo_memory_shard.py

# Start the API server + dashboard
pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload
# Open http://localhost:8000/docs for API
# Open dashboard/scbe_monitor.html for real-time UI
```

**Status (Jan 2026):** Pilot-ready prototype with functional core. Not yet compliance-certified for regulated enterprise.

---

## What This Is

SCBE-AETHERMOORE is a framework for **agent-to-agent governance and communication** where:

- Agents exchange messages in a **sealed envelope** (cryptographic binding)
- The receiver runs a **deterministic governance gate** before acting
- "Context drift" and abnormal behavior are scored using **geometry + coherence metrics**
- Outputs are **fail-closed** with **low-leak error behavior** (no helpful oracle)

Think of it as:

> **"Crypto for the message + math-based rules for whether the agent is allowed to act on it."**

---

## What It's For

Designed for situations like:

- AI "workers" collaborating (tools, planners, schedulers, multi-agent systems)
- Preventing **replay**, **tampering**, **downgrade**, and **unauthorized actions**
- Forcing agent actions through **ALLOW / QUARANTINE / DENY** decisions
- Maintaining an **audit trail** across all decisions

---

## Core Idea (Plain English)

1. Turn "what's happening" (context, intent, telemetry) into a vector
2. Embed that vector into a **Poincaré ball** (hyperbolic trust space)
3. Measure **distance to a trusted realm** and apply a risk amplifier
4. Combine with coherence checks (spectral / phase stability)
5. Output a governance decision

This produces a score that's:

- **Auditable** — you can show the math
- **Stable** — small changes don't cause chaotic decisions
- **Integratable** — wraps around existing systems

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SCBE-AETHERMOORE Authorization Flow                  │
└─────────────────────────────────────────────────────────────────────────────┘

  Agent A                                                           Agent B
  ───────                                                           ───────
     │                                                                 │
     │  1. Create sealed envelope                                      │
     │     ┌──────────────────────────┐                                │
     │     │ Payload + Context Vector │                                │
     │     │ + PQC Signatures (opt)   │                                │
     │     └──────────────────────────┘                                │
     │                │                                                │
     │                ▼                                                │
     │     ┌──────────────────────────┐                                │
     │     │   SCBE Gateway (API)     │◄───────────────────────────────│
     │     │   /seal-memory           │   2. Submit for authorization  │
     │     └──────────────────────────┘                                │
     │                │                                                │
     │                ▼                                                │
     │     ┌──────────────────────────────────────────────────────┐    │
     │     │              14-Layer Pipeline                       │    │
     │     │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ │    │
     │     │  │Complex  │→│Poincaré │→│Coherence│→│  Decision   │ │    │
     │     │  │Encoding │ │Embedding│ │ Scoring │ │   Engine    │ │    │
     │     │  └─────────┘ └─────────┘ └─────────┘ └─────────────┘ │    │
     │     └──────────────────────────────────────────────────────┘    │
     │                │                                                │
     │                ▼                                                │
     │     ┌──────────────────────────┐    ┌────────────────────────┐  │
     │     │   ALLOW │ QUARANTINE │   │───►│  Audit Log (append)    │  │
     │     │         │ DENY       │   │    │  + WebSocket broadcast │  │
     │     └──────────────────────────┘    └────────────────────────┘  │
     │                │                                                │
     │                ▼                                                │
     │     3. Decision returned                                        │
     │        (action gated)                                           │
     └─────────────────────────────────────────────────────────────────┘

  Components:
  ├── src/api/main.py         FastAPI server with /seal-memory, /retrieve-memory
  ├── src/harmonic/           Hyperbolic geometry + Möbius transforms
  ├── src/crypto/             PQC (ML-KEM-768, ML-DSA-65) + envelope signing
  ├── src/fleet/              Redis/BullMQ agent orchestration
  └── dashboard/              Real-time monitoring UI (WebSocket)
```

### 14-Layer Reference Model

| Layer | Name                | Purpose                                      | Technology                        |
| ----- | ------------------- | -------------------------------------------- | --------------------------------- |
| 1     | Complex State       | Context → complex vector (amplitude + phase) | NumPy complex128                  |
| 2     | Realification       | ℂᴰ → ℝ²ᴰ embedding                           | Linear algebra                    |
| 3     | Weighted Transform  | SPD metric weighting                         | Positive-definite matrices        |
| 4     | Poincaré Embedding  | Map to hyperbolic ball (‖u‖ < 1)             | Hyperbolic geometry               |
| 5     | Hyperbolic Distance | d_ℍ(u,v) via arcosh formula                  | `src/harmonic/hyperbolic.ts`      |
| 6     | Breathing Transform | Temporal modulation (diffeomorphism)         | Time-varying scaling              |
| 7     | Phase Transform     | Möbius isometry (gyrovector addition)        | `src/harmonic/mobius.ts`          |
| 8     | Realm Distance      | Min distance to trusted zone centers         | Trust realm configuration         |
| 9     | Spectral Coherence  | FFT-based pattern stability                  | SciPy FFT                         |
| 10    | Spin Coherence      | Phase alignment measure                      | Complex phase analysis            |
| 11    | Triadic Temporal    | Multi-timescale distance aggregation         | Weighted averaging                |
| 12    | Harmonic Scaling    | Risk amplifier: H(d,R) = φᵈ / (1 + e⁻ᴿ)      | `src/harmonic/scaling.ts`         |
| 13    | Risk Decision       | ALLOW / QUARANTINE / DENY                    | Threshold engine                  |
| 14    | Audio Axis          | Hilbert transform telemetry                  | Real-time streaming               |

> **Note:** Context vectors are configurable dimension (typically 6D or 12D depending on integration).

---

## What Is Implemented Today

### ✅ Implemented (Pilot-Ready)

- **Hyperbolic embedding + distance metric** (Poincaré ball)
- **Möbius transformations** (true isometries via gyrovector operations)
- **Risk amplification** (harmonic scaling)
- **Three-way governance decision**: `ALLOW / QUARANTINE / DENY`
- **Demo**: Memory shard sealed retrieval controlled by governance
- **12,000+ lines of tests** (TypeScript + Python)
- **Dashboard**: Real-time monitoring UI
- **API endpoints**: 6 REST endpoints for integration
- **Fleet orchestration**: Redis/BullMQ-based agent coordination

### ⚠️ Optional / Environment-Dependent

- **PQC integration** (ML-KEM-768, ML-DSA-65) — requires liboqs bindings
- **Sacred Tongue tokenizer** — full 6×256 vocabulary implemented
- **Enterprise compliance** (SOC2/FIPS/ISO) — requires external certification

---

## Test Suite Overview

**692 tests** across 25 test files covering all critical paths:

| Category | Tests | What's Covered |
|----------|-------|----------------|
| **Hyperbolic Geometry** | ~180 | Poincaré ball embedding, distance metrics, Möbius transforms, boundary behavior |
| **Harmonic Pipeline** | ~120 | 14-layer pipeline, scaling functions, CFI properties, audio axis |
| **Enterprise Compliance** | ~100 | FIPS 140-3 properties, SOC 2 controls, agentic guardrails, quantum readiness |
| **Spectral Coherence** | ~80 | FFT stability, phase alignment, pattern detection |
| **Network Security** | ~70 | Combat routing, trust management, path health tracking |
| **Crypto & Envelope** | ~60 | Nonce reuse prevention, tamper detection, provider switching |
| **Integration** | ~82 | RWP policy enforcement, performance budgets, acceptance tests |

```bash
# Run all tests
npm test

# Run specific category
npm test -- tests/harmonic/
npm test -- tests/enterprise/

# Run with coverage
npm test -- --coverage
```

---

## Demo Output

When you run `python demo_memory_shard.py`, you'll see:

```
═══════════════════════════════════════════════════════════════════
  SCBE-AETHERMOORE Memory Shard Demo
  14-Layer Governance Pipeline
═══════════════════════════════════════════════════════════════════

[1] Creating memory shard...
    Shard ID: shard_7a3f2b1c
    Content: "classified: alpha-7 coordinates..."

[2] Sealing with governance envelope...
    Context vector: [0.82, 0.15, 0.91, 0.33, 0.67, 0.44]
    Poincaré embedding: ||u|| = 0.847

[3] Authorization request from Agent-B...
    Hyperbolic distance: 1.23
    Coherence score: 0.89
    Risk amplifier: 2.14

    ╔══════════════════════════════════════╗
    ║  DECISION: ALLOW                     ║
    ║  Latency: 3.2ms                      ║
    ╚══════════════════════════════════════╝

[4] Memory retrieved successfully.
```

---

## How to Try It: Fleet Scenario Demo

The **Fleet Scenario API** lets you simulate an entire multi-agent workflow in one request. This is the "whole body" demo - register agents, run actions through SCBE governance, and see who gets ALLOW / QUARANTINE / DENY.

### Quick Start

```bash
# 1. Start the API server
export SCBE_API_KEY=demo-key-123
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080

# 2. Run a fleet scenario (in another terminal)
curl -X POST http://localhost:8080/v1/fleet/run-scenario \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-123" \
  -d @- << 'EOF'
{
  "scenario_name": "security-test-001",
  "agents": [
    {"agent_id": "codegen-001", "name": "CodeGen-GPT4", "role": "developer", "trust_score": 0.7},
    {"agent_id": "security-001", "name": "Security-Claude", "role": "security", "trust_score": 0.85},
    {"agent_id": "deploy-001", "name": "Deploy-Bot", "role": "deployer", "trust_score": 0.6}
  ],
  "actions": [
    {"agent_id": "codegen-001", "action": "WRITE", "target": "src/main.py", "sensitivity": 0.3},
    {"agent_id": "security-001", "action": "READ", "target": "secrets/api_keys", "sensitivity": 0.9},
    {"agent_id": "deploy-001", "action": "DEPLOY", "target": "production", "sensitivity": 0.8, "requires_consensus": true}
  ],
  "consensus_threshold": 0.67
}
EOF
```

### Expected Output

```json
{
  "scenario_id": "scenario_a3f7b2c1d4e5",
  "scenario_name": "security-test-001",
  "total_actions": 3,
  "allowed": 2,
  "quarantined": 1,
  "denied": 0,
  "allow_rate": 0.667,
  "deny_rate": 0.0,
  "results": [
    {
      "agent_id": "codegen-001",
      "action": "WRITE",
      "decision": "ALLOW",
      "score": 0.712,
      "token": "scbe_a3f7b2c1_dec_..."
    },
    {
      "agent_id": "security-001",
      "action": "READ",
      "decision": "QUARANTINE",
      "score": 0.451,
      "explanation": { "tier_override": "High sensitivity action" }
    },
    {
      "agent_id": "deploy-001",
      "action": "DEPLOY",
      "decision": "ALLOW",
      "score": 0.623,
      "consensus_result": {
        "approval_rate": 0.75,
        "votes_for": 2,
        "votes_against": 0,
        "consensus_reached": true
      }
    }
  ],
  "fleet_health": {
    "total_agents": 3,
    "avg_trust_score": 0.717,
    "healthy": true
  },
  "execution_time_ms": 12.4
}
```

### What's Happening

1. **Agents registered** - Each agent gets a trust score and governance tier
2. **Actions evaluated** - Each action runs through the 14-layer SCBE pipeline
3. **Governance tiers applied** - Actions map to Sacred Tongue tiers (KO→DR)
   - `READ/LIST/QUERY` → KO (trust ≥ 0.1)
   - `WRITE/UPDATE` → AV (trust ≥ 0.3)
   - `EXECUTE/RUN` → RU (trust ≥ 0.5)
   - `DEPLOY/RELEASE` → CA (trust ≥ 0.7)
   - `CONFIGURE/ADMIN` → UM (trust ≥ 0.85)
   - `DELETE/DESTROY` → DR (trust ≥ 0.95)
4. **Consensus (optional)** - Sensitive actions get voted on by other agents
5. **Results aggregated** - Full audit trail with tokens for allowed actions

### Scenario JSON Format

```typescript
{
  "scenario_name": string,        // Identifier for logging
  "agents": [
    {
      "agent_id": string,         // Unique ID
      "name": string,             // Display name
      "role": string,             // captain, developer, security, etc.
      "trust_score": number,      // 0.0 to 1.0
      "governance_tier": string,  // KO, AV, RU, CA, UM, DR
      "capabilities": string[]    // Optional capability tags
    }
  ],
  "actions": [
    {
      "agent_id": string,         // Which agent performs this
      "action": string,           // READ, WRITE, EXECUTE, DEPLOY, DELETE
      "target": string,           // Resource being accessed
      "sensitivity": number,      // 0.0 to 1.0 (higher = riskier)
      "requires_consensus": bool, // Other agents vote?
      "context": object           // Optional extra context
    }
  ],
  "consensus_threshold": number   // 0.5 to 1.0 (default 0.67)
}
```

### Integration with Spiralverse Protocol

The fleet scenario endpoint bridges **SCBE-AETHERMOORE** (security/governance) with **Spiralverse Protocol** (AI-to-AI communication):

- **Six Sacred Tongues** map to governance tiers (trust thresholds)
- **6D Trust Vectors** feed into hyperbolic distance calculations
- **Roundtable consensus** uses Byzantine-resilient voting
- **Polly Pads** track agent XP and progression (TypeScript layer)

See `spiralverse-protocol/` submodule for the full protocol spec.

---

## Additional Commands

### Generate Proof Pack

```bash
./scripts/make_proof_pack.sh
```

Creates timestamped evidence at `docs/evidence/<timestamp>/`:

- `system_info.txt` — environment details
- `npm_test_output.txt` — TypeScript test results (692 tests)
- `pytest_output.txt` — Python test results
- `demo_memory_shard_output.txt` — end-to-end demo
- `pip_freeze.txt`, `npm_deps.txt` — dependency snapshots

### Docker (with real PQC)

```bash
docker build -t scbe-aethermoore .
docker run -p 8000:8000 scbe-aethermoore
```

The Docker image includes liboqs for real ML-KEM-768 and ML-DSA-65 post-quantum cryptography.

---

## API Examples

```bash
# Health check
curl http://localhost:8000/health

# Seal a memory shard (create authorization envelope)
curl -X POST http://localhost:8000/seal-memory \
  -H "Content-Type: application/json" \
  -d '{"shard_id": "test-001", "content": "secret data", "context": [0.5, 0.3, 0.8, 0.2, 0.9, 0.4]}'

# Retrieve with governance check
curl -X POST http://localhost:8000/retrieve-memory \
  -H "Content-Type: application/json" \
  -d '{"shard_id": "test-001", "agent_id": "agent-b", "context": [0.5, 0.3, 0.8, 0.2, 0.9, 0.4]}'

# Response:
# {"decision": "ALLOW", "risk_score": 0.23, "d_star": 1.15, "latency_ms": 4.2}
```

See full API docs at `http://localhost:8000/docs` (Swagger UI) when the server is running.

---

## Security Model

- **Confidentiality/integrity/authenticity** reduce to standard cryptography (AEAD, HKDF, signatures; PQC when enabled)
- **Hyperbolic governance + coherence scoring** are best described as:
  - Authorization policy + anomaly detection, **not** "new cryptography"
- **Fail-closed with low-leak error behavior:** Unauthorized requests return uniform errors, do not expose internal state, and avoid acting as an oracle

---

## Key Differentiator

| Traditional Security   | SCBE-AETHERMOORE                                  |
| ---------------------- | ------------------------------------------------- |
| "Do you have the key?" | "Are you the right entity, in the right context?" |
| Possession-based       | Context-based                                     |
| Binary (yes/no)        | Three-way (allow/quarantine/deny)                 |
| Opaque ML models       | Deterministic, auditable math                     |

---

## Project Structure

```
scbe-aethermoore-demo/
├── src/
│   ├── crypto/          # Cryptographic primitives + PQC
│   ├── harmonic/        # Hyperbolic geometry + scaling
│   ├── fleet/           # Redis/BullMQ orchestration
│   ├── api/             # REST API endpoints
│   └── scbe_14layer_reference.py  # Reference implementation
├── dashboard/           # Monitoring UI
├── tests/               # 12,000+ lines of tests
├── docs/                # Documentation + evidence
└── scripts/             # Build + proof pack generation
```

---

## Roadmap

| Track          | Status         | Description                                            |
| -------------- | -------------- | ------------------------------------------------------ |
| **Pilot**      | ✅ Now         | Demos + integration adapters + runbooks                |
| **Security**   | 🔄 In Progress | Independent audit + threat model + adversarial harness |
| **Enterprise** | ⏳ Planned     | SOC 2 Type II + pentest + operational readiness        |

---

## Comparison: Darktrace vs SCBE-AETHERMOORE

| Dimension           | Darktrace                         | SCBE-AETHERMOORE                      |
| ------------------- | --------------------------------- | ------------------------------------- |
| **Maturity**        | 10+ yrs, thousands of deployments | Pilot-ready prototype                 |
| **Mechanism**       | ML + behavioral baselining        | Deterministic scoring + thresholds    |
| **Explainability**  | Good (factors, summaries)         | Excellent (explicit score components) |
| **AI/Agent Focus**  | Generalized endpoint security     | Purpose-built for agent-to-agent      |
| **Prevention**      | Mostly detection + response       | Action gating (deny/quarantine)       |
| **False Positives** | Tuned over time                   | Deterministic thresholds, tuneable    |
| **PQC**             | Varies by deployment              | Optional, environment-dependent       |

**Positioning:**

- **Darktrace:** Detect weird behavior in networks
- **SCBE-AETHERMOORE:** Govern what AI agents are allowed to do to each other

---

## Patent Notice

> **US Provisional Application #63/961,403** (filed Jan 15, 2026)
> This repository is licensed under Apache 2.0.

---

## Contact

- **Author:** Issac Daniel Davis
- **GitHub:** [@ISDanDavis2](https://github.com/ISDanDavis2)
- **Email:** issdandavis@gmail.com

---

## Documentation

| Document                             | Description                                   |
| ------------------------------------ | --------------------------------------------- |
| `QUICKSTART.md`                      | Get running in 5 minutes                      |
| `FULL_SYSTEM_ENABLEMENT.md`          | Complete technical specification (~20K words) |
| `SYSTEM_INTEGRATION.md`              | Component integration guide                   |
| `docs/ENTERPRISE_TESTING_GUIDE.md`   | 41 correctness properties                     |
| `docs/AUDIT_RESPONSE_ACTION_PLAN.md` | Patent + security fixes                       |
| `PATENT_STRATEGY_ACTION_ITEMS.md`    | Critical improvements                         |

---

_Built for the age of autonomous agents._
