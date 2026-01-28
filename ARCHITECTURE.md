# Architecture Overview

This document serves as a critical, living template designed to equip agents with a rapid and comprehensive understanding of the SCBE-AETHERMOORE codebase's architecture, enabling efficient navigation and effective contribution from day one.

---

## 1. Project Structure

```
scbe-aethermoore-demo/
├── src/                              # Main TypeScript/Python source code
│   ├── harmonic/                     # 14-Layer pipeline (CORE - 40+ files)
│   │   ├── pipeline14.ts             # Layers 1-14 implementation
│   │   ├── hyperbolic.ts             # Poincaré ball operations (L5, L7)
│   │   ├── harmonicScaling.ts        # Risk amplification (L12)
│   │   ├── audioAxis.ts              # Hilbert transform telemetry (L14)
│   │   ├── halAttention.ts           # Attention mechanisms
│   │   ├── hamiltonianCFI.ts         # Hamiltonian dynamics
│   │   ├── sacredTongues.ts          # 6×256 vocabulary tokenizer
│   │   ├── spiralSeal.ts             # Spiral sealing protocol
│   │   ├── phdm.ts                   # Poincaré half-plane drift monitor
│   │   ├── qcLattice.ts              # Quantum coherence lattice
│   │   ├── languesMetric.ts          # Language metrics
│   │   ├── vacuumAcoustics.ts        # Vacuum field analysis
│   │   └── assertions.ts             # Runtime assertions
│   ├── crypto/                       # Cryptographic primitives (9 files)
│   │   ├── envelope.ts               # AES-256-GCM sealed envelopes
│   │   ├── pqc.ts                    # Post-quantum (ML-KEM-768, ML-DSA-65)
│   │   ├── hkdf.ts                   # HKDF-SHA256 key derivation
│   │   ├── kms.ts                    # Key management system
│   │   ├── nonceManager.ts           # Nonce generation & tracking
│   │   ├── replayGuard.ts            # Replay attack prevention (Bloom filter)
│   │   ├── bloom.ts                  # Bloom filter implementation
│   │   ├── jcs.ts                    # JSON Canonicalization Scheme
│   │   └── sacred_eggs.py            # Sacred eggs integration (Python)
│   ├── spiralverse/                  # RWP v2.1 protocol (4 files)
│   │   ├── rwp.ts                    # Multi-signature envelopes
│   │   ├── policy.ts                 # Policy enforcement matrix
│   │   ├── types.ts                  # Protocol types
│   │   └── index.ts                  # Exports
│   ├── symphonic/                    # Symphonic cipher (6 files)
│   │   ├── HybridCrypto.ts           # Hybrid encryption scheme
│   │   ├── Feistel.ts                # Feistel network
│   │   ├── FFT.ts                    # Fast Fourier Transform
│   │   ├── Complex.ts                # Complex number operations
│   │   ├── SymphonicAgent.ts         # Agent framework
│   │   └── audio/                    # Audio processing
│   ├── network/                      # Network security (4 files)
│   │   ├── combat-network.ts         # Multi-path routing (redundancy)
│   │   ├── space-tor-router.ts       # SpaceTor relay routing
│   │   ├── hybrid-crypto.ts          # Hybrid space encryption
│   │   └── index.ts
│   ├── fleet/                        # Multi-agent orchestration (2 files)
│   │   ├── redis-orchestrator.ts     # Redis/BullMQ fleet manager
│   │   └── index.ts
│   ├── rollout/                      # Deployment strategies (2 files)
│   │   ├── canary.ts                 # Canary deployments
│   │   └── circuitBreaker.ts         # Circuit breaker pattern
│   ├── selfHealing/                  # Self-healing systems (3 files)
│   │   ├── coordinator.ts
│   │   ├── deepHealing.ts
│   │   └── quickFixBot.ts
│   ├── metrics/                      # Telemetry & observability
│   │   └── telemetry.ts
│   ├── api/                          # REST API (Python - 2 files)
│   │   ├── main.py                   # FastAPI server (6 endpoints)
│   │   └── persistence.py            # Firebase Firestore integration
│   └── index.ts                      # Main exports
├── api/                              # Production API source
│   ├── main.py
│   └── persistence.py
├── tests/                            # 728+ tests (40+ Python files, TS files)
│   ├── harmonic/                     # Layer tests (~180 tests)
│   ├── enterprise/                   # Compliance tests (~100 tests)
│   ├── network/                      # Network tests (~70 tests)
│   ├── spiralverse/                  # RWP tests (~82 tests)
│   ├── spectral/                     # Coherence tests (~80 tests)
│   ├── symphonic/                    # Symphonic cipher tests
│   └── orchestration/                # Fleet tests
├── aws/                              # AWS Lambda deployment
│   ├── lambda_handler.py             # Lambda entry point (Mangum)
│   ├── requirements-lambda.txt
│   └── README.md
├── examples/                         # Example agents & integrations
│   ├── MyAgent/
│   └── SCBEAgent/
├── docs/                             # Documentation structure
│   ├── 00-overview/
│   ├── 01-architecture/
│   ├── 03-deployment/
│   ├── 05-industry-guides/
│   ├── 06-integration/
│   └── 08-reference/
├── .github/workflows/                # CI/CD pipelines
│   ├── ci.yml                        # TypeScript & Python testing
│   ├── deploy-aws.yml                # AWS Lambda deployment
│   ├── docs.yml
│   └── release.yml
└── config/                           # Configuration files
```

**Statistics:**
- TypeScript source files: 494 files
- Python source files: 101 files
- Total test files: 728+ tests
- Documentation: Comprehensive multi-section docs

---

## 2. High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SCBE-AETHERMOORE                               │
│                     AI Governance & Safety Framework                         │
└─────────────────────────────────────────────────────────────────────────────┘

                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REST API (FastAPI)                                │
│  POST /v1/authorize │ POST /v1/agents │ GET /v1/audit │ WS /ws/dashboard   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        14-LAYER HARMONIC PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  L1: Complex State    │  L2: Realification     │  L3: SPD Weighting        │
│  L4: Poincaré Embed   │  L5: Hyperbolic Dist   │  L6: Breathing Transform  │
│  L7: Möbius Addition  │  L8: Realm Distance    │  L9: Spectral Coherence   │
│  L10: Spin Coherence  │  L11: Triadic Temporal │  L12: Harmonic Scaling    │
│  L13: Risk Decision   │  L14: Audio Axis       │                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Cryptography │         │   RWP Protocol  │         │  Fleet Manager  │
│   AES-256-GCM │         │   Multi-Sig     │         │  Redis/BullMQ   │
│   ML-KEM-768  │         │   6 Tongues     │         │  9 Agent Roles  │
│   ML-DSA-65   │         │   Policy Matrix │         │  Job Queue      │
└───────────────┘         └─────────────────┘         └─────────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE LAYER                                  │
│              Firebase Firestore │ Redis (optional) │ In-Memory              │
│        audit_logs │ trust_scores │ agent_registry │ alerts                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DECISION OUTPUT                                  │
│                    ALLOW  │  QUARANTINE  │  DENY                           │
│                    + risk_score + reasoning + audit_id                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Components

### 3.1. REST API Gateway

**Name:** FastAPI REST Server

**Description:** Production-ready API for agent governance decisions with audit trail. Provides endpoints for authorization decisions, agent registration, consensus management, and real-time monitoring.

**Technologies:** FastAPI, Uvicorn, Pydantic, Mangum (AWS Lambda adapter)

**Deployment:** AWS Lambda (us-west-2), Docker containers, local development

**Endpoints:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/v1/authorize` | Main governance decision gate (ALLOW/DENY/QUARANTINE) |
| POST | `/v1/agents` | Register AI agent |
| GET | `/v1/agents/{id}` | Get agent metadata |
| POST | `/v1/consensus` | Multi-signature approval |
| GET | `/v1/audit/{id}` | Retrieve decision audit |
| GET | `/v1/health` | Health check |
| WS | `/ws/dashboard` | Real-time monitoring |

---

### 3.2. 14-Layer Harmonic Pipeline

**Name:** Core Governance Pipeline

**Description:** The heart of SCBE-AETHERMOORE. Maps context into hyperbolic space for deterministic governance decisions. Prevents unauthorized actions through geometric + coherence metrics.

**Technologies:** TypeScript, NumPy, SciPy, FFT, Poincaré ball geometry

**Location:** `src/harmonic/`

| Layer | Function | Purpose |
|-------|----------|---------|
| **L1** | `layer1ComplexState` | Context → Complex vector (amplitude + phase) |
| **L2** | `layer2Realification` | ℂᴰ → ℝ²ᴰ isometric embedding |
| **L3** | `layer3WeightedTransform` | SPD metric weighting (golden ratio φᵏ) |
| **L4** | `layer4PoincareEmbedding` | Embed into Poincaré ball (‖u‖ < 1) |
| **L5** | `hyperbolicDistance` | d_ℍ(u,v) via arcosh formula |
| **L6** | `layer6BreathingTransform` | Temporal modulation (diffeomorphism) |
| **L7** | `mobiusAddition` | Möbius isometry (gyrovector space) |
| **L8** | `layer8RealmDistance` | Min distance to trusted realm centers |
| **L9** | `layer9SpectralCoherence` | FFT-based pattern stability |
| **L10** | `layer10SpinCoherence` | Phase alignment measure |
| **L11** | `layer11TriadicTemporal` | Multi-timescale aggregation |
| **L12** | `harmonicScale` | Risk amplifier: H(d,R) = φᵈ / (1 + e⁻ᴿ) |
| **L13** | `layer13RiskDecision` | **ALLOW / QUARANTINE / DENY** |
| **L14** | `computeAudioAxisFeatures` | Hilbert transform telemetry |

---

### 3.3. Cryptographic System

**Name:** Sealed Envelope Cryptography

**Description:** Cryptographically seal messages between agents. Tamper-evident, replay-protected, quantum-resistant.

**Technologies:** AES-256-GCM, HKDF-SHA256, liboqs (ML-KEM-768, ML-DSA-65), Argon2id

**Location:** `src/crypto/`

**Components:**
- **envelope.ts** - AES-256-GCM sealed containers with AAD binding
- **pqc.ts** - ML-KEM-768 (KEMs) + ML-DSA-65 (signatures) - NIST FIPS 203/204
- **replayGuard.ts** - Nonce + Bloom filter to prevent replay attacks
- **hkdf.ts** - HKDF-SHA256 key derivation
- **kms.ts** - Master key management
- **bloom.ts** - Probabilistic duplicate detection
- **jcs.ts** - JSON canonicalization for consistent hashing

---

### 3.4. RWP Protocol

**Name:** Recursive Weighted Protocol v2.1

**Description:** Domain-separated multi-signature protocol. Policies enforce which signatures are required for governance actions.

**Technologies:** HMAC-SHA256, Sacred Tongues tokenization

**Location:** `src/spiralverse/`

**Components:**
- **rwp.ts** - Multi-signature HMAC-SHA256 via Sacred Tongues (6 domain-separated versions)
- **policy.ts** - Policy enforcement matrix (standard/strict/secret/critical)
- **Tongue IDs:** KO, RU, UM, AV, JE, DR (6×256 vocabulary)

---

### 3.5. Fleet Orchestration

**Name:** Redis/BullMQ Fleet Manager

**Description:** Horizontal scaling for multi-agent systems. Survives server restarts with job persistence.

**Technologies:** Redis, BullMQ, TypeScript

**Location:** `src/fleet/`

**Agent Roles:**
- captain, architect, researcher, developer, qa, security, reviewer, deployer, monitor

**Features:**
- Job persistence across restarts
- Automatic retry logic
- Concurrency control

---

### 3.6. Network Security

**Name:** Combat Network & SpaceTor Router

**Description:** Redundant routing for high-stakes communication. Prevents single-path failures.

**Technologies:** TypeScript, multi-path routing, hybrid encryption

**Location:** `src/network/`

**Components:**
- **combat-network.ts** - Multi-path routing with disjoint paths (no shared nodes)
- **space-tor-router.ts** - SpaceTor relay network (multi-hop routing)
- **hybrid-crypto.ts** - Hybrid encryption for transit

---

### 3.7. Symphonic Cipher

**Name:** Frequency-Domain Encryption

**Description:** Advanced encryption using FFT & frequency analysis. Resistant to linear cryptanalysis.

**Technologies:** FFT, Feistel network, complex number arithmetic

**Location:** `src/symphonic/`

---

### 3.8. Self-Healing System

**Name:** Automated Recovery

**Description:** Automatic fault detection & remediation without manual intervention.

**Technologies:** TypeScript, health monitoring, circuit breakers

**Location:** `src/selfHealing/`

---

## 4. Data Stores

### 4.1. Firebase Firestore (Primary)

**Name:** Primary Cloud Database

**Type:** Firebase Firestore (document database)

**Purpose:** Durable storage with immutable audit trail for governance decisions, agent registry, and trust scores.

**Collections:**

| Collection | Schema | Purpose |
|-----------|--------|---------|
| `audit_logs` | `{decision_id, timestamp, agent_id, decision, risk_score, reasoning, context}` | Immutable decision records |
| `trust_scores` | `{agent_id, score, history[], updated_at}` | Historical trust metrics |
| `agent_registry` | `{agent_id, public_key, capabilities, registered_at}` | Agent metadata |
| `alerts` | `{id, type, message, timestamp, webhook_url}` | Webhook/Zapier events |

**Setup:**
1. Create Firebase project at https://console.firebase.google.com
2. Download service account key JSON
3. Set `GOOGLE_APPLICATION_CREDENTIALS` or `FIREBASE_CONFIG` env variable

---

### 4.2. Redis (Optional)

**Name:** In-Memory Data Store

**Type:** Redis

**Purpose:** Used for BullMQ job queues and distributed state in fleet orchestration.

**Configuration:** Default localhost:6379

---

### 4.3. In-Memory Stores (Demo)

**Name:** Temporary Storage

**Type:** In-memory dictionaries

**Purpose:** Development and demo mode storage within `api/main.py`

**Stores:**
- `AGENTS_STORE` - Temporary agent registry
- `DECISIONS_STORE` - Temporary decision cache
- `CONSENSUS_STORE` - Multi-signature state

---

## 5. External Integrations / APIs

| Service | Purpose | Integration Method |
|---------|---------|-------------------|
| **Firebase/Firestore** | Cloud persistence, audit logs | Firebase Admin SDK |
| **AWS Lambda** | Serverless compute deployment | Mangum ASGI adapter |
| **Redis** | Job queue, distributed state | ioredis client |
| **Webhooks/Zapier** | Alert notifications | HTTP POST callbacks |

---

## 6. Deployment & Infrastructure

### Cloud Provider
AWS (primary), GCP (Firebase), Docker (local/self-hosted)

### Key Services Used
- **AWS Lambda** - Serverless compute (us-west-2)
- **AWS API Gateway** - HTTP routing
- **Firebase Firestore** - Document database
- **Redis** - Job queues (optional)
- **Docker** - Container images

### CI/CD Pipeline
**GitHub Actions** (`.github/workflows/`)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push/PR to main, develop | Test TypeScript (20.x, 22.x) + Python (3.9-3.12) |
| `deploy-aws.yml` | push to main | Build & deploy to AWS Lambda |
| `docs.yml` | Documentation updates | Generate TypeDoc + publish |
| `release.yml` | Version tags | Create GitHub releases |

### Monitoring & Logging
- JSON structured logging with timestamps
- CloudWatch (AWS Lambda)
- Telemetry via `src/metrics/telemetry.ts`

### Docker Configuration

**Multi-stage Dockerfile:**
1. TypeScript build (Node 20-alpine)
2. liboqs C library compilation (Python 3.11-slim)
3. Python environment with PQC
4. Final runtime (Python 3.11-slim + Node 20)

**Environment Variables:**
```bash
SCBE_ENV=production
SCBE_PQC_BACKEND=liboqs
SCBE_LOG_LEVEL=INFO
SCBE_API_KEY=<set at runtime>
FIREBASE_CONFIG=<optional Firebase JSON>
```

---

## 7. Security Considerations

### Authentication
- **API Key** - X-API-Key header for REST endpoints
- **Bearer Token** - WebSocket authentication

### Authorization
- **Policy Matrix** - standard/strict/secret/critical levels
- **Multi-signature** - Consensus required for critical operations

### Data Encryption
- **In Transit** - TLS 1.3
- **At Rest** - AES-256-GCM
- **Key Derivation** - HKDF-SHA256
- **Password Hashing** - Argon2id (0.5s/attempt)

### Post-Quantum Cryptography
- **ML-KEM-768** - Key encapsulation (NIST FIPS 203)
- **ML-DSA-65** - Digital signatures (NIST FIPS 204)
- **Backend** - liboqs library

### Key Security Practices
- Replay attack prevention (Bloom filter + nonce tracking)
- Tamper detection via HMAC verification
- JSON Canonicalization Scheme (JCS) for consistent hashing
- Non-root container execution (UID 1000)
- No hardcoded secrets in Docker images

---

## 8. Development & Testing Environment

### Local Setup Instructions

```bash
# Clone repository
git clone <repository-url>
cd scbe-aethermoore-demo

# Install dependencies
npm install
pip install -r requirements.txt

# Build TypeScript
npm run build

# Run tests
npm test              # TypeScript tests (Vitest)
npm run test:python   # Python tests (pytest)
npm run test:all      # Both suites

# Start API server
uvicorn api.main:app --reload --port 8080

# Docker deployment
npm run docker:compose
```

### Testing Frameworks

**TypeScript:**
- **Vitest 4.0+** - Unit and integration tests
- **fast-check** - Property-based testing
- **Coverage:** 80% lines target

**Python:**
- **pytest 7.4+** - Test framework
- **Hypothesis** - Property-based testing (200+ iterations)
- **pytest-cov** - Coverage reporting
- **pytest-asyncio** - Async test support

### Code Quality Tools

| Tool | Language | Purpose |
|------|----------|---------|
| **Prettier 3.2** | TypeScript/JS | Code formatting |
| **TypeScript Compiler** | TypeScript | Type checking |
| **Black** | Python | Code formatting |
| **flake8** | Python | Linting |
| **mypy** | Python | Type checking |
| **Snyk** | All | Security scanning |

### npm Scripts

```bash
npm run clean              # Remove dist/
npm run build              # Compile TypeScript
npm run build:watch        # Watch mode
npm test                   # Run Vitest
npm run test:python        # Run pytest
npm run test:all           # Both suites
npm run typecheck          # Type check only
npm run format             # Prettier format
npm run lint               # Prettier check
npm run docker:build       # Build container
npm run docker:run         # Run container
npm run docker:compose     # Docker Compose up
```

---

## 9. Future Considerations / Roadmap

- **Patent Filing** - USPTO deadline January 31, 2026 (Provisional #63/961,403)
- **FIPS 140-3 Certification** - Enterprise cryptographic module validation
- **SOC 2 Type II** - Compliance audit preparation
- **ISO 27001** - Information security certification
- **Event-Driven Architecture** - Real-time streaming for high-volume deployments
- **Kubernetes Deployment** - Horizontal pod autoscaling

---

## 10. Project Identification

| Field | Value |
|-------|-------|
| **Project Name** | SCBE-AETHERMOORE |
| **Full Name** | Sealed Container Binary Envelope - AETHERMOORE |
| **Version** | 3.0.0 |
| **Repository URL** | https://github.com/issdandavis/scbe-aethermoore-demo |
| **Primary Contact** | issdandavis |
| **Date of Last Update** | 2026-01-28 |

---

## 11. Glossary / Acronyms

| Acronym | Definition |
|---------|------------|
| **SCBE** | Sealed Container Binary Envelope - cryptographic envelope format |
| **AETHERMOORE** | AI governance framework name |
| **RWP** | Recursive Weighted Protocol - multi-signature scheme |
| **PQC** | Post-Quantum Cryptography |
| **ML-KEM** | Module-Lattice Key Encapsulation Mechanism (NIST FIPS 203) |
| **ML-DSA** | Module-Lattice Digital Signature Algorithm (NIST FIPS 204) |
| **HKDF** | HMAC-based Key Derivation Function |
| **JCS** | JSON Canonicalization Scheme |
| **GCM** | Galois/Counter Mode (AES encryption mode) |
| **AAD** | Additional Authenticated Data |
| **SPD** | Symmetric Positive Definite (matrix type) |
| **FFT** | Fast Fourier Transform |
| **PHDM** | Poincaré Half-plane Drift Monitor |
| **Sacred Tongues** | 6 domain-separated signature vocabularies (KO, RU, UM, AV, JE, DR) |
| **Fleet** | Multi-agent orchestration system |
| **BullMQ** | Redis-based job queue for Node.js |
