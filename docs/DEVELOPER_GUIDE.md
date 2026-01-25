# SCBE-AETHERMOORE: Developer Guide

## Quick Start

```bash
# Clone and install
git clone https://github.com/issdandavis/scbe-aethermoore-demo.git
cd scbe-aethermoore-demo
pip install -r requirements.txt
npm install

# Run tests
npm test          # TypeScript tests (770+)
pytest tests/     # Python tests (500+)

# Run the demo
python demo/demo_memory_shard.py
```

---

## Core API

### The 14-Layer Pipeline

```python
from src.scbe_14layer_reference import scbe_14layer_pipeline
import numpy as np

# Agent position in 6D hyperbolic space
position = np.array([0.1, 0.2, -0.3, 0.4, -0.1, 0.2])

# Run the full 14-layer security pipeline
result = scbe_14layer_pipeline(
    t=position,           # 6D position vector
    D=6,                  # Dimensions
    w_d=0.3,              # Distance weight
    w_c=0.2,              # Coherence weight
    w_s=0.2,              # Spectral weight
    w_tau=0.15,           # Trust weight
    w_a=0.15,             # Audio weight
    theta1=0.3,           # ALLOW threshold
    theta2=0.7            # QUARANTINE threshold
)

print(f"Decision: {result['decision']}")      # ALLOW, QUARANTINE, or DENY
print(f"Risk Score: {result['risk_prime']}")  # 0.0 to 1.0
print(f"Harmonic Factor: {result['H']}")      # Risk amplification
```

### Understanding the Result

```python
result = {
    'decision': 'ALLOW',      # ALLOW | QUARANTINE | DENY
    'risk_prime': 0.15,       # Final risk score (0-1)
    'H': 1.02,                # Harmonic scaling factor
    'd_hyp': 0.23,            # Hyperbolic distance from origin
    'coherence': 0.95,        # Spectral coherence (0-1)
    'spin_coherence': 0.88,   # Phase alignment (0-1)
    'audio_feature': 0.12,    # Audio axis contribution
}
```

---

## The 14 Layers Explained

| Layer | Function | Purpose |
|-------|----------|---------|
| 0 | Intent Modulation | HMAC-based position scrambling |
| 1 | Complex State | Convert to complex representation |
| 2 | Realification | Map to real 2D coordinates |
| 3 | Weighted Transform | Apply metric tensor G |
| 4 | Poincaré Embedding | Project into hyperbolic ball |
| 5 | Hyperbolic Distance | Calculate curved-space distance |
| 6 | Breathing Transform | Time-varying radius modulation |
| 7 | Phase Transform | Möbius rotation and translation |
| 8 | Realm Distance | Distance to nearest trust realm |
| 9 | Spectral Coherence | FFT-based signal analysis |
| 10 | Spin Coherence | Phase alignment across dimensions |
| 11 | Triadic Temporal | Multi-distance consensus |
| 12 | Harmonic Scaling | H(d,R) = R^(d²) amplification |
| 13 | Risk Decision | ALLOW/QUARANTINE/DENY gate |
| 14 | Audio Axis | Optional audio feature fusion |

---

## Key Security Concepts

### 1. Hyperbolic Distance (Poincaré Ball)

```python
from src.scbe_14layer_reference import layer_5_hyperbolic_distance
import numpy as np

# Two points in the Poincaré ball (must have ||x|| < 1)
u = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2])
v = np.array([0.3, 0.4, 0.3, 0.4, 0.3, 0.4])

distance = layer_5_hyperbolic_distance(u, v)
# Distance grows exponentially near the boundary
```

### 2. Harmonic Scaling (Risk Amplification)

```python
from src.scbe_14layer_reference import layer_12_harmonic_scaling

# Small deviation = minimal amplification
H_small = layer_12_harmonic_scaling(d=0.1, R=10.0)  # ~1.01

# Large deviation = exponential amplification
H_large = layer_12_harmonic_scaling(d=2.0, R=10.0)  # ~7.39

# Attacker deviation = massive amplification
H_attack = layer_12_harmonic_scaling(d=3.0, R=10.0)  # ~20.09
```

### 3. Decision Gate

```python
from src.scbe_14layer_reference import layer_13_risk_decision

# Thresholds
theta1 = 0.3  # Below this = ALLOW
theta2 = 0.7  # Above this = DENY

decision, risk_prime = layer_13_risk_decision(
    Risk_base=0.2,
    H=1.5,
    theta1=theta1,
    theta2=theta2
)
# risk_prime = 0.2 * 1.5 = 0.3 → ALLOW (barely)
```

---

## RWP v3 Protocol (Encryption)

```python
from src.crypto.rwp_v3 import RWPv3Protocol

rwp = RWPv3Protocol()

# Encrypt
envelope = rwp.encrypt(
    password=b"secure_password",
    plaintext=b"Secret message"
)

# Decrypt
plaintext = rwp.decrypt(
    password=b"secure_password",
    envelope=envelope
)
```

---

## Sacred Tongues (Semantic Binding)

```python
from src.crypto.sacred_tongues import SacredTongueTokenizer

tokenizer = SacredTongueTokenizer()

# Tokenize with sacred tongue binding
tokens = tokenizer.tokenize("deploy production")
# Returns tokens with semantic domain verification
```

---

## Attack Scenario: Stolen Credentials

```python
import numpy as np
from src.scbe_14layer_reference import scbe_14layer_pipeline

# Legitimate agent at correct position
legit_position = np.array([0.1, 0.2, 0.1, 0.2, 0.1, 0.2])

# Attacker with stolen key but WRONG position
attacker_position = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

# Legitimate request
legit_result = scbe_14layer_pipeline(t=legit_position, D=6)
print(f"Legitimate: {legit_result['decision']}")  # ALLOW

# Attacker request (same credentials, wrong geometry)
attack_result = scbe_14layer_pipeline(t=attacker_position, D=6)
print(f"Attacker: {attack_result['decision']}")   # DENY
print(f"Risk amplification: {attack_result['H']:.2f}x")
```

---

## API Server

```bash
# Start the server
uvicorn src.api.main:app --reload

# Endpoints:
# POST /seal     - Encrypt and seal data
# POST /retrieve - Decrypt and retrieve data
# GET  /health   - Health check
# GET  /docs     - OpenAPI documentation
```

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| TypeScript (vitest) | 770+ | ✅ |
| Python (pytest) | 500+ | ✅ |
| Enterprise compliance | 50+ | ✅ |
| Property-based | 100+ | ✅ |

```bash
# Run all tests
npm test && pytest tests/ -v
```

---

## File Structure

```
src/
├── scbe_14layer_reference.py   # Core 14-layer pipeline
├── crypto/
│   ├── rwp_v3.py               # RWP encryption protocol
│   ├── sacred_tongues.py       # Semantic tokenizer
│   └── pqc_liboqs.py           # Post-quantum crypto
├── api/
│   └── main.py                 # FastAPI server
└── harmonic/                   # TypeScript harmonic modules

tests/
├── spiralverse/                # RWP envelope tests
├── enterprise/                 # Compliance tests
└── industry_standard/          # Research-grade tests
```

---

## Mathematical Foundation

### Poincaré Distance
$$d(u,v) = \text{arccosh}\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$$

### Harmonic Scaling
$$H(d,R) = R^{d^2}$$

### Risk Gate
$$\text{Risk}' = \text{Risk}_{\text{base}} \times H(d)$$

---

## License

Apache 2.0 - Free for commercial use

Patent Pending: USPTO #63/961,403
