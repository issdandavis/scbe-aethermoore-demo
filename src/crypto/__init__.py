"""
SCBE-AETHERMOORE Cryptographic Module
=====================================
Post-quantum cryptography with NIST FIPS 203/204 compliance.

Exports:
- MLKEM768: ML-KEM-768 key encapsulation (FIPS 203)
- MLDSA65: ML-DSA-65 digital signatures (FIPS 204)
- RWPv3Protocol: Real World Protocol v3.0 encryption
- SacredTongueTokenizer: Six Sacred Tongues encoding
- SacredEgg: Cryptographically sealed token container
- CrossTokenizer: Cross-tongue translation
"""

from .pqc_liboqs import (
    MLKEM768,
    MLDSA65,
    MLKEMKeyPair,
    MLDSAKeyPair,
    is_liboqs_available,
    get_pqc_backend,
    create_dual_lattice_keys,
    compute_consensus_hash,
)

from .sacred_tongues import (
    SACRED_TONGUE_TOKENIZER,
    SECTION_TONGUES,
)

from .sacred_eggs import (
    SacredEgg,
    HatchResult,
    CrossTokenizer,
    SacredEggIntegrator,
    XlateAttestation,
    ConcentricRingPolicy,
    geoseal_encrypt,
    geoseal_decrypt,
)

__all__ = [
    # PQC primitives
    "MLKEM768",
    "MLDSA65",
    "MLKEMKeyPair",
    "MLDSAKeyPair",
    "is_liboqs_available",
    "get_pqc_backend",
    "create_dual_lattice_keys",
    "compute_consensus_hash",
    # Sacred Tongues
    "SACRED_TONGUE_TOKENIZER",
    "SECTION_TONGUES",
    # Sacred Eggs
    "SacredEgg",
    "HatchResult",
    "CrossTokenizer",
    "SacredEggIntegrator",
    "XlateAttestation",
    "ConcentricRingPolicy",
    "geoseal_encrypt",
    "geoseal_decrypt",
]
