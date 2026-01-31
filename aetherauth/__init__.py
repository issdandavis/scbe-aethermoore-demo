"""
AetherAuth - Hyperbolic OAuth Alternative

Context-aware API authentication using geometric envelopes.
Credentials only decrypt when the requesting system is in the
correct behavioral state (time, location, intent).

Components:
    - context_capture: 6D context vector generation
    - geoseal_gate: Trust ring validation (Poincare ball)
    - vault_access: Lumo Vault with SS1 envelope decryption
    - crypto: Post-quantum encryption utilities
    - sacred_tongues: SS1 tokenizer for encoding

Usage:
    from aetherauth import AetherAuthGate, LumoVault, capture_context_vector

    # Authenticate
    context = capture_context_vector()
    gate = AetherAuthGate()
    access = gate.check_access(context)

    if access['allowed']:
        vault = LumoVault()
        api_key = vault.get_key('notion', context, access)
"""

from .context_capture import capture_context_vector, ContextVector
from .geoseal_gate import AetherAuthGate, TrustRing
from .vault_access import LumoVault, SS1Envelope
from .crypto import derive_key, aes_gcm_encrypt, aes_gcm_decrypt
from .sacred_tongues import SacredTongueTokenizer

__version__ = "1.0.0"
__author__ = "SCBE-AETHERMOORE Team"

__all__ = [
    'capture_context_vector',
    'ContextVector',
    'AetherAuthGate',
    'TrustRing',
    'LumoVault',
    'SS1Envelope',
    'derive_key',
    'aes_gcm_encrypt',
    'aes_gcm_decrypt',
    'SacredTongueTokenizer',
]
