#!/usr/bin/env python3
"""
Sacred Eggs - Cryptographically Sealed Token Containers
========================================================
A Sacred Egg is a GeoSeal-encrypted container holding a payload encoded in
one or more Sacred Tongues. The "shell" (metadata) is visible but tamper-evident,
while the "yolk" (payload) remains encrypted until hatching conditions are met.

Ritual modes:
- Solitary Whisperer: Simple hatch with single tongue match
- Triadic Round: Require 3 tongues in consensus (weight sum >= threshold)
- Ring Descent: Require inward ring progression

Last Updated: January 27, 2026
Version: 1.0.0 (RWP v3.0 compatible)
"""

import base64
import dataclasses
import hashlib
import hmac
import json
import math
import random
import time
from typing import Dict, List, Tuple, Optional, Iterable

from .sacred_tongues import (
    SacredTongueTokenizer,
    TONGUES,
    TongueSpec,
    SECTION_TONGUES,
)


# ---------- Cross-tokenization ----------

@dataclasses.dataclass
class XlateAttestation:
    """Attestation record for cross-tongue translation."""
    src: str
    dst: str
    mode: str
    ts: float
    phase_delta: float
    weight_ratio: float
    sha256_bytes: str
    hmac_attest: str


class CrossTokenizer:
    """
    Cross-tongue tokenizer for translating between Sacred Tongues.

    Supports:
    - Byte-level retokenization (encode in one tongue, decode in another)
    - Blended encoding (cycle through multiple tongues)
    - Phase and weight attestation for ritual verification
    """

    # Phase angles (radians) - 60-degree intervals around the circle
    PHASE = {
        'ko': 0,
        'av': math.pi / 3,
        'ru': 2 * math.pi / 3,
        'ca': math.pi,
        'um': 4 * math.pi / 3,
        'dr': 5 * math.pi / 3,
    }

    # Golden ratio (phi) weights for ritual consensus
    WEIGHT = {
        'ko': 1.000,   # Base
        'av': 1.618,   # phi^1
        'ru': 2.618,   # phi^2
        'ca': 4.236,   # phi^3
        'um': 6.854,   # phi^4
        'dr': 11.090,  # phi^5
    }

    def __init__(self, tokenizer: SacredTongueTokenizer = None):
        """
        Initialize with a SacredTongueTokenizer instance.

        Args:
            tokenizer: SacredTongueTokenizer instance. If None, creates a new one.
        """
        self.tok = tokenizer or SacredTongueTokenizer(TONGUES)

    def to_bytes_from_tokens(self, tongue: str, token_text: str) -> bytes:
        """Convert space-separated tokens back to raw bytes."""
        tokens = self.normalize_token_stream(token_text)
        return self.tok.decode_tokens(tongue, tokens)

    def to_tokens_from_bytes(self, tongue: str, data: bytes) -> List[str]:
        """Convert raw bytes to tokens in specified tongue."""
        return self.tok.encode_bytes(tongue, data)

    def normalize_token_stream(self, text: str) -> List[str]:
        """Parse space/comma-separated token string into list."""
        tokens = []
        for part in text.replace(",", " ").split():
            part = part.strip()
            if part:
                tokens.append(part)
        return tokens

    def retokenize(
        self,
        src_tongue: str,
        dst_tongue: str,
        token_text: str,
        mode: str = "byte",
        attest_key: bytes = None
    ) -> Tuple[List[str], XlateAttestation]:
        """
        Cross-tokenize from one Sacred Tongue to another.

        Args:
            src_tongue: Source tongue code (ko, av, ru, ca, um, dr)
            dst_tongue: Destination tongue code
            token_text: Space-separated tokens in source tongue
            mode: 'byte' (default) or 'semantic'
            attest_key: Optional key for HMAC attestation

        Returns:
            Tuple of (output_tokens, attestation)
        """
        if mode not in ("byte", "semantic"):
            raise ValueError("mode must be 'byte' or 'semantic'")

        # Decode from source tongue to bytes
        raw_bytes = self.to_bytes_from_tokens(src_tongue, token_text)

        # Encode to destination tongue
        out_tokens = self.to_tokens_from_bytes(dst_tongue, raw_bytes)

        # Build attestation
        sha = hashlib.sha256(raw_bytes).hexdigest()
        phase_delta = (self.PHASE[dst_tongue] - self.PHASE[src_tongue]) % (2 * math.pi)
        weight_ratio = self.WEIGHT[dst_tongue] / self.WEIGHT[src_tongue]

        msg = f"{src_tongue}->{dst_tongue}|{mode}|{sha}|{phase_delta:.6f}|{weight_ratio:.6f}|{int(time.time())}".encode()
        key = attest_key or b"aether-attest-default"
        h = base64.b64encode(hmac.new(key, msg, hashlib.sha256).digest()).decode()

        attest = XlateAttestation(
            src=src_tongue,
            dst=dst_tongue,
            mode=mode,
            ts=time.time(),
            phase_delta=phase_delta,
            weight_ratio=weight_ratio,
            sha256_bytes=sha,
            hmac_attest=h
        )

        return out_tokens, attest

    def blend(self, pattern: List[str], data: bytes) -> List[Tuple[str, str]]:
        """
        Encode bytes using a rotating pattern of tongues.

        Args:
            pattern: List of tongue codes to cycle through
            data: Raw bytes to encode

        Returns:
            List of (tongue_code, token) tuples
        """
        out: List[Tuple[str, str]] = []
        for i, byte in enumerate(data):
            tongue = pattern[i % len(pattern)]
            # Encode single byte and extract the token
            tokens = self.tok.encode_bytes(tongue, bytes([byte]))
            out.append((tongue, tokens[0]))
        return out

    def unblend(self, pattern: List[str], pairs: List[Tuple[str, str]]) -> bytes:
        """
        Decode blended (tongue, token) pairs back to bytes.

        Args:
            pattern: Original pattern used for blending
            pairs: List of (tongue_code, token) tuples

        Returns:
            Decoded bytes
        """
        arr = bytearray()
        for i, (tongue, token) in enumerate(pairs):
            expected = pattern[i % len(pattern)]
            if tongue != expected:
                raise ValueError(f"Blend pattern mismatch at index {i}: expected {expected}, got {tongue}")
            # Decode single token to byte
            decoded = self.tok.decode_tokens(tongue, [token])
            arr.append(decoded[0])
        return bytes(arr)


# ---------- GeoSeal Geometry Functions ----------

def _zscore(xs: List[float]) -> List[float]:
    """Compute z-score normalized values."""
    if not xs:
        return []
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
    sd = math.sqrt(var) if var > 0 else 1.0
    return [(x - mu) / sd for x in xs]


def project_to_sphere(ctx: List[float]) -> List[float]:
    """Project context vector to unit sphere (3D)."""
    take = (ctx[:3] if len(ctx) >= 3 else (ctx + [0, 0, 0])[:3])
    z = _zscore(list(take))
    norm = math.sqrt(sum(v * v for v in z)) or 1.0
    return [v / norm for v in z]


def project_to_cube(ctx: List[float], m: int = 6) -> List[float]:
    """Project context vector to unit hypercube (m dimensions)."""
    arr = [(math.tanh(x / 5) + 1) / 2 for x in (ctx[:m] if len(ctx) >= m else ctx + [0] * (m - len(ctx)))]
    return [min(1.0, max(0.0, x)) for x in arr]


def healpix_id(u: List[float], L: int) -> str:
    """Generate HEALPix-style cell ID from sphere coordinates."""
    q = tuple(int((v + 1) * 1000) for v in u)
    return f"S{L}:{q}"


def morton_id(v: List[float], L: int) -> str:
    """Generate Morton-code cell ID from cube coordinates."""
    q = tuple(int(x * (10 ** min(3, 1 + L))) for x in v[:min(6, len(v))])
    return f"C{L}:{q}"


def potentials(u: List[float], v: List[float]) -> Tuple[float, float]:
    """Compute potential P and margin values for classification."""
    R = sum(abs(x) for x in u) + 0.1 * sum(v)
    T = 0.5 + 0.05 * len([x for x in v if x < 0.2])
    P = 0.7 * R - 0.3 * T
    margin = 0.5 - abs(u[0])
    return P, margin


def classify(h: str, z: str, P: float, margin: float) -> str:
    """Classify geometric path as 'interior' or 'exterior'."""
    return "interior" if ("S" in h and "C" in z and P < 0.6 and margin > 0.05) else "exterior"


class ConcentricRingPolicy:
    """
    Policy for concentric ring-based security zones.

    Rings (from center outward):
    - core (0.0-0.3): Highest security, lowest latency
    - inner (0.3-0.5): High security
    - middle (0.5-0.7): Balanced
    - outer (0.7-0.9): Lower security
    - edge (0.9-1.0): Boundary zone
    """

    # (r_min, r_max, name, max_latency_ms, required_sigs, pow_bits, trust_decay)
    RINGS = [
        (0.0, 0.3, "core", 5, 1, 8, 0.001),
        (0.3, 0.5, "inner", 20, 1, 8, 0.005),
        (0.5, 0.7, "middle", 100, 2, 16, 0.01),
        (0.7, 0.9, "outer", 500, 3, 24, 0.05),
        (0.9, 1.0, "edge", 5000, 4, 32, 0.2),
    ]

    def classify(self, r: float) -> dict:
        """Classify radial distance to ring parameters."""
        for rmin, rmax, name, lat, sigs, powb, decay in self.RINGS:
            if rmin <= r < rmax:
                return {
                    "ring": name,
                    "max_latency_ms": lat,
                    "required_signatures": sigs,
                    "pow_bits": powb,
                    "trust_decay_rate": decay
                }
        return {"ring": "beyond", "action": "REJECT"}


# ---------- Demo/Mock PQC Primitives ----------

def hkdf(key: bytes, info: str) -> bytes:
    """Simple HKDF-like key derivation using HMAC-SHA256."""
    return hmac.new(key, info.encode(), hashlib.sha256).digest()


def kyber_encaps(pk: bytes) -> Tuple[bytes, bytes]:
    """Mock Kyber encapsulation (returns shared secret and ciphertext)."""
    ss = hashlib.sha256(b"ss" + pk).digest()
    ct = hashlib.sha256(b"ct" + pk).digest()
    return ss, ct


def kyber_decaps(sk: bytes, ct: bytes) -> bytes:
    """Mock Kyber decapsulation."""
    return hashlib.sha256(b"ss" + sk).digest()


def dsa_sign(sk: bytes, msg: bytes) -> bytes:
    """Mock DSA signature using HMAC."""
    return hmac.new(sk, msg, hashlib.sha256).digest()


def dsa_verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
    """Mock DSA verification."""
    expected = hmac.new(pk, msg, hashlib.sha256).digest()
    return hmac.compare_digest(expected, sig)


# ---------- GeoSeal Encrypt/Decrypt ----------

def geoseal_encrypt(
    plaintext_b64: str,
    context: List[float],
    pk_kem_b64: str,
    sk_dsa_b64: str,
    Ls: int = 2,
    Lc: int = 2
) -> dict:
    """
    GeoSeal encryption: Encrypt plaintext with geometric attestation.

    Args:
        plaintext_b64: Base64-encoded plaintext
        context: Geometric context vector (list of floats)
        pk_kem_b64: Base64-encoded KEM public key
        sk_dsa_b64: Base64-encoded DSA signing key
        Ls: Sphere resolution level
        Lc: Cube resolution level

    Returns:
        Dictionary with ct_k, ct_spec, attest, sig
    """
    pt = base64.b64decode(plaintext_b64) if isinstance(plaintext_b64, str) else plaintext_b64

    # Geometric projection
    u = project_to_sphere(context)
    v = project_to_cube(context)
    h = healpix_id(u, Ls)
    z = morton_id(v, Lc)
    P, margin = potentials(u, v)
    path = classify(h, z, P, margin)

    # Key encapsulation
    ss, ct_k = kyber_encaps(base64.b64decode(pk_kem_b64))

    # Derive message key
    Ks = hkdf(ss, f"geo:sphere|{h}|{Ls}")
    Kc = hkdf(ss, f"geo:cube|{z}|{Lc}")
    Kmsg = hkdf(bytes(x ^ y for x, y in zip(Ks, Kc)), "geo:msg")

    # XOR encryption with derived mask
    mask_seed = hashlib.sha256(Kmsg).digest()
    mask = (mask_seed * ((len(pt) // len(mask_seed)) + 2))[:len(pt)]
    ct_spec = bytes(a ^ b for a, b in zip(pt, mask))

    # Attestation
    attest = {
        "h": h,
        "z": z,
        "L_s": Ls,
        "L_c": Lc,
        "P": round(P, 6),
        "margin": round(margin, 6),
        "ts": int(time.time()),
        "path": path
    }

    # Sign attestation + ciphertext
    sig_data = hashlib.sha256(json.dumps(attest, sort_keys=True).encode() + ct_spec).digest()
    sig = dsa_sign(base64.b64decode(sk_dsa_b64), sig_data)

    return {
        "ct_k": base64.b64encode(ct_k).decode(),
        "ct_spec": base64.b64encode(ct_spec).decode(),
        "attest": attest,
        "sig": base64.b64encode(sig).decode()
    }


def geoseal_decrypt(
    env: dict,
    context: List[float],
    sk_kem_b64: str,
    pk_dsa_b64: str
) -> Tuple[bool, Optional[bytes]]:
    """
    GeoSeal decryption: Decrypt with signature verification.

    Args:
        env: GeoSeal envelope dictionary
        context: Geometric context vector
        sk_kem_b64: Base64-encoded KEM secret key
        pk_dsa_b64: Base64-encoded DSA verification key

    Returns:
        Tuple of (success, plaintext_bytes or None)
    """
    ct_k = base64.b64decode(env["ct_k"]) if isinstance(env["ct_k"], str) else env["ct_k"]
    ct_spec = base64.b64decode(env["ct_spec"]) if isinstance(env["ct_spec"], str) else env["ct_spec"]
    attest = env["attest"]
    sig = base64.b64decode(env["sig"]) if isinstance(env["sig"], str) else env["sig"]

    # Verify signature
    sig_data = hashlib.sha256(json.dumps(attest, sort_keys=True).encode() + ct_spec).digest()
    if not dsa_verify(base64.b64decode(pk_dsa_b64), sig_data, sig):
        return False, None

    # Decapsulate key
    ss = kyber_decaps(base64.b64decode(sk_kem_b64), ct_k)

    # Derive message key using attested geometry
    Ks = hkdf(ss, f"geo:sphere|{attest['h']}|{attest['L_s']}")
    Kc = hkdf(ss, f"geo:cube|{attest['z']}|{attest['L_c']}")
    Kmsg = hkdf(bytes(x ^ y for x, y in zip(Ks, Kc)), "geo:msg")

    # Decrypt
    mask_seed = hashlib.sha256(Kmsg).digest()
    mask = (mask_seed * ((len(ct_spec) // len(mask_seed)) + 2))[:len(ct_spec)]
    pt = bytes(a ^ b for a, b in zip(ct_spec, mask))

    return True, pt


# ---------- Sacred Eggs ----------

@dataclasses.dataclass
class SacredEgg:
    """
    A Sacred Egg: Cryptographically sealed token container.

    Attributes:
        egg_id: Unique identifier (derived from yolk hash)
        primary_tongue: The tongue used to encode the payload
        glyph: Visual/symbolic identifier for the egg
        hatch_condition: Dict specifying when hatching is allowed
        yolk_ct: GeoSeal envelope containing encrypted payload
    """
    egg_id: str
    primary_tongue: str
    glyph: str
    hatch_condition: dict  # e.g., {"ring": "inner", "path": "interior", "min_weight": 5.0}
    yolk_ct: dict          # GeoSeal envelope


@dataclasses.dataclass
class HatchResult:
    """
    Result of attempting to hatch a Sacred Egg.

    Attributes:
        success: Whether hatching succeeded
        tokens: Decoded tokens (if successful)
        attestation: GeoSeal attestation (if successful)
        reason: Human-readable explanation
    """
    success: bool
    tokens: Optional[List[str]]
    attestation: Optional[dict]
    reason: str


class SacredEggIntegrator:
    """
    Integrator for creating and hatching Sacred Eggs.

    Supports ritual modes:
    - solitary: Single tongue match required
    - triadic: Three tongues must reach consensus weight
    - ring_descent: Agent must have descended through rings
    """

    def __init__(self, cross_tokenizer: CrossTokenizer = None):
        """
        Initialize with a CrossTokenizer.

        Args:
            cross_tokenizer: CrossTokenizer instance. If None, creates a new one.
        """
        self.xt = cross_tokenizer or CrossTokenizer()
        self.ring_policy = ConcentricRingPolicy()

    def create_egg(
        self,
        payload: bytes,
        primary_tongue: str,
        glyph: str,
        hatch_condition: dict,
        context: List[float],
        pk_kem_b64: str,
        sk_dsa_b64: str
    ) -> SacredEgg:
        """
        Create a new Sacred Egg.

        Args:
            payload: Raw bytes to seal in the egg
            primary_tongue: Tongue code for encoding (ko, av, ru, ca, um, dr)
            glyph: Visual identifier for the egg
            hatch_condition: Dict with hatching requirements
            context: Geometric context for GeoSeal
            pk_kem_b64: Base64 KEM public key
            sk_dsa_b64: Base64 DSA signing key

        Returns:
            SacredEgg instance
        """
        pt_b64 = base64.b64encode(payload).decode()
        env = geoseal_encrypt(pt_b64, context, pk_kem_b64, sk_dsa_b64)
        egg_id = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()[:16]

        return SacredEgg(
            egg_id=egg_id,
            primary_tongue=primary_tongue,
            glyph=glyph,
            hatch_condition=hatch_condition,
            yolk_ct=env
        )

    def hatch_egg(
        self,
        egg: SacredEgg | dict,
        current_context: List[float],
        agent_tongue: str,
        sk_kem_b64: str,
        pk_dsa_b64: str,
        ritual_mode: str = "solitary",
        path_history: List[dict] = None,
        triad_tongues: List[str] = None
    ) -> HatchResult:
        """
        Attempt to hatch a Sacred Egg.

        Args:
            egg: SacredEgg instance or dict representation
            current_context: Agent's current geometric context
            agent_tongue: Agent's preferred tongue
            sk_kem_b64: Base64 KEM secret key
            pk_dsa_b64: Base64 DSA verification key
            ritual_mode: 'solitary', 'triadic', or 'ring_descent'
            path_history: For ring_descent, list of previous positions
            triad_tongues: For triadic, the three tongues in consensus

        Returns:
            HatchResult with success status and tokens/reason
        """
        # Convert dict to SacredEgg if needed
        if isinstance(egg, dict):
            egg = SacredEgg(**egg)

        # Compute current geometric state
        u = project_to_sphere(current_context)
        v = project_to_cube(current_context)
        h = healpix_id(u, 2)
        z = morton_id(v, 2)
        P, margin = potentials(u, v)
        path = classify(h, z, P, margin)

        # Use deterministic radial distance from context
        r = min(1.0, max(0.0, abs(current_context[0]) / 10.0)) if current_context else 0.5
        current_ring = self.ring_policy.classify(r)

        # Check base geometric condition
        required_path = egg.hatch_condition.get("path", "interior")
        required_ring = egg.hatch_condition.get("ring", "inner")

        if path != required_path:
            return HatchResult(
                success=False,
                tokens=None,
                attestation=None,
                reason=f"Geometric misalignment - path is {path}, requires {required_path}."
            )

        if current_ring["ring"] != required_ring:
            return HatchResult(
                success=False,
                tokens=None,
                attestation=None,
                reason=f"Ring misalignment - at {current_ring['ring']}, requires {required_ring}."
            )

        # Ritual-specific checks
        if ritual_mode == "solitary":
            if agent_tongue != egg.primary_tongue:
                return HatchResult(
                    success=False,
                    tokens=None,
                    attestation=None,
                    reason="Tongue mismatch - the egg whispers only to its own."
                )

        elif ritual_mode == "triadic":
            tongues = triad_tongues or [egg.primary_tongue, "ru", "um"]
            triad_weights = [self.xt.WEIGHT.get(t, 0) for t in tongues]
            min_weight = egg.hatch_condition.get("min_weight", 10.0)

            if sum(triad_weights) < min_weight:
                return HatchResult(
                    success=False,
                    tokens=None,
                    attestation=None,
                    reason=f"Insufficient consensus - triad weight {sum(triad_weights):.2f} < {min_weight}."
                )

        elif ritual_mode == "ring_descent":
            history = path_history or []
            ring_order = {"core": 0, "inner": 1, "middle": 2, "outer": 3, "edge": 4}

            # Check for inward descent (decreasing ring order)
            if len(history) >= 2:
                valid_descent = True
                for i in range(len(history) - 1):
                    curr_ring = history[i].get("ring", "edge")
                    next_ring = history[i + 1].get("ring", "edge")
                    if ring_order.get(curr_ring, 4) <= ring_order.get(next_ring, 4):
                        valid_descent = False
                        break

                if not valid_descent:
                    return HatchResult(
                        success=False,
                        tokens=None,
                        attestation=None,
                        reason="The descent wavered - the path is not true."
                    )

        # Decrypt the yolk
        ok, yolk_bytes = geoseal_decrypt(egg.yolk_ct, current_context, sk_kem_b64, pk_dsa_b64)

        if not ok:
            return HatchResult(
                success=False,
                tokens=None,
                attestation=None,
                reason="Cryptographic seal unbroken - the yolk turns to noise."
            )

        # Decode in primary tongue
        tokens = self.xt.tok.encode_bytes(egg.primary_tongue, yolk_bytes)
        attest = dict(egg.yolk_ct["attest"])

        # Cross-tokenize to agent tongue if different
        if agent_tongue != egg.primary_tongue:
            token_text = " ".join(tokens)
            tokens, xlate_attest = self.xt.retokenize(
                egg.primary_tongue,
                agent_tongue,
                token_text
            )
            attest["xlate"] = dataclasses.asdict(xlate_attest)

        return HatchResult(
            success=True,
            tokens=tokens,
            attestation=attest,
            reason=f"The egg hatches - revelation in {agent_tongue}."
        )


# ---------- Selftest ----------

def selftest() -> int:
    """
    Run self-tests for Sacred Eggs module.

    Returns:
        0 on success, 1 on failure
    """
    print("Running Sacred Eggs selftest...")

    # Test CrossTokenizer
    xt = CrossTokenizer()
    test_data = b"sacred secret"

    # Encode/decode round-trip
    tokens = xt.to_tokens_from_bytes('ko', test_data)
    decoded = xt.to_bytes_from_tokens('ko', " ".join(tokens))
    assert decoded == test_data, f"Round-trip failed: {decoded} != {test_data}"
    print("  [PASS] CrossTokenizer round-trip")

    # Cross-tokenization
    ko_tokens = xt.to_tokens_from_bytes('ko', test_data)
    ru_tokens, attest = xt.retokenize('ko', 'ru', " ".join(ko_tokens))
    decoded_ru = xt.to_bytes_from_tokens('ru', " ".join(ru_tokens))
    assert decoded_ru == test_data, "Cross-tokenization failed"
    assert attest.src == 'ko' and attest.dst == 'ru', "Attestation src/dst mismatch"
    print("  [PASS] CrossTokenizer retokenize")

    # Blend/unblend
    pattern = ['ko', 'ru', 'av']
    blended = xt.blend(pattern, test_data)
    unblended = xt.unblend(pattern, blended)
    assert unblended == test_data, "Blend/unblend failed"
    print("  [PASS] CrossTokenizer blend/unblend")

    # GeoSeal round-trip
    ctx = [0.2, -0.3, 0.7, 1.0, -2.0, 0.5, 3.1, -9.9, 0.0]
    kem_key = base64.b64encode(b"kem-key-32bytes-demo____").decode()
    dsa_key = base64.b64encode(b"dsa-key-32bytes-demo____").decode()

    pt_b64 = base64.b64encode(test_data).decode()
    env = geoseal_encrypt(pt_b64, ctx, kem_key, dsa_key)
    ok, decrypted = geoseal_decrypt(env, ctx, kem_key, dsa_key)
    assert ok and decrypted == test_data, "GeoSeal round-trip failed"
    print("  [PASS] GeoSeal encrypt/decrypt")

    # Sacred Egg creation and hatching
    # Note: Most contexts produce "exterior" path; ring is determined by abs(ctx[0])/10
    # With ctx[0]=0.2, r=0.02 -> "core" ring
    sei = SacredEggIntegrator(xt)
    payload = b"sacred secret"
    cond = {"ring": "core", "path": "exterior"}

    egg = sei.create_egg(payload, "ko", "\u25c7", cond, ctx, kem_key, dsa_key)
    assert egg.egg_id and len(egg.egg_id) == 16, "Egg ID generation failed"
    print("  [PASS] SacredEgg creation")

    # Successful hatch (same tongue)
    result = sei.hatch_egg(
        dataclasses.asdict(egg), ctx, "ko", kem_key, dsa_key, "solitary"
    )
    assert result.success, f"Hatch failed: {result.reason}"

    # Verify decoded tokens match original
    decoded_bytes = xt.to_bytes_from_tokens('ko', " ".join(result.tokens))
    assert decoded_bytes == payload, f"Payload mismatch: {decoded_bytes} != {payload}"
    print("  [PASS] SacredEgg hatch (solitary)")

    # Failed hatch (wrong tongue)
    result = sei.hatch_egg(
        dataclasses.asdict(egg), ctx, "dr", kem_key, dsa_key, "solitary"
    )
    assert not result.success, "Should have failed with wrong tongue"
    assert "tongue" in result.reason.lower(), f"Wrong error: {result.reason}"
    print("  [PASS] SacredEgg hatch rejection (wrong tongue)")

    # Triadic ritual
    egg_triadic = sei.create_egg(
        payload, "ko", "\u25c7",
        {"ring": "core", "path": "exterior", "min_weight": 5.0},
        ctx, kem_key, dsa_key
    )
    result = sei.hatch_egg(
        dataclasses.asdict(egg_triadic), ctx, "ko", kem_key, dsa_key,
        "triadic", triad_tongues=["ko", "ru", "um"]
    )
    assert result.success, f"Triadic hatch failed: {result.reason}"
    print("  [PASS] SacredEgg hatch (triadic)")

    # Ring descent
    result = sei.hatch_egg(
        dataclasses.asdict(egg), ctx, "ko", kem_key, dsa_key,
        "ring_descent",
        path_history=[{"ring": "outer"}, {"ring": "middle"}, {"ring": "inner"}]
    )
    assert result.success, f"Ring descent hatch failed: {result.reason}"
    print("  [PASS] SacredEgg hatch (ring_descent)")

    print("selftest ok (with eggs)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
