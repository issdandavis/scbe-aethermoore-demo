"""
Sacred Tongue Tokenizer (SS1 Protocol)

Encodes/decodes data using the Six Sacred Tongues:
    - KO (Kor'aelin): Control - used for envelope headers
    - AV (Avali): Transport - used for network data
    - RU (Runethic): Policy - used for salts/IVs
    - CA (Cassisivadan): Compute - used for ciphertext
    - UM (Umbroth): Security - used for signatures
    - DR (Draumric): Schema - used for metadata

Each tongue has a unique encoding pattern based on
phi^n weights and phase offsets.
"""

import base64
import hashlib
import struct
from typing import Optional, Dict, Tuple
from enum import Enum
import math


# Golden ratio
PHI = (1 + math.sqrt(5)) / 2


class Tongue(Enum):
    """The Six Sacred Tongues."""
    KO = ("Kor'aelin", "Control", 0, PHI ** 0)      # 1.000
    AV = ("Avali", "Transport", 60, PHI ** 1)       # 1.618
    RU = ("Runethic", "Policy", 120, PHI ** 2)      # 2.618
    CA = ("Cassisivadan", "Compute", 180, PHI ** 3) # 4.236
    UM = ("Umbroth", "Security", 240, PHI ** 4)     # 6.854
    DR = ("Draumric", "Schema", 300, PHI ** 5)      # 11.090

    def __init__(self, full_name: str, role: str, phase_deg: int, weight: float):
        self.full_name = full_name
        self.role = role
        self.phase_deg = phase_deg
        self.phase_rad = math.radians(phase_deg)
        self.weight = weight


class SacredTongueTokenizer:
    """
    SS1 Tokenizer for encoding/decoding with Sacred Tongues.

    Encoding format: {tongue}:{base64_data}:{checksum}

    The checksum incorporates the tongue's phase, providing
    an additional layer of validation.
    """

    # Tongue prefixes for encoded strings
    TONGUE_PREFIXES: Dict[str, str] = {
        'KO': 'ko',
        'AV': 'av',
        'RU': 'ru',
        'CA': 'ca',
        'UM': 'um',
        'DR': 'dr',
    }

    def __init__(self):
        self.tongues = {t.name: t for t in Tongue}

    def encode(self, data: bytes, tongue: str = 'CA') -> str:
        """
        Encode bytes using specified Sacred Tongue.

        Args:
            data: Raw bytes to encode
            tongue: Tongue name (KO, AV, RU, CA, UM, DR)

        Returns:
            Encoded string: {prefix}:{base64}:{checksum}
        """
        if tongue not in self.tongues:
            raise ValueError(f"Unknown tongue: {tongue}")

        t = self.tongues[tongue]
        prefix = self.TONGUE_PREFIXES[tongue]

        # Base64 encode
        b64 = base64.urlsafe_b64encode(data).decode('ascii')

        # Generate tongue-specific checksum
        checksum = self._tongue_checksum(data, t)

        return f"{prefix}:{b64}:{checksum}"

    def decode(self, encoded: str, tongue: Optional[str] = None) -> bytes:
        """
        Decode a Sacred Tongue encoded string.

        Args:
            encoded: Encoded string {prefix}:{base64}:{checksum}
            tongue: Expected tongue (optional, inferred from prefix)

        Returns:
            Decoded bytes

        Raises:
            ValueError: If checksum fails or format is invalid
        """
        parts = encoded.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid SS1 format: expected 3 parts, got {len(parts)}")

        prefix, b64_data, checksum = parts

        # Infer tongue from prefix
        tongue_name = None
        for name, pfx in self.TONGUE_PREFIXES.items():
            if pfx == prefix:
                tongue_name = name
                break

        if tongue_name is None:
            raise ValueError(f"Unknown prefix: {prefix}")

        if tongue and tongue != tongue_name:
            raise ValueError(f"Tongue mismatch: expected {tongue}, got {tongue_name}")

        t = self.tongues[tongue_name]

        # Decode base64
        try:
            data = base64.urlsafe_b64decode(b64_data)
        except Exception as e:
            raise ValueError(f"Base64 decode failed: {e}")

        # Verify checksum
        expected_checksum = self._tongue_checksum(data, t)
        if checksum != expected_checksum:
            raise ValueError("Checksum verification failed")

        return data

    def _tongue_checksum(self, data: bytes, tongue: Tongue) -> str:
        """
        Generate tongue-specific checksum.

        Incorporates the tongue's phase and weight into the hash.
        """
        # Combine data with tongue parameters
        phase_bytes = struct.pack('f', tongue.phase_rad)
        weight_bytes = struct.pack('f', tongue.weight)

        combined = data + phase_bytes + weight_bytes

        # SHA-256 and take first 8 chars
        digest = hashlib.sha256(combined).hexdigest()[:8]

        return digest

    def validate(self, encoded: str) -> Tuple[bool, str]:
        """
        Validate an encoded string without decoding.

        Returns:
            (is_valid, message)
        """
        try:
            self.decode(encoded)
            return True, "Valid"
        except ValueError as e:
            return False, str(e)

    def get_tongue_info(self, tongue: str) -> Dict:
        """Get information about a tongue."""
        if tongue not in self.tongues:
            raise ValueError(f"Unknown tongue: {tongue}")

        t = self.tongues[tongue]
        return {
            'name': t.name,
            'full_name': t.full_name,
            'role': t.role,
            'phase_deg': t.phase_deg,
            'weight': t.weight,
            'prefix': self.TONGUE_PREFIXES[tongue],
        }


class SS1Envelope:
    """
    SS1 Envelope container for encrypted data.

    Format: SS1|kid={key_id}|salt={ru_encoded}|ct={ca_encoded}|sig={um_encoded}

    Components:
        - kid: Key identifier
        - salt: Salt encoded with RU (Runethic)
        - ct: Ciphertext encoded with CA (Cassisivadan)
        - sig: Signature encoded with UM (Umbroth) - optional
    """

    def __init__(
        self,
        key_id: str,
        salt: bytes,
        ciphertext: bytes,
        signature: Optional[bytes] = None
    ):
        self.key_id = key_id
        self.salt = salt
        self.ciphertext = ciphertext
        self.signature = signature
        self.tokenizer = SacredTongueTokenizer()

    def serialize(self) -> str:
        """Serialize envelope to SS1 format string."""
        parts = [
            "SS1",
            f"kid={self.key_id}",
            f"salt={self.tokenizer.encode(self.salt, 'RU')}",
            f"ct={self.tokenizer.encode(self.ciphertext, 'CA')}",
        ]

        if self.signature:
            parts.append(f"sig={self.tokenizer.encode(self.signature, 'UM')}")

        return "|".join(parts)

    @classmethod
    def parse(cls, envelope_str: str) -> 'SS1Envelope':
        """Parse SS1 format string to envelope."""
        tokenizer = SacredTongueTokenizer()

        parts = envelope_str.split('|')
        if not parts or parts[0] != 'SS1':
            raise ValueError("Invalid SS1 envelope: must start with 'SS1'")

        # Parse key-value pairs
        kv = {}
        for part in parts[1:]:
            if '=' in part:
                key, value = part.split('=', 1)
                kv[key] = value

        # Required fields
        if 'kid' not in kv:
            raise ValueError("Missing key_id (kid)")
        if 'salt' not in kv:
            raise ValueError("Missing salt")
        if 'ct' not in kv:
            raise ValueError("Missing ciphertext (ct)")

        # Decode components
        salt = tokenizer.decode(kv['salt'], 'RU')
        ciphertext = tokenizer.decode(kv['ct'], 'CA')
        signature = tokenizer.decode(kv['sig'], 'UM') if 'sig' in kv else None

        return cls(
            key_id=kv['kid'],
            salt=salt,
            ciphertext=ciphertext,
            signature=signature
        )


if __name__ == "__main__":
    # Demo
    print("Sacred Tongue Tokenizer Demo")
    print("=" * 50)

    tokenizer = SacredTongueTokenizer()

    # Show tongue info
    print("\nSix Sacred Tongues:")
    for name in ['KO', 'AV', 'RU', 'CA', 'UM', 'DR']:
        info = tokenizer.get_tongue_info(name)
        print(f"  {name} ({info['full_name']}): weight={info['weight']:.3f}, role={info['role']}")

    # Test encoding/decoding
    test_data = b"secret_api_key_12345"
    print(f"\nOriginal: {test_data}")

    for tongue in ['RU', 'CA', 'UM']:
        encoded = tokenizer.encode(test_data, tongue)
        print(f"\n{tongue} encoded: {encoded}")

        decoded = tokenizer.decode(encoded)
        print(f"{tongue} decoded: {decoded}")

        is_valid, msg = tokenizer.validate(encoded)
        print(f"{tongue} valid: {is_valid}")

    # Test SS1 Envelope
    print("\n" + "=" * 50)
    print("SS1 Envelope Demo")

    import os
    salt = os.urandom(16)
    ciphertext = os.urandom(64)

    envelope = SS1Envelope(
        key_id="notion-pplx-v1",
        salt=salt,
        ciphertext=ciphertext
    )

    serialized = envelope.serialize()
    print(f"\nSerialized envelope ({len(serialized)} chars):")
    print(serialized[:100] + "...")

    # Parse back
    parsed = SS1Envelope.parse(serialized)
    print(f"\nParsed key_id: {parsed.key_id}")
    print(f"Salt matches: {parsed.salt == salt}")
    print(f"Ciphertext matches: {parsed.ciphertext == ciphertext}")
