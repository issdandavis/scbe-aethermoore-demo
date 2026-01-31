"""
Cryptographic Utilities for AetherAuth

Provides:
    - Key derivation (PBKDF2 with SHA-256)
    - AES-GCM encryption/decryption
    - Context-bound key derivation
    - Fail-to-Noise fallback

Security Note: In production, consider using liboqs-python
for post-quantum key encapsulation (ML-KEM/Kyber).
"""

import os
import hashlib
import hmac
import struct
from typing import Tuple, Optional
from dataclasses import dataclass

# Try to import cryptography library
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# Constants
SALT_SIZE = 16
NONCE_SIZE = 12
TAG_SIZE = 16
KEY_SIZE = 32  # AES-256
ITERATIONS = 100_000


@dataclass
class EncryptedBlob:
    """Container for encrypted data with metadata."""
    ciphertext: bytes
    nonce: bytes
    salt: bytes
    tag: bytes  # Extracted from ciphertext in GCM mode

    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage."""
        # Format: salt (16) | nonce (12) | ciphertext+tag
        return self.salt + self.nonce + self.ciphertext

    @classmethod
    def from_bytes(cls, data: bytes) -> 'EncryptedBlob':
        """Deserialize from bytes."""
        if len(data) < SALT_SIZE + NONCE_SIZE + TAG_SIZE:
            raise ValueError("Data too short for EncryptedBlob")

        salt = data[:SALT_SIZE]
        nonce = data[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
        ciphertext = data[SALT_SIZE + NONCE_SIZE:]

        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            tag=b''  # Tag is embedded in ciphertext for GCM
        )


def derive_key(
    passphrase: str,
    salt: bytes,
    iterations: int = ITERATIONS
) -> bytes:
    """
    Derive a 256-bit key from passphrase using PBKDF2.

    Args:
        passphrase: User-provided secret
        salt: Random salt (16 bytes recommended)
        iterations: PBKDF2 iterations (default 100,000)

    Returns:
        32-byte key suitable for AES-256
    """
    if CRYPTO_AVAILABLE:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=salt,
            iterations=iterations,
        )
        return kdf.derive(passphrase.encode())
    else:
        # Fallback using hashlib
        return hashlib.pbkdf2_hmac(
            'sha256',
            passphrase.encode(),
            salt,
            iterations,
            dklen=KEY_SIZE
        )


def derive_context_key(
    master_key: bytes,
    context_vector: list,
    salt: bytes
) -> bytes:
    """
    Derive a context-bound key.

    The key changes based on the context vector, making
    stolen ciphertext useless without the correct context.

    Args:
        master_key: The master encryption key
        context_vector: 6D context vector [0,1]^6
        salt: Additional salt

    Returns:
        32-byte context-bound key
    """
    # Serialize context vector
    context_bytes = struct.pack('6f', *context_vector)

    # Combine master key with context
    combined = master_key + context_bytes + salt

    # Derive new key
    return hashlib.sha256(combined).digest()


def aes_gcm_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt using AES-256-GCM.

    Args:
        plaintext: Data to encrypt
        key: 32-byte encryption key

    Returns:
        salt (16) | nonce (12) | ciphertext + tag
    """
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)

    if CRYPTO_AVAILABLE:
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    else:
        # Fallback: XOR-based (NOT SECURE - demo only)
        # In production, require the cryptography library
        ciphertext = _xor_fallback(plaintext, key, nonce)

    return salt + nonce + ciphertext


def aes_gcm_decrypt(ciphertext_blob: bytes, key: bytes) -> bytes:
    """
    Decrypt AES-256-GCM ciphertext.

    Args:
        ciphertext_blob: salt (16) | nonce (12) | ciphertext + tag
        key: 32-byte decryption key

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If decryption fails (wrong key or tampering)
    """
    if len(ciphertext_blob) < SALT_SIZE + NONCE_SIZE + TAG_SIZE:
        raise ValueError("Ciphertext too short")

    salt = ciphertext_blob[:SALT_SIZE]
    nonce = ciphertext_blob[SALT_SIZE:SALT_SIZE + NONCE_SIZE]
    ciphertext = ciphertext_blob[SALT_SIZE + NONCE_SIZE:]

    if CRYPTO_AVAILABLE:
        aesgcm = AESGCM(key)
        try:
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    else:
        # Fallback
        return _xor_fallback(ciphertext, key, nonce)


def _xor_fallback(data: bytes, key: bytes, nonce: bytes) -> bytes:
    """
    Simple XOR cipher fallback (NOT SECURE - demo only).

    WARNING: This provides NO real security. It exists only
    to allow the demo to run without the cryptography library.
    In production, always use proper AES-GCM.
    """
    # Generate keystream from key + nonce
    keystream_seed = key + nonce
    keystream = b''
    counter = 0

    while len(keystream) < len(data):
        block = hashlib.sha256(keystream_seed + counter.to_bytes(4, 'big')).digest()
        keystream += block
        counter += 1

    keystream = keystream[:len(data)]

    return bytes(a ^ b for a, b in zip(data, keystream))


def fail_to_noise(length: int = 64) -> bytes:
    """
    Generate random noise for fail-to-noise fallback.

    When decryption fails (wrong context, tampering), return
    garbage instead of an error. This prevents timing attacks
    and doesn't reveal whether decryption succeeded.
    """
    return os.urandom(length)


def secure_compare(a: bytes, b: bytes) -> bool:
    """
    Constant-time comparison to prevent timing attacks.
    """
    return hmac.compare_digest(a, b)


def generate_envelope_id() -> str:
    """Generate a unique envelope ID."""
    return os.urandom(8).hex()


if __name__ == "__main__":
    # Demo
    print("AetherAuth Crypto Demo")
    print("=" * 50)
    print(f"Cryptography library available: {CRYPTO_AVAILABLE}")

    # Test key derivation
    passphrase = "my-secure-passphrase"
    salt = os.urandom(SALT_SIZE)

    key = derive_key(passphrase, salt)
    print(f"\nDerived key (hex): {key.hex()[:32]}...")

    # Test encryption/decryption
    message = b"secret_api_key_12345"
    print(f"\nOriginal: {message}")

    encrypted = aes_gcm_encrypt(message, key)
    print(f"Encrypted ({len(encrypted)} bytes): {encrypted.hex()[:64]}...")

    decrypted = aes_gcm_decrypt(encrypted, key)
    print(f"Decrypted: {decrypted}")

    # Test context-bound key
    context = [0.5, 0.5, 0.3, 0.4, 0.5, 0.8]
    context_key = derive_context_key(key, context, salt)
    print(f"\nContext-bound key: {context_key.hex()[:32]}...")

    # Different context = different key
    bad_context = [0.9, 0.9, 0.9, 0.9, 0.1, 0.0]
    bad_key = derive_context_key(key, bad_context, salt)
    print(f"Bad context key:   {bad_key.hex()[:32]}...")

    print(f"\nKeys match: {secure_compare(context_key, bad_key)}")
