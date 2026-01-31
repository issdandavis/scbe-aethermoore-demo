"""
Lumo Vault - Context-Bound Credential Storage

Stores API keys encrypted with AetherAuth's geometric validation.
Keys only decrypt when the requesting system is in the correct
behavioral state (time, location, intent).

Storage options:
    - Local encrypted file (.aether/vault/)
    - Environment variables
    - External vault (HashiCorp, AWS Secrets Manager)
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
import time

# Import from sibling modules
try:
    from .crypto import (
        derive_key, derive_context_key, aes_gcm_encrypt, aes_gcm_decrypt,
        fail_to_noise, SALT_SIZE
    )
    from .sacred_tongues import SacredTongueTokenizer, SS1Envelope
    from .context_capture import ContextVector
    from .geoseal_gate import AccessResult, TrustRing
except ImportError:
    from crypto import (
        derive_key, derive_context_key, aes_gcm_encrypt, aes_gcm_decrypt,
        fail_to_noise, SALT_SIZE
    )
    from sacred_tongues import SacredTongueTokenizer, SS1Envelope
    from context_capture import ContextVector
    from geoseal_gate import AccessResult, TrustRing


@dataclass
class VaultEntry:
    """A single entry in the vault."""
    key_id: str
    encrypted_value: bytes
    salt: bytes
    created_at: float
    expires_at: Optional[float] = None
    access_level: str = "full"  # full, read_only


class LumoVault:
    """
    Secure credential vault with context-bound encryption.

    Keys are encrypted with AES-GCM using a key derived from:
    1. Master passphrase (AETHER_MASTER_KEY env var)
    2. Context vector at decryption time
    3. Per-entry salt

    This means stolen encrypted keys are useless without
    both the master key AND the correct runtime context.
    """

    DEFAULT_VAULT_DIR = ".aether/vault"

    def __init__(
        self,
        vault_dir: Optional[str] = None,
        master_key: Optional[str] = None
    ):
        """
        Initialize the vault.

        Args:
            vault_dir: Directory for vault files (default: .aether/vault)
            master_key: Master encryption key (default: AETHER_MASTER_KEY env)
        """
        self.vault_dir = Path(vault_dir or self.DEFAULT_VAULT_DIR)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

        self.master_key = master_key or os.getenv("AETHER_MASTER_KEY")
        if not self.master_key:
            raise ValueError(
                "Master key required. Set AETHER_MASTER_KEY environment variable "
                "or pass master_key parameter."
            )

        self.tokenizer = SacredTongueTokenizer()

        # In-memory cache (cleared on vault reload)
        self._cache: Dict[str, VaultEntry] = {}

    def store_key(
        self,
        key_id: str,
        value: str,
        expires_in: Optional[int] = None,
        access_level: str = "full"
    ) -> str:
        """
        Store an API key in the vault.

        Args:
            key_id: Unique identifier for the key
            value: The secret value to store
            expires_in: Seconds until expiration (optional)
            access_level: Required access level (full, read_only)

        Returns:
            Path to the stored envelope file
        """
        # Generate salt
        salt = os.urandom(SALT_SIZE)

        # Derive encryption key from master passphrase
        encryption_key = derive_key(self.master_key, salt)

        # Encrypt the value
        ciphertext = aes_gcm_encrypt(value.encode(), encryption_key)

        # Create SS1 envelope
        envelope = SS1Envelope(
            key_id=key_id,
            salt=salt,
            ciphertext=ciphertext
        )

        # Serialize and write
        envelope_str = envelope.serialize()
        file_path = self.vault_dir / f"{key_id}.ss1"

        with open(file_path, 'w') as f:
            f.write(envelope_str)

        # Store metadata
        meta = {
            'key_id': key_id,
            'created_at': time.time(),
            'expires_at': time.time() + expires_in if expires_in else None,
            'access_level': access_level,
        }

        meta_path = self.vault_dir / f"{key_id}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        return str(file_path)

    def get_key(
        self,
        key_id: str,
        context: ContextVector,
        access: AccessResult,
        use_context_binding: bool = False
    ) -> Optional[str]:
        """
        Retrieve and decrypt an API key.

        Args:
            key_id: The key identifier
            context: Current context vector
            access: Access result from GeoSeal gate

        Returns:
            Decrypted key value, or None if access denied

        Note:
            If access is denied, returns None (not random noise)
            to allow proper error handling. The caller should
            use fail_to_noise() if needed.
        """
        # Check access
        if not access.allowed:
            return None

        # Check access level
        meta = self._load_meta(key_id)
        if meta:
            required_level = meta.get('access_level', 'full')
            if required_level == 'full' and access.access_level == 'read_only':
                return None

            # Check expiration
            expires_at = meta.get('expires_at')
            if expires_at and time.time() > expires_at:
                return None

        # Load envelope
        envelope = self._load_envelope(key_id)
        if not envelope:
            return None

        # Derive decryption key
        if use_context_binding:
            # Context-bound: include context vector in key derivation
            base_key = derive_key(self.master_key, envelope.salt)
            decryption_key = derive_context_key(
                base_key,
                context.dimensions,
                envelope.salt
            )
        else:
            # Standard: just master key + salt
            decryption_key = derive_key(self.master_key, envelope.salt)

        # Decrypt
        try:
            plaintext = aes_gcm_decrypt(envelope.ciphertext, decryption_key)
            return plaintext.decode()
        except Exception:
            # Decryption failed (wrong context or tampered)
            return None

    def get_keys(
        self,
        key_ids: list,
        context: ContextVector,
        access: AccessResult
    ) -> Dict[str, Optional[str]]:
        """
        Retrieve multiple keys at once.

        Returns dict mapping key_id to decrypted value (or None).
        """
        return {
            key_id: self.get_key(key_id, context, access)
            for key_id in key_ids
        }

    def delete_key(self, key_id: str) -> bool:
        """Delete a key from the vault."""
        file_path = self.vault_dir / f"{key_id}.ss1"
        meta_path = self.vault_dir / f"{key_id}.meta.json"

        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()

        if key_id in self._cache:
            del self._cache[key_id]

        return deleted

    def list_keys(self) -> list:
        """List all key IDs in the vault."""
        keys = []
        for file_path in self.vault_dir.glob("*.ss1"):
            key_id = file_path.stem
            keys.append(key_id)
        return keys

    def _load_envelope(self, key_id: str) -> Optional[SS1Envelope]:
        """Load an envelope from disk."""
        file_path = self.vault_dir / f"{key_id}.ss1"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            envelope_str = f.read().strip()

        return SS1Envelope.parse(envelope_str)

    def _load_meta(self, key_id: str) -> Optional[Dict]:
        """Load metadata for a key."""
        meta_path = self.vault_dir / f"{key_id}.meta.json"

        if not meta_path.exists():
            return None

        with open(meta_path) as f:
            return json.load(f)


class EnvironmentVault:
    """
    Simple vault using environment variables.

    For development/testing when you don't need full
    context-bound encryption.
    """

    def __init__(self, prefix: str = "AETHER_"):
        self.prefix = prefix

    def get_key(self, key_id: str) -> Optional[str]:
        """Get key from environment variable."""
        env_name = f"{self.prefix}{key_id.upper()}"
        return os.getenv(env_name)

    def set_key(self, key_id: str, value: str):
        """Set key in environment (current process only)."""
        env_name = f"{self.prefix}{key_id.upper()}"
        os.environ[env_name] = value


def setup_vault_interactive():
    """Interactive vault setup for CLI."""
    print("AetherAuth Vault Setup")
    print("=" * 50)

    # Get master key
    master_key = os.getenv("AETHER_MASTER_KEY")
    if not master_key:
        import getpass
        master_key = getpass.getpass("Enter master passphrase: ")
        os.environ["AETHER_MASTER_KEY"] = master_key

    vault = LumoVault(master_key=master_key)

    # Get keys to store
    print("\nEnter API keys (empty key_id to finish):")

    while True:
        key_id = input("Key ID (e.g., 'notion', 'perplexity'): ").strip()
        if not key_id:
            break

        import getpass
        value = getpass.getpass(f"Value for {key_id}: ")

        if value:
            path = vault.store_key(key_id, value)
            print(f"  Stored: {path}")

    print("\nVault setup complete!")
    print(f"Keys stored: {vault.list_keys()}")


if __name__ == "__main__":
    # Demo
    print("Lumo Vault Demo")
    print("=" * 50)

    # Set temporary master key for demo
    os.environ["AETHER_MASTER_KEY"] = "demo-master-key-not-for-production"

    # Create vault
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = LumoVault(vault_dir=tmpdir)

        # Store a key
        print("\nStoring test key...")
        vault.store_key("notion", "secret_notionAPIkey123")
        vault.store_key("perplexity", "pplx-abcdef123456")

        print(f"Keys in vault: {vault.list_keys()}")

        # Create mock context and access
        from context_capture import capture_context_vector
        from geoseal_gate import AccessResult, TrustRing

        context = capture_context_vector()
        access = AccessResult(
            allowed=True,
            ring=TrustRing.CORE,
            distance=0.1,
            access_level="full"
        )

        # Retrieve keys
        print("\nRetrieving keys...")
        notion_key = vault.get_key("notion", context, access)
        pplx_key = vault.get_key("perplexity", context, access)

        print(f"Notion key: {notion_key}")
        print(f"Perplexity key: {pplx_key}")

        # Test denied access
        denied_access = AccessResult(
            allowed=False,
            ring=TrustRing.WALL,
            distance=0.8
        )

        denied_key = vault.get_key("notion", context, denied_access)
        print(f"\nDenied access result: {denied_key}")

        print("\nDemo complete!")
