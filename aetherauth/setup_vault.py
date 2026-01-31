"""
AetherAuth Vault Setup

Interactive script to securely store API keys in the Lumo Vault.
All inputs are hidden (password-style) for security.

Usage:
    python -m aetherauth.setup_vault
"""

import os
import sys
import getpass
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aetherauth.vault_access import LumoVault
from aetherauth.sacred_tongues import SacredTongueTokenizer


def setup_vault():
    """Interactive vault setup."""
    print("=" * 60)
    print("AetherAuth Vault Setup")
    print("=" * 60)
    print()
    print("This will securely store your API keys in the Lumo Vault.")
    print("All inputs are hidden for security.")
    print()

    # Get or set master key
    master_key = os.getenv("AETHER_MASTER_KEY")

    if master_key:
        print("Using existing AETHER_MASTER_KEY from environment.")
        use_existing = input("Continue with this key? [Y/n]: ").strip().lower()
        if use_existing == 'n':
            master_key = None

    if not master_key:
        print("\nEnter a master passphrase (min 16 characters):")
        print("This encrypts all your API keys. DO NOT FORGET IT.")

        while True:
            master_key = getpass.getpass("Master passphrase: ")
            if len(master_key) < 16:
                print("Passphrase too short. Minimum 16 characters.")
                continue

            confirm = getpass.getpass("Confirm passphrase: ")
            if master_key != confirm:
                print("Passphrases don't match. Try again.")
                continue

            break

        # Set in environment for this session
        os.environ["AETHER_MASTER_KEY"] = master_key
        print("\nMaster key set for this session.")
        print("To persist, add to your environment:")
        print('  export AETHER_MASTER_KEY="your-passphrase"')

    # Initialize vault
    vault_dir = ".aether/vault"
    print(f"\nVault directory: {vault_dir}")

    vault = LumoVault(vault_dir=vault_dir, master_key=master_key)

    # Show existing keys
    existing = vault.list_keys()
    if existing:
        print(f"\nExisting keys in vault: {existing}")

    # Define keys to store
    keys_to_store = [
        ("notion", "Notion API Key", "secret_..."),
        ("perplexity", "Perplexity API Key", "pplx-..."),
    ]

    print("\n" + "-" * 60)
    print("Enter your API keys (press Enter to skip)")
    print("-" * 60)

    stored = []

    for key_id, name, hint in keys_to_store:
        print(f"\n{name} ({hint}):")

        if key_id in existing:
            overwrite = input(f"  Key '{key_id}' exists. Overwrite? [y/N]: ").strip().lower()
            if overwrite != 'y':
                print(f"  Skipping {key_id}")
                continue

        value = getpass.getpass(f"  {key_id}: ")

        if not value:
            print(f"  Skipping {key_id} (empty)")
            continue

        # Validate format
        if key_id == "notion" and not value.startswith("secret_"):
            print("  Warning: Notion keys typically start with 'secret_'")
            proceed = input("  Store anyway? [y/N]: ").strip().lower()
            if proceed != 'y':
                continue

        if key_id == "perplexity" and not value.startswith("pplx-"):
            print("  Warning: Perplexity keys typically start with 'pplx-'")
            proceed = input("  Store anyway? [y/N]: ").strip().lower()
            if proceed != 'y':
                continue

        # Store the key
        path = vault.store_key(key_id, value)
        stored.append(key_id)
        print(f"  Stored: {path}")

    # Ask for additional keys
    print("\n" + "-" * 60)
    print("Add additional keys? (empty key_id to finish)")
    print("-" * 60)

    while True:
        key_id = input("\nKey ID (e.g., 'openai', 'anthropic'): ").strip()
        if not key_id:
            break

        # Sanitize key_id
        key_id = key_id.lower().replace(" ", "_")

        if key_id in existing and key_id not in stored:
            overwrite = input(f"  Key '{key_id}' exists. Overwrite? [y/N]: ").strip().lower()
            if overwrite != 'y':
                continue

        value = getpass.getpass(f"  Value for {key_id}: ")

        if not value:
            print("  Skipping (empty value)")
            continue

        path = vault.store_key(key_id, value)
        stored.append(key_id)
        print(f"  Stored: {path}")

    # Summary
    print("\n" + "=" * 60)
    print("Vault Setup Complete")
    print("=" * 60)

    all_keys = vault.list_keys()
    print(f"\nKeys in vault: {all_keys}")
    print(f"Keys added this session: {stored}")

    print("\nNext steps:")
    print("1. Set AETHER_MASTER_KEY in your environment")
    print("2. Use the vault in your code:")
    print()
    print("   from aetherauth import capture_context_vector, AetherAuthGate, LumoVault")
    print()
    print("   context = capture_context_vector()")
    print("   gate = AetherAuthGate()")
    print("   access = gate.check_access(context)")
    print()
    print("   if access.allowed:")
    print("       vault = LumoVault()")
    print("       notion_key = vault.get_key('notion', context, access)")
    print()

    # Security reminder
    print("-" * 60)
    print("SECURITY REMINDERS:")
    print("- Never commit .aether/vault/ to git")
    print("- Add '.aether/' to your .gitignore")
    print("- Back up your master passphrase securely")
    print("-" * 60)


if __name__ == "__main__":
    setup_vault()
