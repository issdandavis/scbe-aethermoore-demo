"""
Quick Vault Setup (Windows-friendly)

Simple setup without hidden input (for terminals that don't support getpass).
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("AetherAuth Quick Vault Setup")
    print("=" * 60)
    print()
    print("WARNING: Input will be VISIBLE. Clear your terminal after.")
    print()

    # Import here to catch errors
    try:
        from aetherauth.vault_access import LumoVault
        print("[OK] Vault module loaded")
    except Exception as e:
        print(f"[ERROR] Failed to import vault: {e}")
        return

    # Get master key
    master_key = os.getenv("AETHER_MASTER_KEY")

    if not master_key:
        print("\nNo AETHER_MASTER_KEY found in environment.")
        master_key = input("Enter master passphrase (min 16 chars): ").strip()

        if len(master_key) < 16:
            print("ERROR: Passphrase too short")
            return

        os.environ["AETHER_MASTER_KEY"] = master_key
        print("[OK] Master key set")
    else:
        print("[OK] Using AETHER_MASTER_KEY from environment")

    # Create vault
    vault_dir = ".aether/vault"
    try:
        vault = LumoVault(vault_dir=vault_dir, master_key=master_key)
        print(f"[OK] Vault initialized at {vault_dir}")
    except Exception as e:
        print(f"[ERROR] Vault init failed: {e}")
        return

    # Show existing
    existing = vault.list_keys()
    if existing:
        print(f"\nExisting keys: {existing}")

    # Store Notion key
    print("\n--- Notion API Key ---")
    notion_key = input("Notion key (or Enter to skip): ").strip()
    if notion_key:
        vault.store_key("notion", notion_key)
        print("[OK] Notion key stored")

    # Store Perplexity key
    print("\n--- Perplexity API Key ---")
    pplx_key = input("Perplexity key (or Enter to skip): ").strip()
    if pplx_key:
        vault.store_key("perplexity", pplx_key)
        print("[OK] Perplexity key stored")

    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"Keys in vault: {vault.list_keys()}")
    print()
    print("Set your master key permanently:")
    print('  $env:AETHER_MASTER_KEY = "your-passphrase"')
    print()
    print("CLEAR YOUR TERMINAL NOW (cls) to hide typed keys!")

if __name__ == "__main__":
    main()
