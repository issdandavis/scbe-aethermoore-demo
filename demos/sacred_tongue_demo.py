#!/usr/bin/env python3
"""
Sacred Tongue Tokenizer Demo
============================

Demonstrates the Six Sacred Tongues cryptographic encoding system.
Each tongue uses 16 prefixes × 16 suffixes = 256 unique tokens.

Usage:
    python demos/sacred_tongue_demo.py
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# THE SIX SACRED TONGUES
# ═══════════════════════════════════════════════════════════════════════════════

TONGUES = {
    'ko': {
        'name': "Kor'aelin",
        'domain': 'nonce/flow/intent',
        'prefixes': ['sil', 'kor', 'vel', 'zar', 'keth', 'thul', 'nav', 'ael',
                     'ra', 'med', 'gal', 'lan', 'joy', 'good', 'nex', 'vara'],
        'suffixes': ['a', 'ae', 'ei', 'ia', 'oa', 'uu', 'eth', 'ar',
                     'or', 'il', 'an', 'en', 'un', 'ir', 'oth', 'esh'],
    },
    'av': {
        'name': 'Avali',
        'domain': 'aad/header/metadata',
        'prefixes': ['saina', 'talan', 'vessa', 'maren', 'oriel', 'serin', 'nurel', 'lirea',
                     'kiva', 'lumen', 'calma', 'ponte', 'verin', 'nava', 'sela', 'tide'],
        'suffixes': ['a', 'e', 'i', 'o', 'u', 'y', 'la', 're',
                     'na', 'sa', 'to', 'mi', 've', 'ri', 'en', 'ul'],
    },
    'ru': {
        'name': 'Runethic',
        'domain': 'salt/binding',
        'prefixes': ['khar', 'drath', 'bront', 'vael', 'ur', 'mem', 'krak', 'tharn',
                     'groth', 'basalt', 'rune', 'sear', 'oath', 'gnarl', 'rift', 'iron'],
        'suffixes': ['ak', 'eth', 'ik', 'ul', 'or', 'ar', 'um', 'on',
                     'ir', 'esh', 'nul', 'vek', 'dra', 'kh', 'va', 'th'],
    },
    'ca': {
        'name': 'Cassisivadan',
        'domain': 'ciphertext/bitcraft',
        'prefixes': ['bip', 'bop', 'klik', 'loopa', 'ifta', 'thena', 'elsa', 'spira',
                     'rythm', 'quirk', 'fizz', 'gear', 'pop', 'zip', 'mix', 'chass'],
        'suffixes': ['a', 'e', 'i', 'o', 'u', 'y', 'ta', 'na',
                     'sa', 'ra', 'lo', 'mi', 'ki', 'zi', 'qwa', 'sh'],
    },
    'um': {
        'name': 'Umbroth',
        'domain': 'redaction/veil',
        'prefixes': ['veil', 'zhur', 'nar', 'shul', 'math', 'hollow', 'hush', 'thorn',
                     'dusk', 'echo', 'ink', 'wisp', 'bind', 'ache', 'null', 'shade'],
        'suffixes': ['a', 'e', 'i', 'o', 'u', 'ae', 'sh', 'th',
                     'ak', 'ul', 'or', 'ir', 'en', 'on', 'vek', 'nul'],
    },
    'dr': {
        'name': 'Draumric',
        'domain': 'tag/structure',
        'prefixes': ['anvil', 'tharn', 'mek', 'grond', 'draum', 'ektal', 'temper', 'forge',
                     'stone', 'steam', 'oath', 'seal', 'frame', 'pillar', 'rivet', 'ember'],
        'suffixes': ['a', 'e', 'i', 'o', 'u', 'ae', 'rak', 'mek',
                     'tharn', 'grond', 'vek', 'ul', 'or', 'ar', 'en', 'on'],
    },
}

# Section to tongue mapping
SECTION_TONGUES = {
    'aad': 'av',    # Avali for metadata/context
    'salt': 'ru',   # Runethic for binding
    'nonce': 'ko',  # Kor'aelin for flow/intent
    'ct': 'ca',     # Cassisivadan for ciphertext
    'tag': 'dr',    # Draumric for auth tag
    'redact': 'um', # Umbroth for redaction
}


class SacredTongueTokenizer:
    """
    Sacred Tongue Tokenizer

    Encodes/decodes bytes to spell-text tokens using the 16×16 grid.
    Each byte maps to exactly one token: prefix'suffix
    """

    def __init__(self, tongue_code: str = 'ko'):
        if tongue_code not in TONGUES:
            raise ValueError(f"Unknown tongue: {tongue_code}. Valid: {list(TONGUES.keys())}")

        self.tongue = TONGUES[tongue_code]
        self.code = tongue_code

        # Build lookup tables
        self.byte_to_token: List[str] = []
        self.token_to_byte: Dict[str, int] = {}

        for b in range(256):
            prefix_idx = b >> 4      # High nibble (0-15)
            suffix_idx = b & 0x0f    # Low nibble (0-15)
            token = f"{self.tongue['prefixes'][prefix_idx]}'{self.tongue['suffixes'][suffix_idx]}"
            self.byte_to_token.append(token)
            self.token_to_byte[token] = b

    def encode_byte(self, b: int) -> str:
        """Encode a single byte to a token"""
        if not 0 <= b <= 255:
            raise ValueError(f"Byte must be 0-255, got {b}")
        return self.byte_to_token[b]

    def decode_token(self, token: str) -> int:
        """Decode a single token to a byte"""
        if token not in self.token_to_byte:
            raise ValueError(f"Unknown token: {token}")
        return self.token_to_byte[token]

    def encode(self, data: bytes) -> str:
        """Encode bytes to space-separated spell-text tokens with tongue prefix"""
        tokens = [f"{self.code}:{self.byte_to_token[b]}" for b in data]
        return ' '.join(tokens)

    def encode_compact(self, data: bytes) -> str:
        """Encode bytes without tongue prefix (compact format)"""
        tokens = [self.byte_to_token[b] for b in data]
        return ' '.join(tokens)

    def decode(self, spelltext: str) -> bytes:
        """Decode spell-text tokens back to bytes"""
        result = []
        for token in spelltext.split():
            if not token:
                continue
            # Strip tongue prefix if present (e.g., "ko:sil'a" → "sil'a")
            clean_token = token.split(':')[1] if ':' in token else token
            if clean_token not in self.token_to_byte:
                raise ValueError(f"Unknown token: {clean_token}")
            result.append(self.token_to_byte[clean_token])
        return bytes(result)

    def get_all_tokens(self) -> List[str]:
        """Get all 256 tokens for this tongue"""
        return self.byte_to_token.copy()


def encode_to_spelltext(data: bytes, section: str) -> str:
    """Encode bytes using the canonical tongue for a section"""
    tongue_code = SECTION_TONGUES.get(section, 'ca')
    tokenizer = SacredTongueTokenizer(tongue_code)
    return f"{tongue_code}:{tokenizer.encode_compact(data)}"


def decode_from_spelltext(spelltext: str, section: str) -> bytes:
    """Decode spell-text using the canonical tongue for a section"""
    tongue_code = SECTION_TONGUES.get(section, 'ca')
    tokenizer = SacredTongueTokenizer(tongue_code)

    # Remove tongue prefix if present
    text = spelltext
    if text.startswith(f"{tongue_code}:"):
        text = text[len(tongue_code) + 1:]

    return tokenizer.decode(text)


def format_ss1_blob(kid: str, aad: str, salt: bytes, nonce: bytes,
                    ciphertext: bytes, tag: bytes) -> str:
    """Format a complete SS1 spell-text blob"""
    parts = [
        'SS1',
        f'kid={kid}',
        f'aad={aad}',
        encode_to_spelltext(salt, 'salt'),
        encode_to_spelltext(nonce, 'nonce'),
        encode_to_spelltext(ciphertext, 'ct'),
        encode_to_spelltext(tag, 'tag'),
    ]
    return '|'.join(parts)


def parse_ss1_blob(blob: str) -> Dict:
    """Parse an SS1 spell-text blob"""
    if not blob.startswith('SS1|'):
        raise ValueError("Invalid SS1 blob: must start with 'SS1|'")

    result = {'version': 'SS1'}
    parts = blob.split('|')

    for part in parts[1:]:
        if part.startswith('kid='):
            result['kid'] = part[4:]
        elif part.startswith('aad='):
            result['aad'] = part[4:]
        elif part.startswith('ru:'):
            result['salt'] = decode_from_spelltext(part, 'salt')
        elif part.startswith('ko:'):
            result['nonce'] = decode_from_spelltext(part, 'nonce')
        elif part.startswith('ca:'):
            result['ct'] = decode_from_spelltext(part, 'ct')
        elif part.startswith('dr:'):
            result['tag'] = decode_from_spelltext(part, 'tag')

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo_manual_encoding():
    """Tutorial 1: Manual byte-to-token encoding"""
    print("=" * 60)
    print("Tutorial 1: Manual Encoding (Learning)")
    print("=" * 60)

    # Example nonce bytes
    nonce_bytes = bytes([0x3c, 0x5a, 0x7f, 0x2e])

    print(f"\nRaw bytes: {nonce_bytes.hex()}")
    print("\nManual encoding process:")

    tokenizer = SacredTongueTokenizer('ko')

    for i, b in enumerate(nonce_bytes):
        high_nibble = b >> 4
        low_nibble = b & 0x0f
        prefix = TONGUES['ko']['prefixes'][high_nibble]
        suffix = TONGUES['ko']['suffixes'][low_nibble]
        token = f"{prefix}'{suffix}"

        print(f"  Byte {i}: 0x{b:02x} = high:{high_nibble}, low:{low_nibble}")
        print(f"         → prefix[{high_nibble}]='{prefix}', suffix[{low_nibble}]='{suffix}'")
        print(f"         → Token: \"{token}\"")

    # Full encoding
    spell = tokenizer.encode(nonce_bytes)
    print(f"\nComplete spell-text: {spell}")

    # Verify round-trip
    recovered = tokenizer.decode(spell)
    assert recovered == nonce_bytes
    print("✓ Round-trip verified!")


def demo_ss1_blob():
    """Tutorial 2: Building a complete SS1 encrypted blob"""
    print("\n" + "=" * 60)
    print("Tutorial 2: SS1 Encrypted Blob")
    print("=" * 60)

    # Generate cryptographic material
    salt = os.urandom(16)
    nonce = os.urandom(12)
    ciphertext = os.urandom(48)  # Simulated encrypted data
    tag = os.urandom(16)

    print("\nCryptographic material:")
    print(f"  Salt (16 bytes):       {salt.hex()[:32]}...")
    print(f"  Nonce (12 bytes):      {nonce.hex()}")
    print(f"  Ciphertext (48 bytes): {ciphertext.hex()[:32]}...")
    print(f"  Tag (16 bytes):        {tag.hex()}")

    # Format as SS1 blob
    kid = f"demo-key-{int(time.time())}"
    aad = "demo-context"

    blob = format_ss1_blob(kid, aad, salt, nonce, ciphertext, tag)

    print(f"\nSS1 Blob (spell-text):")
    print("-" * 60)
    # Print in sections for readability
    for i, section in enumerate(blob.split('|')):
        if i == 0:
            print(f"  {section}")
        elif len(section) > 50:
            print(f"  |{section[:50]}...")
        else:
            print(f"  |{section}")

    # Parse back
    parsed = parse_ss1_blob(blob)
    print("\n✓ Parsed successfully!")
    print(f"  kid: {parsed['kid']}")
    print(f"  aad: {parsed['aad']}")
    print(f"  salt recovered: {parsed['salt'] == salt}")
    print(f"  nonce recovered: {parsed['nonce'] == nonce}")
    print(f"  ciphertext recovered: {parsed['ct'] == ciphertext}")
    print(f"  tag recovered: {parsed['tag'] == tag}")


def demo_tongue_comparison():
    """Compare all six tongues encoding the same byte"""
    print("\n" + "=" * 60)
    print("Tongue Comparison: Same byte, different tongues")
    print("=" * 60)

    test_byte = 0x5A  # Arbitrary test byte

    print(f"\nByte 0x{test_byte:02x} ({test_byte}) encoded in each tongue:")
    print("-" * 40)

    for code, spec in TONGUES.items():
        tokenizer = SacredTongueTokenizer(code)
        token = tokenizer.encode_byte(test_byte)
        print(f"  {code.upper():2} ({spec['name']:12}): {token:15} [{spec['domain']}]")


def demo_all_256_tokens():
    """Show sample of the 256 tokens for one tongue"""
    print("\n" + "=" * 60)
    print("Sample: First 16 tokens of Kor'aelin")
    print("=" * 60)

    tokenizer = SacredTongueTokenizer('ko')
    tokens = tokenizer.get_all_tokens()

    print("\nByte → Token mapping (first 16):")
    for i in range(16):
        print(f"  0x{i:02x} → {tokens[i]}")

    print(f"\n...and 240 more tokens (total: {len(tokens)})")


def main():
    """Run all demos"""
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " SACRED TONGUE TOKENIZER DEMO ".center(58) + "║")
    print("║" + " SCBE SpiralSeal SS1 Encoding ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    demo_manual_encoding()
    demo_ss1_blob()
    demo_tongue_comparison()
    demo_all_256_tokens()

    print("\n" + "=" * 60)
    print("✓ All demos completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
