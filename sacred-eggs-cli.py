#!/usr/bin/env python3
"""
Sacred Eggs CLI - Command-line interface for Sacred Egg operations.

Usage:
  # Create an egg
  python sacred-eggs-cli.py egg-create --payload-b64 <base64> --primary-tongue ko \
      --glyph "diamond" --hatch-condition '{"ring":"inner","path":"interior"}' \
      --context '[0.2,-0.3,0.7]' --kem-key <base64> --dsa-key <base64>

  # Hatch an egg
  python sacred-eggs-cli.py egg-hatch --egg-json egg.json --agent-tongue ko \
      --ritual-mode solitary --context '[0.2,-0.3,0.7]' \
      --kem-key <base64> --dsa-pk <base64>

  # Run selftest
  python sacred-eggs-cli.py selftest
"""

import argparse
import base64
import dataclasses
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from crypto.sacred_tongues import TONGUES, SacredTongueTokenizer
from crypto.sacred_eggs import (
    SacredEgg,
    HatchResult,
    CrossTokenizer,
    SacredEggIntegrator,
    selftest as sacred_eggs_selftest,
)


TONGUE_CODES = list(TONGUES.keys())
RITUAL_MODES = ["solitary", "triadic", "ring_descent"]


def load_lexicons(path: str = None) -> SacredTongueTokenizer:
    """Load lexicons (tokenizer). For now, uses built-in TONGUES."""
    return SacredTongueTokenizer(TONGUES)


def cmd_egg_create(args) -> int:
    """Create a new Sacred Egg."""
    tok = load_lexicons(getattr(args, 'lexicons', None))
    xt = CrossTokenizer(tok)
    sei = SacredEggIntegrator(xt)

    try:
        payload = base64.b64decode(args.payload_b64)
    except Exception as e:
        print(f"Error decoding payload: {e}", file=sys.stderr)
        return 1

    try:
        ctx = json.loads(args.context)
    except json.JSONDecodeError as e:
        print(f"Error parsing context JSON: {e}", file=sys.stderr)
        return 1

    try:
        cond = json.loads(args.hatch_condition)
    except json.JSONDecodeError as e:
        print(f"Error parsing hatch-condition JSON: {e}", file=sys.stderr)
        return 1

    try:
        egg = sei.create_egg(
            payload=payload,
            primary_tongue=args.primary_tongue,
            glyph=args.glyph,
            hatch_condition=cond,
            context=ctx,
            pk_kem_b64=args.kem_key,
            sk_dsa_b64=args.dsa_key
        )
    except Exception as e:
        print(f"Error creating egg: {e}", file=sys.stderr)
        return 1

    # Serialize egg to JSON
    egg_dict = dataclasses.asdict(egg)
    output = json.dumps(egg_dict, ensure_ascii=False, indent=2)

    if args.outfile:
        with open(args.outfile, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Egg written to {args.outfile}")
    else:
        print(output)

    return 0


def cmd_egg_hatch(args) -> int:
    """Attempt to hatch a Sacred Egg."""
    tok = load_lexicons(getattr(args, 'lexicons', None))
    xt = CrossTokenizer(tok)
    sei = SacredEggIntegrator(xt)

    # Load egg JSON
    try:
        with open(args.egg_json, 'r', encoding='utf-8') as f:
            egg_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Egg file not found: {args.egg_json}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing egg JSON: {e}", file=sys.stderr)
        return 1

    try:
        ctx = json.loads(args.context)
    except json.JSONDecodeError as e:
        print(f"Error parsing context JSON: {e}", file=sys.stderr)
        return 1

    # Parse optional path history (for ring_descent)
    path_history = None
    if hasattr(args, 'path_history') and args.path_history:
        try:
            path_history = json.loads(args.path_history)
        except json.JSONDecodeError as e:
            print(f"Error parsing path-history JSON: {e}", file=sys.stderr)
            return 1

    # Parse optional triad tongues (for triadic)
    triad_tongues = None
    if hasattr(args, 'triad_tongues') and args.triad_tongues:
        triad_tongues = [t.strip() for t in args.triad_tongues.split(',')]

    try:
        result = sei.hatch_egg(
            egg=egg_dict,
            current_context=ctx,
            agent_tongue=args.agent_tongue,
            sk_kem_b64=args.kem_key,
            pk_dsa_b64=args.dsa_pk,
            ritual_mode=args.ritual_mode,
            path_history=path_history,
            triad_tongues=triad_tongues
        )
    except Exception as e:
        print(f"Error during hatch: {e}", file=sys.stderr)
        return 1

    if result.success:
        # Output decoded tokens
        output = " ".join(result.tokens)

        if args.outfile:
            with open(args.outfile, 'w', encoding='utf-8') as f:
                f.write(output + "\n")
            print(f"Tokens written to {args.outfile}")
        else:
            print(output)

        # Optionally output attestation
        if args.verbose:
            print("\n--- Attestation ---")
            print(json.dumps(result.attestation, indent=2))

        return 0
    else:
        print(f"Hatch failed: {result.reason}", file=sys.stderr)
        return 1


def cmd_selftest(args) -> int:
    """Run Sacred Eggs selftest."""
    return sacred_eggs_selftest()


def cmd_demo(args) -> int:
    """Run a quick demo of Sacred Eggs."""
    print("=== Sacred Eggs Demo ===\n")

    # Setup
    payload = b"sacred secret"
    ctx = [0.2, -0.3, 0.7, 1.0, -2.0, 0.5, 3.1, -9.9, 0.0]
    kem_key = base64.b64encode(b"kem-key-32bytes-demo____").decode()
    dsa_key = base64.b64encode(b"dsa-key-32bytes-demo____").decode()
    # Note: ctx[0]=0.2 gives r=0.02 -> "core" ring; classify() gives "exterior" path
    cond = {"ring": "core", "path": "exterior"}

    print(f"Payload: {payload.decode()}")
    print(f"Primary tongue: ko (Kor'aelin)")
    print(f"Hatch condition: {cond}")
    print()

    # Create tokenizer and integrator
    tok = SacredTongueTokenizer(TONGUES)
    xt = CrossTokenizer(tok)
    sei = SacredEggIntegrator(xt)

    # Create egg
    print("Creating Sacred Egg...")
    egg = sei.create_egg(payload, "ko", "\u25c7", cond, ctx, kem_key, dsa_key)
    print(f"  Egg ID: {egg.egg_id}")
    print(f"  Glyph: {egg.glyph}")
    print()

    # Hatch with correct tongue (solitary)
    print("Hatching with correct tongue (solitary mode)...")
    result = sei.hatch_egg(
        dataclasses.asdict(egg), ctx, "ko", kem_key, dsa_key, "solitary"
    )
    if result.success:
        print(f"  Success: {result.reason}")
        print(f"  Tokens: {' '.join(result.tokens[:5])}...")
        decoded = xt.to_bytes_from_tokens('ko', " ".join(result.tokens))
        print(f"  Decoded: {decoded.decode()}")
    else:
        print(f"  Failed: {result.reason}")
    print()

    # Attempt with wrong tongue
    print("Attempting hatch with wrong tongue (dr)...")
    result = sei.hatch_egg(
        dataclasses.asdict(egg), ctx, "dr", kem_key, dsa_key, "solitary"
    )
    if result.success:
        print(f"  Success: {result.reason}")
    else:
        print(f"  Failed: {result.reason}")
    print()

    # Triadic ritual
    print("Hatching with triadic ritual...")
    egg_triadic = sei.create_egg(
        payload, "ko", "\u25c7",
        {"ring": "core", "path": "exterior", "min_weight": 5.0},
        ctx, kem_key, dsa_key
    )
    result = sei.hatch_egg(
        dataclasses.asdict(egg_triadic), ctx, "ko", kem_key, dsa_key,
        "triadic", triad_tongues=["ko", "ru", "um"]
    )
    if result.success:
        print(f"  Success: {result.reason}")
        print(f"  Triad weight: {xt.WEIGHT['ko'] + xt.WEIGHT['ru'] + xt.WEIGHT['um']:.3f}")
    else:
        print(f"  Failed: {result.reason}")
    print()

    print("=== Demo Complete ===")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Sacred Eggs CLI - Cryptographically sealed token containers"
    )
    parser.add_argument('--version', action='version', version='1.0.0')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # egg-create subcommand
    p_create = subparsers.add_parser('egg-create', help='Create a new Sacred Egg')
    p_create.add_argument('--payload-b64', required=True,
                          help='Base64-encoded payload to seal')
    p_create.add_argument('--primary-tongue', required=True, choices=TONGUE_CODES,
                          help='Primary tongue for encoding')
    p_create.add_argument('--glyph', default='\u25c7',
                          help='Visual identifier for the egg')
    p_create.add_argument('--hatch-condition', default='{"ring":"inner","path":"interior"}',
                          help='JSON hatch condition')
    p_create.add_argument('--context', required=True,
                          help='JSON array of context floats')
    p_create.add_argument('--kem-key', required=True,
                          help='Base64 KEM public key')
    p_create.add_argument('--dsa-key', required=True,
                          help='Base64 DSA signing key')
    p_create.add_argument('--out', dest='outfile',
                          help='Output file (default: stdout)')
    p_create.add_argument('--lexicons',
                          help='Path to custom lexicons file')
    p_create.set_defaults(func=cmd_egg_create)

    # egg-hatch subcommand
    p_hatch = subparsers.add_parser('egg-hatch', help='Attempt to hatch a Sacred Egg')
    p_hatch.add_argument('--egg-json', required=True,
                         help='Path to egg JSON file')
    p_hatch.add_argument('--agent-tongue', required=True, choices=TONGUE_CODES,
                         help='Agent\'s preferred tongue')
    p_hatch.add_argument('--ritual-mode', default='solitary', choices=RITUAL_MODES,
                         help='Ritual mode for hatching')
    p_hatch.add_argument('--context', required=True,
                         help='JSON array of current context floats')
    p_hatch.add_argument('--kem-key', required=True,
                         help='Base64 KEM secret key')
    p_hatch.add_argument('--dsa-pk', required=True,
                         help='Base64 DSA verification key')
    p_hatch.add_argument('--path-history',
                         help='JSON array of path history (for ring_descent)')
    p_hatch.add_argument('--triad-tongues',
                         help='Comma-separated tongue codes (for triadic)')
    p_hatch.add_argument('--out', dest='outfile',
                         help='Output file (default: stdout)')
    p_hatch.add_argument('--verbose', '-v', action='store_true',
                         help='Show attestation details')
    p_hatch.add_argument('--lexicons',
                         help='Path to custom lexicons file')
    p_hatch.set_defaults(func=cmd_egg_hatch)

    # selftest subcommand
    p_selftest = subparsers.add_parser('selftest', help='Run Sacred Eggs selftest')
    p_selftest.set_defaults(func=cmd_selftest)

    # demo subcommand
    p_demo = subparsers.add_parser('demo', help='Run a quick demo')
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
