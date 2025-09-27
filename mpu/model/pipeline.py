#!/usr/bin/env python3

# Golden model pipeline runner:
# input -> RS(255, 223) encoder -> byte wise interleaver (I=2) -> ouput

# examples:
#   python -m mpu.model.pipeline --text "HELLO WORLD" --depth 4 --pad zero --print
#   python -m mpu.model.pipeline --hex "0011223344" --depth 2 --pad zero --out out.bin
#   python -m mpu.model.pipeline --infile input.bin --depth 1 --check --print


# Notes:
# - without padding, input length must be a multiple of 223 bytes
# - with --pad zero, the last (partial) 223-byte block is zero-padded


import argparse
import sys
from typing import Optional

from mpu.model.reed_solomon import rs_encode, rs_syndromes, k as RS_K, n as RS_N
from mpu.model.interleaver import interleave, deinterleave, CODEWORD_BYTES, POSSIBLE_DEPTHS
from mpu.model.scrambler import scramble_bits, descramble_bits ,DEFAULT_SEED

def _read_input(args: argparse.Namespace) -> bytes:
    if args.text is not None:
        return args.text.encode("utf-8")
    if args.hex is not None:
        s = args.hex.replace(" ", "").replace("\n", "")
        try:
            return bytes.fromhex(s)
        except ValueError:
            print("Error: --hex contains non-hex characters.", file=sys.stderr)
            sys.exit(2)
    if args.infile is not None:
        with open(args.infile, "rb") as f:
            return f.read()
    # Fallback: read from stdin as text
    data = sys.stdin.read()
    if not data:
        print("No input provided. Use --text/--hex/--infile or pipe data via stdin.", file=sys.stderr)
        sys.exit(2)
    return data.encode("utf-8")

def _pad_to_block(data: bytes, block: int) -> bytes:
    rem = len(data) % block
    if rem == 0:
        return data
    return data + bytes(block - rem)

def _chunk(seq: bytes, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Run RS(255,223) + interleaver golden pipeline.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="UTF-8 text input.")
    src.add_argument("--hex", help="Hex string input (spaces allowed).")
    src.add_argument("--infile", help="Binary input file.")
    p.add_argument("--depth", type=int, default=2, choices=sorted(POSSIBLE_DEPTHS),
                   help=f"Interleaver depth I (default: 2)")
    p.add_argument("--pad", choices=["none", "zero"], default="zero",
                   help="Pad input to multiples of 223 bytes (default: zero).")
    p.add_argument("--no-scramble", action="store_true",
                   help="Disable scrambler (enabled by default).")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED,
                   help="Override scrambler seed (int, e.g. 0b1011101).")
    p.add_argument("--out", help="Write pipeline output (binary) to this file.")
    p.add_argument("--print", dest="do_print", action="store_true",
                   help="Print space-separated bytes of pipeline output (default if no --out given).")
    p.add_argument("--no-print", dest="do_print", action="store_false",
                   help="Suppress printing even if no --out is given.")
    p.set_defaults(do_print=True)
    p.add_argument("--hexout", action="store_true",
                   help="Print hex of pipeline output.")
    p.add_argument("--check", action="store_true",
                   help="After RS encode, verify each 255B codeword has zero syndromes.")
    p.add_argument("--show", action="store_true",
                   help="Show brief sizes & block counts.")
    args = p.parse_args(argv)

    data = _read_input(args)

    # Handle padding policy for RS encoder
    if args.pad == "zero":
        data_for_rs = _pad_to_block(data, RS_K)
    else:
        if len(data) % RS_K != 0:
            print(f"Input length {len(data)} is not a multiple of {RS_K} and --pad is 'none'.",
                  file=sys.stderr)
            return 2
        data_for_rs = data

    if args.show:
        blocks = len(data_for_rs) // RS_K
        print(f"Input bytes: {len(data)} (after pad policy: {len(data_for_rs)})")
        print(f"RS blocks: {blocks} (k={RS_K}, n={RS_N}), interleaver depth I={args.depth}, "
              f"scrambler={'OFF' if args.no_scramble else 'ON'}")

    # RS encode -> concatenated 255B codewords
    rs_out = rs_encode(data_for_rs)

    if args.check:
        # Check each codeword’s syndromes are all zero
        for idx, cw in enumerate(_chunk(rs_out, CODEWORD_BYTES)):
            S = rs_syndromes(cw)
            if any(S):
                print(f"[check] Non-zero syndromes in codeword {idx}: {S}", file=sys.stderr)
                return 1
        if args.show:
            print("[check] All codewords have zero syndromes ✓")

    # Interleave per codeword with depth I
    interleaved = interleave(rs_out, args.depth)

    # Scramble (enabled by default, disable with --no-scramble)
    if args.no_scramble:
        scrambled = interleaved
    else:
        scrambled = scramble_bits(interleaved, seed=args.seed)

    # Output
    final_out = scrambled
    if args.out:
        with open(args.out, "wb") as f:
            f.write(final_out)

    # Output stages
    if args.do_print:
        print("input =")
        print(" ".join(str(b) for b in data))
        print()

        print("RS output =")
        print(" ".join(str(b) for b in rs_out))
        print()

        print(f"interleaver output (I={args.depth}) =")
        print(" ".join(str(b) for b in interleaved))
        print()

        if not args.no_scramble:
            print(f"scrambler output (poly=x^7+x^4+1, seed={args.seed:#09b}) =")
            print(" ".join(str(b) for b in scrambled))
            print()

            # print(f"unscrambler output (poly=x^7+x^4+1, seed={args.seed:#09b}) =")
            # print(" ".join(str(b) for b in descramble_bits(scrambled, seed=args.seed)))
            # print()

        

    if args.hexout:
        print(final_out.hex())

    # If neither print nor hexout nor out is given, show a tiny summary
    if not (args.out or args.do_print or args.hexout):
        print(f"Pipeline OK. Output bytes: {len(final_out)} "
              f"({len(final_out)//CODEWORD_BYTES} codewords). Use --out/--no-print/--hexout to control output.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
