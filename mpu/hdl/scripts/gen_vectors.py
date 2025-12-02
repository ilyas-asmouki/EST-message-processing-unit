#!/usr/bin/env python3
"""Generate stage-by-stage reference vectors from the Python golden model.

These vectors seed HDL testbenches before we have RTL in place. We can export
inputs/outputs for the RS encoder, interleaver, scrambler, convolutional
encoder, and differential encoder, using the exact same configuration as the
software pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from mpu.model.reed_solomon import rs_encode, k as RS_K
from mpu.model.interleaver import interleave, POSSIBLE_DEPTHS
from mpu.model.scrambler import scramble_bits, DEFAULT_SEED
from mpu.model.conv_encoder import conv_encode
from mpu.model.diff_encoder import diff_encode
from mpu.model.qpsk import qpsk_modulate_bytes_fixed, pack_iq_le_bytes
from mpu.model.rrc import rrc_filter, upsample, SAMPLES_PER_SYMBOL, NUM_TAPS
import numpy as np


def _read_input(args: argparse.Namespace) -> bytes:
    if args.text is not None:
        return args.text.encode("utf-8")
    if args.hex is not None:
        s = args.hex.replace(" ", "").replace("\n", "")
        return bytes.fromhex(s)
    if args.infile is not None:
        return Path(args.infile).read_bytes()
    raise SystemExit("Provide --text/--hex/--infile for vector generation")


def _pad_to_block(data: bytes, block: int) -> bytes:
    rem = len(data) % block
    if rem == 0:
        return data
    return data + bytes(block - rem)


def _write_hex_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in range(0, len(data), 16):
            chunk = data[idx : idx + 16]
            f.write(" ".join(f"{b:02X}" for b in chunk))
            f.write("\n")


def _emit_stage(out_dir: Path, stage: str, stage_in: bytes, stage_out: bytes, meta: dict) -> None:
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    _write_hex_file(stage_dir / "input.hex", stage_in)
    _write_hex_file(stage_dir / "output.hex", stage_out)
    with (stage_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def generate_vectors(args: argparse.Namespace) -> int:
    data = _read_input(args)
    padded = _pad_to_block(data, RS_K)

    rs_out = rs_encode(padded)
    interleaved = interleave(rs_out, args.depth)
    scrambled = interleaved if args.no_scramble else scramble_bits(interleaved, seed=args.seed)
    convolved = conv_encode(scrambled)
    diffed = diff_encode(convolved)
    qpsk_out_iq = qpsk_modulate_bytes_fixed(diffed)
    qpsk_bytes = pack_iq_le_bytes(qpsk_out_iq)

    # RRC Processing
    qpsk_i = qpsk_out_iq[0::2]
    qpsk_q = qpsk_out_iq[1::2]
    qpsk_i_up = upsample(qpsk_i, SAMPLES_PER_SYMBOL)
    qpsk_q_up = upsample(qpsk_q, SAMPLES_PER_SYMBOL)
    
    group_delay_samples = (NUM_TAPS - 1) // 2
    total_delay_samples = 2 * group_delay_samples
    qpsk_i_up_padded = np.concatenate([qpsk_i_up, np.zeros(total_delay_samples, dtype=np.int16)])
    qpsk_q_up_padded = np.concatenate([qpsk_q_up, np.zeros(total_delay_samples, dtype=np.int16)])
    
    rrc_i, rrc_q = rrc_filter(qpsk_i_up_padded, qpsk_q_up_padded)
    
    rrc_i_int = np.clip(rrc_i, -32768, 32767).astype(np.int16)
    rrc_q_int = np.clip(rrc_q, -32768, 32767).astype(np.int16)
    
    rrc_out_iq = np.empty(len(rrc_i_int) + len(rrc_q_int), dtype=np.int16)
    rrc_out_iq[0::2] = rrc_i_int
    rrc_out_iq[1::2] = rrc_q_int
    rrc_bytes = pack_iq_le_bytes(rrc_out_iq)

    stage_map: dict[str, tuple[bytes, bytes, dict]] = {
        "rs": (padded, rs_out, {"block_bytes": RS_K, "codeword_bytes": len(rs_out)}),
        "interleaver": (rs_out, interleaved, {"depth": args.depth}),
        "scrambler": (interleaved, scrambled, {"enabled": not args.no_scramble, "seed": args.seed}),
        "conv": (scrambled, convolved, {"rate": "1/2", "constraint_length": 7}),
        "diff": (convolved, diffed, {}),
        "qpsk": (diffed, qpsk_bytes, {"format": "Q1.15", "interleaved": True}),
        "rrc": (qpsk_bytes, rrc_bytes, {"sps": SAMPLES_PER_SYMBOL, "taps": NUM_TAPS}),
    }

    out_dir = Path(args.out_dir)
    stages = args.stage or ["rs"]
    for stage in stages:
        if stage not in stage_map:
            raise SystemExit(f"Unknown stage '{stage}'. Choices: {sorted(stage_map)}")
        stage_in, stage_out, meta = stage_map[stage]
        _emit_stage(out_dir, stage, stage_in, stage_out, meta)
        print(f"[OK] Wrote {stage} vectors to {out_dir / stage}")

    summary = {
        "input_len": len(data),
        "padded_len": len(padded),
        "depth": args.depth,
        "scrambler_enabled": not args.no_scramble,
        "seed": args.seed,
        "stages": stages,
    }
    with (out_dir / "vector_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary written to {out_dir / 'vector_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate HDL reference vectors from the golden model")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="UTF-8 text input")
    src.add_argument("--hex", help="Hex string input (spaces allowed)")
    src.add_argument("--infile", help="Binary input file")
    p.add_argument("--depth", type=int, default=2, choices=sorted(POSSIBLE_DEPTHS), help="Interleaver depth I")
    p.add_argument("--no-scramble", action="store_true", help="Disable scrambling for the generated vectors")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED, help="Scrambler seed (default matches model)")
    p.add_argument("--stage", action="append", choices=["rs", "interleaver", "scrambler", "conv", "diff", "qpsk", "rrc"],
                   help="Which pipeline stage(s) to emit (may be specified multiple times). Default: rs")
    p.add_argument("--out-dir", default="mpu/hdl/vectors", help="Destination directory for generated files")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return generate_vectors(args)


if __name__ == "__main__":
    raise SystemExit(main())
