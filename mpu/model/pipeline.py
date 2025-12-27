#!/usr/bin/env python3

# Golden model pipeline runner:
# input -> RS(255, 223) encoder -> byte wise interleaver (I=2) -> ouput

# examples:
#   python -m mpu.model.pipeline --text "HELLO WORLD"
#   python -m mpu.model.pipeline --hex "0011223344" --out out.bin
#   python -m mpu.model.pipeline --infile input.bin --save-figs pipeline


# Notes:
# - without padding, input length must be a multiple of 223 bytes
# - input is always zero-padded to multiples of 223 bytes


import argparse
import csv
import sys
from typing import List, Optional, Sequence, Tuple
import numpy as np

from mpu.model.reed_solomon import rs_encode, k as RS_K
from mpu.model.interleaver import interleave, deinterleave, CODEWORD_BYTES, POSSIBLE_DEPTHS
from mpu.model.scrambler import scramble_bits, descramble_bits, DEFAULT_SEED
from mpu.model.conv_encoder import conv_encode
from mpu.model.diff_encoder import diff_encode, diff_decode
from mpu.model.qpsk import qpsk_modulate_bytes_fixed, qpsk_demod_hard_fixed, pack_iq_le_bytes, FRAC_BITS_DEFAULT
from mpu.model.rrc import (
    rrc_filter,
    upsample,
    fir_filter,
    SAMPLES_PER_SYMBOL,
    FILTER_SPAN,
    ROLL_OFF,
    NUM_TAPS,
    RRC_COEFFS,
)
from mpu.model.helpers import add_awgn, bit_error_rate, hamming_distance_bytes, apply_cfo, apply_sto

CONV_L = 7
CONV_G1_OCT = 0o171
CONV_G2_OCT = 0o133
DATA_BITS_PER_BLOCK = CODEWORD_BYTES * 8
TAIL_BITS = CONV_L - 1
ENC_BITS_PER_BLOCK = 2 * (DATA_BITS_PER_BLOCK + TAIL_BITS)
ENC_BYTES_PER_BLOCK = (ENC_BITS_PER_BLOCK + 7) // 8

_TRELLIS = None
_RS_CODEC = None


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


def _get_decoder_components():
    """Lazily instantiate Trellis/RS codec used for sweeps."""
    global _TRELLIS, _RS_CODEC
    from commpy.channelcoding.convcode import Trellis
    import reedsolo

    if _TRELLIS is None:
        _TRELLIS = Trellis(
            np.array([CONV_L - 1]),
            np.array([[CONV_G1_OCT, CONV_G2_OCT]], dtype=int),
        )
    if _RS_CODEC is None:
        _RS_CODEC = reedsolo.RSCodec(
            32, nsize=255, c_exp=8, generator=2, fcr=1, prim=0x187
        )
    return _TRELLIS, _RS_CODEC


def _parse_value_list(spec: str, label: str) -> List[float]:
    """Parse comma/range based CLI specs (e.g., '0:10:2' or '0,5,10')."""
    if not spec:
        return []
    spec = spec.strip()
    values: List[float] = []
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"{label}: expected start:stop[:step]")
        start = float(parts[0])
        stop = float(parts[1])
        step = float(parts[2]) if len(parts) == 3 else 1.0
        if step <= 0:
            raise ValueError(f"{label}: step must be > 0 (got {step})")
        current = start
        # Include stop (within tolerance)
        while current <= stop + 1e-12:
            values.append(round(current, 6))
            current += step
    else:
        for token in spec.split(","):
            token = token.strip()
            if token:
                values.append(float(token))

    if not values:
        raise ValueError(f"{label}: no values parsed from '{spec}'")
    return values


def _block_error_statistics(
    reference: bytes,
    decoded: bytes,
    block_bytes: int = RS_K,
) -> dict:
    if len(reference) != len(decoded):
        raise ValueError("Reference and decoded payload must match in length.")

    total_bits = len(reference) * 8
    blocks = len(reference) // block_bytes if block_bytes else 0
    if blocks == 0:
        raise ValueError("Reference payload must contain at least one RS block.")

    total_bit_errors = hamming_distance_bytes(reference, decoded)
    total_byte_errors = sum(1 for a, b in zip(reference, decoded) if a != b)

    bit_errors_per_block = total_bit_errors / blocks
    byte_errors_per_block = total_byte_errors / blocks
    ber = bit_error_rate(reference, decoded)

    per_block_bits: List[int] = []
    per_block_bytes: List[int] = []
    for off in range(0, len(reference), block_bytes):
        ref_blk = reference[off:off + block_bytes]
        dec_blk = decoded[off:off + block_bytes]
        per_block_bits.append(hamming_distance_bytes(ref_blk, dec_blk))
        per_block_bytes.append(sum(1 for a, b in zip(ref_blk, dec_blk) if a != b))

    return {
        "ber": ber,
        "total_bit_errors": total_bit_errors,
        "total_byte_errors": total_byte_errors,
        "bit_errors_per_block": bit_errors_per_block,
        "byte_errors_per_block": byte_errors_per_block,
        "blocks": blocks,
        "per_block_bit_errors": per_block_bits,
        "per_block_byte_errors": per_block_bytes,
    }


def _decode_stream_for_metrics(
    qpsk_i_rx: np.ndarray,
    qpsk_q_rx: np.ndarray,
    num_symbols: int,
    args: argparse.Namespace,
) -> Tuple[bool, Optional[dict], Optional[str]]:
    """Lightweight decoder used for SNR sweeps (no pretty-printing)."""
    try:
        trellis, rs_codec = _get_decoder_components()
    except ImportError as e:
        raise
    except Exception as e:  # pragma: no cover - defensive
        return False, None, f"decoder init failed: {e}"

    try:
        from commpy.channelcoding.convcode import viterbi_decode

        # Use fixed point convolution for receiver
        i_matched, q_matched = rrc_filter(qpsk_i_rx, qpsk_q_rx)

        group_delay = (NUM_TAPS - 1) // 2
        total_delay = 2 * group_delay
        i_down = i_matched[total_delay::SAMPLES_PER_SYMBOL][:num_symbols]
        q_down = q_matched[total_delay::SAMPLES_PER_SYMBOL][:num_symbols]

        i_recovered = i_down
        q_recovered = q_down

        iq_recovered = np.empty(len(i_recovered) * 2, dtype=np.int16)
        iq_recovered[0::2] = i_recovered
        iq_recovered[1::2] = q_recovered

        diff_encoded_recovered = qpsk_demod_hard_fixed(iq_recovered, interleaved=True)
        de_diff = diff_decode(diff_encoded_recovered)

        def _bits_from_bytes(data: bytes) -> List[int]:
            bits: List[int] = []
            for b in data:
                for i in range(8):
                    bits.append((b >> (7 - i)) & 1)
            return bits

        def _bytes_from_bits(bits: Sequence[int]) -> bytes:
            out = bytearray()
            acc = 0
            cnt = 0
            for bit in bits:
                acc = (acc << 1) | (bit & 1)
                cnt += 1
                if cnt == 8:
                    out.append(acc)
                    acc = 0
                    cnt = 0
            if cnt:
                out.append(acc << (8 - cnt))
            return bytes(out)

        viterbi_out_bits: List[int] = []
        if len(de_diff) % ENC_BYTES_PER_BLOCK != 0:
            return False, None, "viterbi input length mismatch"

        for off in range(0, len(de_diff), ENC_BYTES_PER_BLOCK):
            blk = de_diff[off:off + ENC_BYTES_PER_BLOCK]
            bits_full = _bits_from_bytes(blk)
            bits = bits_full[:ENC_BITS_PER_BLOCK]
            dec = viterbi_decode(
                np.array(bits, dtype=float),
                trellis,
                tb_depth=5 * (CONV_L - 1),
                decoding_type="hard",
            )
            viterbi_out_bits.extend(int(b) for b in dec[:DATA_BITS_PER_BLOCK])

        viterbi_bytes = _bytes_from_bits(viterbi_out_bits)
        descrambled = (
            viterbi_bytes
            if args.no_scramble
            else descramble_bits(viterbi_bytes, seed=args.seed)
        )
        deintl = deinterleave(descrambled, args.depth)

        if len(deintl) % CODEWORD_BYTES != 0:
            return False, None, "deinterleaver output misaligned"

        restored = bytearray()
        for off in range(0, len(deintl), CODEWORD_BYTES):
            cw = deintl[off:off + CODEWORD_BYTES]
            try:
                msg, _, _ = rs_codec.decode(cw)
                restored.extend(msg)
            except Exception:
                # RS failed, use systematic part
                restored.extend(cw[:223])

        return True, {
            "restored": bytes(restored),
            "deinterleaved": bytes(deintl),
        }, None
    except Exception as exc:  # pragma: no cover - best effort
        return False, None, str(exc)


def _run_noise_sweep(
    mode: str,
    values: Sequence[float],
    qpsk_i_rrc: np.ndarray,
    qpsk_q_rrc: np.ndarray,
    qpsk_i_symbols: np.ndarray,
    reference_payload: bytes,
    args: argparse.Namespace,
) -> List[dict]:
    """Evaluate BER/packet-error stats over a list of noise/SNR values."""
    results: List[dict] = []
    base_seed = args.noise_seed or 0
    blocks = len(reference_payload) // RS_K
    max_bit_err = RS_K * 8
    max_byte_err = RS_K

    for idx, value in enumerate(values):
        seed = base_seed + idx
        if mode == "snr":
            chan_i, chan_q, noise_metrics = add_awgn(
                qpsk_i_rrc,
                qpsk_q_rrc,
                snr_db=value,
                noise_std=None,
                seed=seed,
            )
        else:
            chan_i, chan_q, noise_metrics = add_awgn(
                qpsk_i_rrc,
                qpsk_q_rrc,
                snr_db=None,
                noise_std=value,
                seed=seed,
            )

        ok, decode_artifacts, error_reason = _decode_stream_for_metrics(
            chan_i,
            chan_q,
            len(qpsk_i_symbols),
            args,
        )

        if ok and decode_artifacts is not None:
            restored = decode_artifacts["restored"]
            stats = _block_error_statistics(reference_payload, restored)
            fail_reason = ""

            deintl_bytes = decode_artifacts.get("deinterleaved")
            if deintl_bytes and len(deintl_bytes) % CODEWORD_BYTES == 0:
                no_rs = bytearray()
                for off in range(0, len(deintl_bytes), CODEWORD_BYTES):
                    block = deintl_bytes[off:off + CODEWORD_BYTES]
                    no_rs.extend(block[:RS_K])
                stats_no_rs = _block_error_statistics(reference_payload, bytes(no_rs))
            else:
                stats_no_rs = stats.copy()
        else:
            stats = {
                "ber": getattr(args, "sweep_max_ber", 1.0),
                "total_bit_errors": max_bit_err * blocks,
                "total_byte_errors": max_byte_err * blocks,
                "bit_errors_per_block": float(max_bit_err),
                "byte_errors_per_block": float(max_byte_err),
                "blocks": blocks,
                "per_block_bit_errors": [max_bit_err] * blocks,
                "per_block_byte_errors": [max_byte_err] * blocks,
            }
            fail_reason = error_reason or "decoder failure"
            stats_no_rs = stats

        stats.update(
            {
                "mode": mode,
                "requested_value": value,
                "measured_snr_db": noise_metrics.get("measured_snr_db"),
                "noise_std": noise_metrics.get("noise_std"),
                "signal_power": noise_metrics.get("signal_power"),
                "seed": noise_metrics.get("seed"),
                "success": ok,
                "fail_reason": fail_reason,
                "ber_no_rs": stats_no_rs.get("ber", stats["ber"]),
                "bit_errors_per_block_no_rs": stats_no_rs.get("bit_errors_per_block", stats["bit_errors_per_block"]),
                "byte_errors_per_block_no_rs": stats_no_rs.get("byte_errors_per_block", stats["byte_errors_per_block"]),
            }
        )
        results.append(stats)

    return results


def _plot_sweep_results(
    results: Sequence[dict],
    args: argparse.Namespace,
) -> None:
    if not results:
        return

    import matplotlib.pyplot as plt

    results_sorted = sorted(
        results,
        key=lambda r: (float("-inf") if r.get("measured_snr_db") is None else r["measured_snr_db"])
    )

    snrs = [r.get("measured_snr_db", 0.0) for r in results_sorted]
    bers = [r.get("ber", 1.0) for r in results_sorted]
    bers_no_rs = [r.get("ber_no_rs", 1.0) for r in results_sorted]
    bit_errs = [r.get("bit_errors_per_block", 0.0) for r in results_sorted]
    bit_errs_no_rs = [r.get("bit_errors_per_block_no_rs", 0.0) for r in results_sorted]
    byte_errs = [r.get("byte_errors_per_block", 0.0) for r in results_sorted]
    byte_errs_no_rs = [r.get("byte_errors_per_block_no_rs", 0.0) for r in results_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_bers = [max(b, 1e-12) for b in bers]
    plot_bers_no_rs = [max(b, 1e-12) for b in bers_no_rs]
    axes[0].semilogy(snrs, plot_bers, marker="o", linewidth=1.5, label="With RS/CC")
    axes[0].semilogy(snrs, plot_bers_no_rs, marker="s", linestyle="--", linewidth=1.2,
                     label="Without RS correction")
    axes[0].set_xlabel("Measured SNR (dB)")
    axes[0].set_ylabel("Bit Error Rate (BER)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].set_title("BER vs. SNR")
    axes[0].legend()

    axes[1].plot(snrs, bit_errs, marker="o", label="Bit errors / 223B block")
    axes[1].plot(snrs, byte_errs, marker="s", label="Byte errors / 223B block")
    axes[1].set_xlabel("Measured SNR (dB)")
    axes[1].set_ylabel("Errors per RS(255,223) payload")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_title("Per-Block Errors vs. SNR")

    fig.suptitle("BER / Packet Error Sweep", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    prefix = getattr(args, "sweep_save", None) or getattr(args, "save_figs", None) or "ber_sweep"
    png_path = f"{prefix}_ber.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = f"{prefix}_ber.csv"
    fieldnames = [
        "mode",
        "requested_value",
        "measured_snr_db",
        "noise_std",
        "signal_power",
        "ber",
        "ber_no_rs",
        "bit_errors_per_block",
        "bit_errors_per_block_no_rs",
        "byte_errors_per_block",
        "byte_errors_per_block_no_rs",
        "total_bit_errors",
        "total_byte_errors",
        "blocks",
        "success",
        "fail_reason",
        "seed",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_sorted:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"\nSaved BER sweep plot -> {png_path}")
    print(f"Saved BER sweep data  -> {csv_path}")


def _print_sweep_summary(results: Sequence[dict], label: str) -> None:
    if not results:
        return
    results_sorted = sorted(
        results,
        key=lambda r: (float("-inf") if r.get("measured_snr_db") is None else r["measured_snr_db"])
    )
    print(f"\n{label} sweep summary (per RS(255,223) block):")
    print("  SNR[dB]\tBER(FEC)\tBER(no RS)\tbit errs\tbit errs(no RS)\tstatus")
    for row in results_sorted:
        snr = row.get("measured_snr_db")
        ber = row.get("ber", 1.0)
        ber_no_rs = row.get("ber_no_rs", ber)
        bit_err = row.get("bit_errors_per_block", 0.0)
        bit_err_no_rs = row.get("bit_errors_per_block_no_rs", bit_err)
        byte_err = row.get("byte_errors_per_block", 0.0)
        status = "OK" if row.get("success") else f"FAIL ({row.get('fail_reason')})"
        snr_str = f"{snr:7.2f}" if snr is not None else "   n/a"
        print(f"  {snr_str}\t{ber:>.3e}\t{ber_no_rs:>.3e}\t{bit_err:>9.2f}\t{bit_err_no_rs:>9.2f}\t{status}")


def _maybe_run_sweeps(
    args: argparse.Namespace,
    qpsk_i_rrc: np.ndarray,
    qpsk_q_rrc: np.ndarray,
    qpsk_i_symbols: np.ndarray,
    reference_payload: bytes,
) -> None:
    if not getattr(args, "snr_sweep", None) and not getattr(args, "noise_std_sweep", None):
        return

    sweep_specs: List[Tuple[str, List[float]]] = []
    if args.snr_sweep:
        try:
            sweep_specs.append(("snr", _parse_value_list(args.snr_sweep, "--snr-sweep")))
        except ValueError as exc:
            print(f"\nError parsing --snr-sweep: {exc}", file=sys.stderr)
    if args.noise_std_sweep:
        try:
            sweep_specs.append(("noise_std", _parse_value_list(args.noise_std_sweep, "--noise-std-sweep")))
        except ValueError as exc:
            print(f"\nError parsing --noise-std-sweep: {exc}", file=sys.stderr)

    for mode, values in sweep_specs:
        if not values:
            continue
        label = "SNR" if mode == "snr" else "Noise σ"
        print(f"\n=== Running {label} sweep ({len(values)} points) ===")
        print(f"Values: {values}\n")
        try:
            results = _run_noise_sweep(
                mode,
                values,
                qpsk_i_rrc,
                qpsk_q_rrc,
                qpsk_i_symbols,
                reference_payload,
                args,
            )
        except ImportError as e:
            print(f"\nCannot run {label} sweep - decoder dependency missing: {e}", file=sys.stderr)
            break

        _print_sweep_summary(results, label)
        _plot_sweep_results(results, args)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Run RS(255,223) + interleaver golden pipeline.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="UTF-8 text input.")
    src.add_argument("--hex", help="Hex string input (spaces allowed).")
    src.add_argument("--infile", help="Binary input file.")
    p.add_argument("--depth", type=int, default=2, choices=sorted(POSSIBLE_DEPTHS),
                   help=f"Interleaver depth I (default: 2)")
    p.add_argument("--no-scramble", action="store_true",
                   help="Disable scrambler (enabled by default).")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED,
                   help="Override scrambler seed (int, e.g. 0b1011101).")
    p.add_argument("--out", help="Write pipeline output (binary) to this file.")
    p.add_argument("--save-figs", metavar="PREFIX",
                   help="Save visualization figures with this prefix (e.g., 'pipeline' -> 'pipeline_1_rs.png').")

    noise_group = p.add_argument_group("Channel Impairments")
    noise_level = noise_group.add_mutually_exclusive_group()
    noise_level.add_argument("--snr-db", type=float,
                            help="Add complex AWGN with this target SNR (dB) referenced to the TX RRC output power.")
    noise_level.add_argument("--noise-std", type=float,
                             help="Add AWGN using this per-dimension standard deviation (same units as the I/Q samples).")
    noise_group.add_argument("--noise-seed", type=int, default=0,
                             help="Seed for AWGN RNG (default: 0).")
    noise_group.add_argument("--cfo", type=float, default=0.0,
                             help="Carrier Frequency Offset (normalized to Fs). E.g. 1e-4.")
    noise_group.add_argument("--sto", type=float, default=0.0,
                             help="Symbol Timing Offset (in samples). E.g. 0.5.")

    sweep_group = p.add_argument_group("BER Sweeps / Performance Curves")
    sweep_group.add_argument("--snr-sweep", type=str,
                             help="Evaluate multiple SNR points (format: 'start:stop:step' or comma list).")
    sweep_group.add_argument("--noise-std-sweep", type=str,
                             help="Evaluate multiple noise σ values (format: 'start:stop:step' or comma list).")
    sweep_group.add_argument("--sweep-save", metavar="PREFIX",
                             help="Prefix for BER sweep PNG/CSV outputs (defaults to --save-figs or 'ber_sweep').")
    sweep_group.add_argument("--sweep-only", action="store_true",
                             help="Skip stage visualizations and only run sweeps.")
    sweep_group.add_argument("--sweep-max-ber", type=float, default=1.0,
                             help="BER value to log when the decoder fails (default: 1.0).")

    args = p.parse_args(argv)

    data = _read_input(args)

    # Always pad to RS_K
    data_for_rs = _pad_to_block(data, RS_K)

    # RS encode -> concatenated 255B codewords
    rs_out = rs_encode(data_for_rs)

    # Interleave per codeword with depth I
    interleaved = interleave(rs_out, args.depth)

    # Scramble (enabled by default, disable with --no-scramble)
    if args.no_scramble:
        scrambled = interleaved
    else:
        scrambled = scramble_bits(interleaved, seed=args.seed)

    # Convolutional encoder
    convolved = conv_encode(scrambled)

    # Differential encoder
    diff_encoded = diff_encode(convolved)

    # QPSK + RRC (always enabled)
    iq = qpsk_modulate_bytes_fixed(diff_encoded, frac_bits=15, interleaved=True)  # np.int16 [I0,Q0,...]
    # Extract separate I and Q streams
    qpsk_i = iq[0::2]  # I samples: [I0, I1, I2, ...]
    qpsk_q = iq[1::2]  # Q samples: [Q0, Q1, Q2, ...]
    
    # Upsample by SAMPLES_PER_SYMBOL (insert zeros) - keep as int16
    qpsk_i_up = upsample(qpsk_i, SAMPLES_PER_SYMBOL)
    qpsk_q_up = upsample(qpsk_q, SAMPLES_PER_SYMBOL)
    
    # Pad the upsampled signal to compensate for cascade filter delay
    # We need 64 extra samples at the end to ensure we can extract all symbols
    group_delay_samples = (NUM_TAPS - 1) // 2
    total_delay_samples = 2 * group_delay_samples  # 64 samples
    
    # Pad at the END with zeros
    qpsk_i_up_padded = np.concatenate([qpsk_i_up, np.zeros(total_delay_samples, dtype=np.int16)])
    qpsk_q_up_padded = np.concatenate([qpsk_q_up, np.zeros(total_delay_samples, dtype=np.int16)])
    
    # TX RRC filter
    qpsk_i_filtered, qpsk_q_filtered = rrc_filter(qpsk_i_up_padded, qpsk_q_up_padded)
    
    # Store for display (TX output before channel)
    qpsk_i_rrc = qpsk_i_filtered
    qpsk_q_rrc = qpsk_q_filtered

    # Apply Channel Impairments
    # 1. CFO
    qpsk_i_imp, qpsk_q_imp = apply_cfo(qpsk_i_rrc, qpsk_q_rrc, args.cfo)
    
    # 2. STO
    qpsk_i_imp, qpsk_q_imp = apply_sto(qpsk_i_imp, qpsk_q_imp, args.sto)

    # 3. AWGN
    qpsk_i_channel, qpsk_q_channel, noise_metrics = add_awgn(
        qpsk_i_imp,
        qpsk_q_imp,
        snr_db=args.snr_db,
        noise_std=args.noise_std,
        seed=args.noise_seed,
    )

    # Convert back to int16 (with saturation) for DAC output
    qpsk_i_int = np.clip(qpsk_i_filtered, -32768, 32767).astype(np.int16)
    qpsk_q_int = np.clip(qpsk_q_filtered, -32768, 32767).astype(np.int16)
    
    # Re-interleave for output
    iq_rrc = np.empty(len(qpsk_i_int) + len(qpsk_q_int), dtype=np.int16)
    iq_rrc[0::2] = qpsk_i_int
    iq_rrc[1::2] = qpsk_q_int
    iq_channel = np.empty(len(qpsk_i_channel) + len(qpsk_q_channel), dtype=np.int16)
    iq_channel[0::2] = qpsk_i_channel
    iq_channel[1::2] = qpsk_q_channel
    
    final_out = pack_iq_le_bytes(iq_rrc)  # little-endian int16 stream for DAC/file

    # Output to file if specified
    if args.out:
        with open(args.out, "wb") as f:
            f.write(final_out)

    if not args.sweep_only:
        # Generate visualizations
        try:
            from mpu.model.pipeline_visualizer import generate_visualizations
            generate_visualizations(
                data_for_rs,
                rs_out,
                interleaved,
                scrambled,
                convolved,
                diff_encoded,
                iq,
                qpsk_i,
                qpsk_q,
                qpsk_i_rrc,
                qpsk_q_rrc,
                qpsk_i_channel,
                qpsk_q_channel,
                noise_metrics,
                args,
            )
        except ImportError as e:
            print(f"\nWarning: Visualization unavailable - {e}", file=sys.stderr)
            print("Install matplotlib and scipy to enable visualizations.", file=sys.stderr)
        except Exception as e:
            print(f"\nError creating visualizations: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    # Print output stages
    size = len(data)
    print(f"input (size = {size}) =")
    print(" ".join(f"{b:02X}" for b in data))
    print()

    size = len(rs_out)
    print(f"RS output (size = {size}) =")
    print(" ".join(f"{b:02X}" for b in rs_out))
    print()

    size = len(interleaved)
    print(f"interleaver output (I={args.depth}) (size = {size}) =")
    print(" ".join(f"{b:02X}" for b in interleaved))
    print()

    if not args.no_scramble:
        size = len(scrambled)
        print(f"scrambler output (poly=x^7+x^4+1, seed={args.seed:#09b}) (size = {size}) =")
        print(" ".join(f"{b:02X}" for b in scrambled))
        print()

    size = len(convolved)
    print(f"convolutional encoder output (size = {size}) =")
    print(" ".join(f"{b:02X}" for b in convolved))
    print()

    size = len(diff_encoded)
    print(f"diff encoder output (size = {size}) =")
    print(" ".join(f"{b:02X}" for b in diff_encoded))
    print()

    # Display separate I and Q streams
    print(f"qpsk i (size = {len(qpsk_i)}) =")
    print(" ".join(f"{int(i):+6d}" for i in (qpsk_i[:20]/23170)))
    if len(qpsk_i) > 20:
        print("  ...")
    print()
    
    print(f"qpsk q (size = {len(qpsk_q)}) =")
    print(" ".join(f"{int(q):+6d}" for q in (qpsk_q[:20]/23170)))
    if len(qpsk_q) > 20:
        print("  ...")
    print()

    # Display RRC filter info and filtered output
    from mpu.model.rrc import RRC_COEFFS
    
    print(f"=== RRC Filter ===")
    print(f"Filter parameters: {SAMPLES_PER_SYMBOL} samples/symbol, span={FILTER_SPAN}, beta={ROLL_OFF}")
    print(f"Number of taps: {len(RRC_COEFFS)}")
    print(f"Filter coefficients (first 10 and last 10 of {len(RRC_COEFFS)}):")
    print(" ".join(f"{c:+.6f}" for c in RRC_COEFFS[:10]))
    print("  ...")
    print(" ".join(f"{c:+.6f}" for c in RRC_COEFFS[-10:]))
    print()
    
    print(f"rrc i (after upsampling and filtering) (size = {len(qpsk_i_rrc)}) =")
    # Show first 40 samples (5 symbols worth at 8 sps)
    display_samples = min(40, len(qpsk_i_rrc))
    print(" ".join(f"{int(i):+6d}" for i in qpsk_i_rrc[:display_samples]))
    if len(qpsk_i_rrc) > display_samples:
        print("  ...")
    print()
    
    print(f"rrc q (after upsampling and filtering) (size = {len(qpsk_q_rrc)}) =")
    print(" ".join(f"{int(q):+6d}" for q in qpsk_q_rrc[:display_samples]))
    if len(qpsk_q_rrc) > display_samples:
        print("  ...")
    print()

    print("=== Channel / Noise Model ===")
    if args.cfo != 0.0:
        print(f"CFO enabled: {args.cfo} Fs")
    if args.sto != 0.0:
        print(f"STO enabled: {args.sto} samples")
        
    if noise_metrics.get("enabled"):
        target = noise_metrics.get("target_snr_db")
        target_str = f"target SNR = {target:.2f} dB" if target is not None else f"noise σ = {noise_metrics['noise_std']:.2f}"
        print(f"AWGN enabled ({target_str}, seed={noise_metrics.get('seed')})")
        print(f"Measured SNR = {noise_metrics.get('measured_snr_db'):.2f} dB")
    else:
        print("AWGN disabled")
    print(f"noisy rrc i (size = {len(qpsk_i_channel)}) =")
    print(" ".join(f"{int(i):+6d}" for i in qpsk_i_channel[:display_samples]))
    if len(qpsk_i_channel) > display_samples:
        print("  ...")
    print()
    print(f"noisy rrc q (size = {len(qpsk_q_channel)}) =")
    print(" ".join(f"{int(q):+6d}" for q in qpsk_q_channel[:display_samples]))
    if len(qpsk_q_channel) > display_samples:
        print("  ...")
    print()

    print("=== Reverse Path (Decoding) ===")
    print()

    # ===== Oracle reverse (QPSK hard demod -> Diff decoder -> Viterbi -> descrambler -> deinterleaver -> RS decode) =====
    try:
        from commpy.channelcoding.convcode import Trellis, viterbi_decode
        import reedsolo

        # --- helpers (bit packing) ---
        def _bits_from_bytes(data: bytes) -> list[int]:
            out = []
            for b in data:
                for i in range(8):
                    out.append((b >> (7 - i)) & 1)
            return out

        def _bytes_from_bits(bits: list[int]) -> bytes:
            out = bytearray()
            acc = 0
            n = 0
            for bit in bits:
                acc = (acc << 1) | (bit & 1)
                n += 1
                if n == 8:
                    out.append(acc)
                    acc = 0
                    n = 0
            if n:
                out.append(acc << (8 - n))
            return bytes(out)

        # --- CC (171,133)o, L=7, terminated per 255B block ---
        trellis = Trellis(np.array([CONV_L-1]), np.array([[CONV_G1_OCT, CONV_G2_OCT]], dtype=int))

        # Apply RRC matched filtering and downsample
        print("=== RRC Receiver (Matched Filter + Downsampling) ===")
        print()
        
        from mpu.model.rrc import RRC_COEFFS
        
        # Apply matched filter
        i_matched, q_matched = rrc_filter(qpsk_i_channel, qpsk_q_channel)
        
        print(f"receiver matched filter output i (size = {len(i_matched)}):")
        print(" ".join(f"{int(i):+6d}" for i in i_matched[:40]))
        if len(i_matched) > 40:
            print("  ...")
        print()
        
        print(f"receiver matched filter output q (size = {len(q_matched)}):")
        print(" ".join(f"{int(q):+6d}" for q in q_matched[:40]))
        if len(q_matched) > 40:
            print("  ...")
        print()
        
        # Group delay through cascaded RRC filters
        group_delay_samples = (NUM_TAPS - 1) // 2  # Per filter, in upsampled domain
        total_delay_samples = 2 * group_delay_samples  # TX + RX cascade
        
        print(f"Group delay (per filter): {group_delay_samples} samples")
        print(f"Total delay (TX + RX): {total_delay_samples} samples") 
        print(f"Downsampling: every {SAMPLES_PER_SYMBOL}th sample")
        print()
        
        # Downsample starting at the cascade delay point
        # This gets us the ISI-free sampling instants
        i_downsampled = i_matched[total_delay_samples::SAMPLES_PER_SYMBOL]
        q_downsampled = q_matched[total_delay_samples::SAMPLES_PER_SYMBOL]
        
        # Trim to the expected number of symbols (discard tail)
        num_symbols = len(qpsk_i)
        i_downsampled = i_downsampled[:num_symbols]
        q_downsampled = q_downsampled[:num_symbols]
        
        print(f"receiver downsampled i (size = {len(i_downsampled)}):")
        print(" ".join(f"{int(i):+6d}" for i in i_downsampled[:20]))
        if len(i_downsampled) > 20:
            print("  ...")
        print()
        
        print(f"receiver downsampled q (size = {len(q_downsampled)}):")
        print(" ".join(f"{int(q):+6d}" for q in q_downsampled[:20]))
        if len(q_downsampled) > 20:
            print("  ...")
        print()
        
        # Convert to int16 and clip
        i_recovered = np.clip(i_downsampled, -32768, 32767).astype(np.int16)
        q_recovered = np.clip(q_downsampled, -32768, 32767).astype(np.int16)
        
        # Re-interleave
        iq_recovered = np.empty(len(i_recovered) + len(q_recovered), dtype=np.int16)
        iq_recovered[0::2] = i_recovered
        iq_recovered[1::2] = q_recovered
        
        # Hard demodulate
        diff_encoded_recovered = qpsk_demod_hard_fixed(iq_recovered, interleaved=True)
        
        print(f"receiver qpsk hard demod output (after RRC receiver) (size = {len(diff_encoded_recovered)}) =")
        print(" ".join(f"{b:02X}" for b in diff_encoded_recovered[:64]))
        if len(diff_encoded_recovered) > 64:
            print("  ...")
        print()
        print()

        # 1) differential decode
        de_diff = diff_decode(diff_encoded_recovered)

        # 2) slice encoded stream per block in BYTES, drop pad bits, then Viterbi (hard)
        if len(de_diff) % ENC_BYTES_PER_BLOCK != 0:
            raise RuntimeError(f"Encoded byte stream length {len(de_diff)} not multiple of per-block {ENC_BYTES_PER_BLOCK} bytes.")
        viterbi_out_bits: list[int] = []
        for off in range(0, len(de_diff), ENC_BYTES_PER_BLOCK):
            blk_bytes = de_diff[off:off + ENC_BYTES_PER_BLOCK]
            bits_full = _bits_from_bytes(blk_bytes)               # 4096 bits
            bits = bits_full[:ENC_BITS_PER_BLOCK]                 # drop 4 pad bits -> 4092
            dec = viterbi_decode(np.array(bits, dtype=float),
                                trellis,
                                tb_depth=5*(CONV_L-1),
                                decoding_type='hard')
            # Keep only the 2040 data bits
            viterbi_out_bits.extend(int(b) for b in dec[:DATA_BITS_PER_BLOCK])

        viterbi_bytes = _bytes_from_bits(viterbi_out_bits)
        print(f"viterbi decoder output (size = {len(viterbi_bytes)}) =")
        print(" ".join(f"{b:02X}" for b in viterbi_bytes))
        print()

        # 3) descrambler
        descrambled = viterbi_bytes if args.no_scramble else descramble_bits(viterbi_bytes, seed=args.seed)
        print(f"descrambler output (size = {len(descrambled)}) =")
        print(" ".join(f"{b:02X}" for b in descrambled))
        print()

        # 4) deinterleaver (per 255B block, depth I)
        deintl = deinterleave(descrambled, args.depth)
        print(f"deinterleaver output (I={args.depth}) (size = {len(deintl)}) =")
        print(" ".join(f"{b:02X}" for b in deintl))
        print()

        # 5) RS reverse via reedsolo oracle (RS(255,223), prim=0x187, alpha=2, fcr=1)
        RS = reedsolo.RSCodec(32, nsize=255, c_exp=8, generator=2, fcr=1, prim=0x187)
        if len(deintl) % CODEWORD_BYTES != 0:
            raise RuntimeError(f"deinterleaver output {len(deintl)} not multiple of {CODEWORD_BYTES}")
        restored = bytearray()
        rs_failures = 0
        rs_failure_examples: list[str] = []
        for off in range(0, len(deintl), CODEWORD_BYTES):
            cw = deintl[off:off + CODEWORD_BYTES]
            try:
                msg, _, _ = RS.decode(cw)  # decode returns (message, ecc, errata)
                restored.extend(msg)
            except reedsolo.ReedSolomonError as exc:
                rs_failures += 1
                if len(rs_failure_examples) < 3:
                    rs_failure_examples.append(str(exc))
                # fall back to systematic portion so downstream steps still run
                restored.extend(cw[:RS_K])

        print(f"reverse RS output (size = {len(restored)}) =")
        print(" ".join(f"{b:02X}" for b in restored))
        print()
        null_char = chr(0)
        restored_text = restored.decode('utf-8', errors='replace').replace(null_char, '\\x00')
        print(f"restored ascii = '{restored_text}'")
        if rs_failures:
            print(f"\nWarning: RS decode failed on {rs_failures} block(s); "
                  "restored payload contains uncorrected bytes.")
            if rs_failure_examples:
                print("Example RS error(s): " + " | ".join(rs_failure_examples))
    
    except ImportError as e:
        print(f"\nWarning: Decoder unavailable - {e}", file=sys.stderr)
        print("Install commpy and reedsolo to enable decoding verification.", file=sys.stderr)

    _maybe_run_sweeps(
        args,
        qpsk_i_imp,
        qpsk_q_imp,
        qpsk_i,
        data_for_rs,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
