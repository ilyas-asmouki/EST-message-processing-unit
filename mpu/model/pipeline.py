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
import numpy as np

from mpu.model.reed_solomon import rs_encode, rs_syndromes, k as RS_K, n as RS_N
from mpu.model.interleaver import interleave, deinterleave, CODEWORD_BYTES, POSSIBLE_DEPTHS
from mpu.model.scrambler import scramble_bits, descramble_bits ,DEFAULT_SEED
from mpu.model.conv_encoder import conv_encode
from mpu.model.diff_encoder import diff_encode, diff_decode
from mpu.model.qpsk import qpsk_modulate_bytes_fixed, qpsk_demod_hard_fixed, pack_iq_le_bytes, FRAC_BITS_DEFAULT
from mpu.model.rrc import rrc_filter, upsample, fir_filter, SAMPLES_PER_SYMBOL, FILTER_SPAN, ROLL_OFF, NUM_TAPS


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
                   help="Print space-separated bytes of pipeline output.")
    p.add_argument("--no-print", dest="do_print", action="store_false",
                   help="Suppress printing.")
    p.set_defaults(do_print=True)
    p.add_argument("--hexout", action="store_true",
                   help="Print hex of pipeline output.")
    p.add_argument("--check", action="store_true",
                   help="After RS encode, verify each 255B codeword has zero syndromes.")
    p.add_argument("--show", action="store_true",
                   help="Show brief sizes & block counts.")
    p.add_argument("--qpsk-fixed", action="store_true",
                   help="After diff encode, QPSK modulate to int16 I/Q (Q1.frac), then include hard-demod in reverse checks.")
    p.set_defaults(qpsk_fixed=True)
    p.add_argument("--frac-bits", type=int, default=FRAC_BITS_DEFAULT,
                   help="Fractional bits for Q1.frac fixed-point QPSK (default: 15).")
    p.add_argument("--rrc", action="store_true",
                   help="Apply Root Raised Cosine pulse shaping after QPSK (with upsampling).")
    p.add_argument("--no-rrc", dest="rrc", action="store_false",
                   help="Disable RRC pulse shaping.")
    p.set_defaults(rrc=True)
    p.add_argument("--iq-out", help="If --qpsk-fixed, write interleaved int16 LE [I0,Q0,I1,Q1,...] to this file.")
    p.add_argument("--iq-print", type=int, default=0,
                   help="If --qpsk-fixed, print first N I/Q pairs as integers.")

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
        if args.rrc:
            print(f"RRC: samples/symbol={SAMPLES_PER_SYMBOL}, span={FILTER_SPAN}, beta={ROLL_OFF}")

    # RS encode -> concatenated 255B codewords
    rs_out = rs_encode(data_for_rs)

    if args.check:
        # Check each codeword's syndromes are all zero
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

    # Convolutional encoder
    convolved = conv_encode(scrambled)

    # Differential encoder
    diff_encoded = diff_encode(convolved)

    # QPSK + RRC
    iq = None  # will hold np.int16 interleaved I/Q if QPSK is active
    qpsk_i = None  # separate I stream
    qpsk_q = None  # separate Q stream
    qpsk_i_rrc = None  # RRC filtered I stream (if RRC enabled)
    qpsk_q_rrc = None  # RRC filtered Q stream (if RRC enabled)
    
    if args.qpsk_fixed:
        iq = qpsk_modulate_bytes_fixed(diff_encoded, frac_bits=args.frac_bits, interleaved=True)  # np.int16 [I0,Q0,...]
        # Extract separate I and Q streams
        qpsk_i = iq[0::2]  # I samples: [I0, I1, I2, ...]
        qpsk_q = iq[1::2]  # Q samples: [Q0, Q1, Q2, ...]
        
        # Apply RRC pulse shaping if enabled
        if args.rrc:
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
            
            # Store for display
            qpsk_i_rrc = qpsk_i_filtered
            qpsk_q_rrc = qpsk_q_filtered
            
            # Convert back to int16 (with saturation)
            qpsk_i_int = np.clip(qpsk_i_filtered, -32768, 32767).astype(np.int16)
            qpsk_q_int = np.clip(qpsk_q_filtered, -32768, 32767).astype(np.int16)
            
            # Re-interleave for output
            iq_rrc = np.empty(len(qpsk_i_int) + len(qpsk_q_int), dtype=np.int16)
            iq_rrc[0::2] = qpsk_i_int
            iq_rrc[1::2] = qpsk_q_int
            
            final_out = pack_iq_le_bytes(iq_rrc)  # little-endian int16 stream for DAC/file
        else:
            final_out = pack_iq_le_bytes(iq)  # little-endian int16 stream for DAC/file

        if args.iq_print > 0:
            N = min(args.iq_print, iq.size // 2)
            print(f"qpsk int16 (Q1.{args.frac_bits}) first {N} I/Q pairs:")
            for k in range(N):
                ii = int(iq[2*k]); qq = int(iq[2*k+1])
                print(f"{k:04d}: I={ii:+6d}  Q={qq:+6d}")
            print()

        if args.iq_out:
            with open(args.iq_out, "wb") as f:
                f.write(final_out)

        if args.show:
            sym_count = iq.size // 2
            print(f"QPSK symbols: {sym_count} (int16 Q1.{args.frac_bits}, interleaved I/Q)")
            if args.rrc:
                rrc_samples = len(qpsk_i_rrc)
                print(f"RRC output: {rrc_samples} samples ({rrc_samples // SAMPLES_PER_SYMBOL} symbols × {SAMPLES_PER_SYMBOL} sps)")

    else:
        final_out = diff_encoded

    # Output
    if args.out:
        with open(args.out, "wb") as f:
            f.write(final_out)

    # Output stages
    if args.do_print:
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

        # Display separate I and Q streams if QPSK is enabled
        if args.qpsk_fixed and qpsk_i is not None and qpsk_q is not None:
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
        
        # Display RRC filter info and filtered output if enabled
        if args.rrc:
            from mpu.model.rrc import RRC_COEFFS
            
            print(f"=== RRC Filter ===")
            print(f"Filter parameters: {SAMPLES_PER_SYMBOL} samples/symbol, span={FILTER_SPAN}, beta={ROLL_OFF}")
            print(f"Number of taps: {len(RRC_COEFFS)}")
            print(f"Filter coefficients (first 10 and last 10 of {len(RRC_COEFFS)}):")
            print(" ".join(f"{c:+.6f}" for c in RRC_COEFFS[:10]))
            print("  ...")
            print(" ".join(f"{c:+.6f}" for c in RRC_COEFFS[-10:]))
            print()
            
            if qpsk_i_rrc is not None and qpsk_q_rrc is not None:
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
        
        print("=== Reverse Path (Decoding) ===")
        print()

        # ===== Oracle reverse (QPSK hard demod if used -> Diff decoder -> Viterbi -> descrambler -> deinterleaver -> RS decode) =====
        # We can run this in both modes. If QPSK was active, start from the hard-demodulated bytes.
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
        L = 7
        G1_OCT = 0o171
        G2_OCT = 0o133
        DATA_BITS_PER_BLOCK = CODEWORD_BYTES * 8            # 255*8 = 2040
        TAIL_BITS = L - 1                                   # 6
        ENC_BITS_PER_BLOCK = 2 * (DATA_BITS_PER_BLOCK + TAIL_BITS)   # 4092
        ENC_BYTES_PER_BLOCK = (ENC_BITS_PER_BLOCK + 7) // 8          # 512
        PAD_BITS = ENC_BYTES_PER_BLOCK * 8 - ENC_BITS_PER_BLOCK      # 4

        trellis = Trellis(np.array([L-1]), np.array([[G1_OCT, G2_OCT]], dtype=int))

        # 0) If QPSK was used, hard-demod the IQ back to diff-encoded bytes
        # If RRC was used, we need to apply matched filtering and downsample first
        if iq is not None and args.rrc:
            print("=== RRC Receiver (Matched Filter + Downsampling) ===")
            print()
            
            from mpu.model.rrc import RRC_COEFFS
            
            # Apply matched filter (same RRC filter)
            i_matched = fir_filter(qpsk_i_rrc, RRC_COEFFS)
            q_matched = fir_filter(qpsk_q_rrc, RRC_COEFFS)
            
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
            start_bytes = diff_encoded_recovered
            
        elif iq is not None and not args.rrc:
            # Original path without RRC
            diff_encoded_recovered = qpsk_demod_hard_fixed(iq, interleaved=True)
            print(f"qpsk hard demod output (size = {len(diff_encoded_recovered)}) =")
            print(" ".join(f"{b:02X}" for b in diff_encoded_recovered))
            print()
            start_bytes = diff_encoded_recovered
        else:
            start_bytes = diff_encoded

        # 1) differential decode
        de_diff = diff_decode(start_bytes)

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
                                tb_depth=5*(L-1),
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
        for off in range(0, len(deintl), CODEWORD_BYTES):
            cw = deintl[off:off + CODEWORD_BYTES]
            msg, _, _ = RS.decode(cw)  # decode returns (message, ecc, errata)
            restored.extend(msg)

        print(f"reverse RS output (size = {len(restored)}) =")
        print(" ".join(f"{b:02X}" for b in restored))
        print()
        print(f"restored ascii = '{restored.decode('utf-8', errors='replace').replace(chr(0), '\\x00')}'")

    if args.hexout and not args.qpsk_fixed:
        print(final_out.hex())

    # If neither print nor hexout nor out is given, show a tiny summary
    if not (args.out or args.do_print or args.hexout):
        print(f"Pipeline OK. Output bytes: {len(final_out)} "
              f"({len(final_out)//CODEWORD_BYTES} codewords). Use --out/--no-print/--hexout to control output.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())