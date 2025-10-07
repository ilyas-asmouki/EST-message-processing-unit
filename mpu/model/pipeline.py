#!/usr/bin/env python3

# Golden model pipeline runner with integrated visualization:
# input -> RS(255, 223) encoder -> byte wise interleaver (I=2) -> output

# examples:
#   python -m mpu.model.pipeline --text "HELLO WORLD"
#   python -m mpu.model.pipeline --hex "0011223344" --out out.bin
#   python -m mpu.model.pipeline --infile input.bin
#   python -m mpu.model.pipeline --text "HELLO WORLD" --visualize
#   python -m mpu.model.pipeline --text "HELLO WORLD" --save-figs pipeline


import argparse
import sys
from typing import Optional
import numpy as np

from mpu.model.reed_solomon import rs_encode, rs_syndromes, k as RS_K, n as RS_N
from mpu.model.interleaver import interleave, deinterleave, CODEWORD_BYTES
from mpu.model.scrambler import scramble_bits, descramble_bits, DEFAULT_SEED
from mpu.model.conv_encoder import conv_encode
from mpu.model.diff_encoder import diff_encode, diff_decode
from mpu.model.qpsk import qpsk_modulate_bytes_fixed, qpsk_demod_hard_fixed, pack_iq_le_bytes, FRAC_BITS_DEFAULT
from mpu.model.rrc import rrc_filter, upsample, fir_filter, SAMPLES_PER_SYMBOL, FILTER_SPAN, ROLL_OFF, NUM_TAPS
from mpu.model.helpers import pad_to_block, bits_from_bytes, bytes_from_bits

# Pipeline constants
INTERLEAVER_DEPTH = 2
SCRAMBLER_SEED = DEFAULT_SEED
FRAC_BITS = FRAC_BITS_DEFAULT


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


def _generate_visualizations(data_padded: bytes, rs_out: bytes, interleaved: bytes,
                             scrambled: bytes, convolved: bytes, diff_encoded: bytes,
                             iq: np.ndarray, qpsk_i: np.ndarray,
                             qpsk_q: np.ndarray, qpsk_i_rrc: np.ndarray,
                             qpsk_q_rrc: np.ndarray, args: argparse.Namespace):
    """Generate visualization figures for the pipeline stages"""
    try:
        # Import visualization module (only when needed)
        from mpu.model import pipeline_visualizer as pv
        
        print("\nGenerating visualizations...")
        figures = []
        figure_names = []
        
        print("  [7/7] Decoder Path...")
        fig7 = pv.create_decoder_figure(diff_encoded, iq, qpsk_i_rrc, qpsk_q_rrc, args)
        figures.append(fig7)
        figure_names.append("7_decoder")
        
        print("  [6/7] RRC Pulse Shaping...")
        fig6 = pv.create_rrc_figure(qpsk_i, qpsk_q, qpsk_i_rrc, qpsk_q_rrc)
        figures.append(fig6)
        figure_names.append("6_rrc")
        
        print("  [5/7] QPSK Modulation...")
        fig5 = pv.create_qpsk_figure(diff_encoded, iq, qpsk_i, qpsk_q, args)
        figures.append(fig5)
        figure_names.append("5_qpsk")
        
        print("  [4/7] Differential Encoding...")
        fig4 = pv.create_diff_encoder_figure(convolved, diff_encoded)
        figures.append(fig4)
        figure_names.append("4_diff_encoder")
        
        print("  [3/7] Convolutional Encoding...")
        fig3 = pv.create_conv_encoder_figure(scrambled, convolved)
        figures.append(fig3)
        figure_names.append("3_conv_encoder")
        
        print("  [2/7] Scrambling...")
        fig2 = pv.create_scrambler_figure(interleaved, scrambled, rs_out, args)
        figures.append(fig2)
        figure_names.append("2_scrambler")
        
        print("  [1/7] RS & Interleaving...")
        fig1 = pv.create_rs_interleaver_figure(data_padded, rs_out, interleaved, args)
        figures.append(fig1)
        figure_names.append("1_rs_interleaver")
        
        # Save or display
        if args.save_figs:
            print(f"\nSaving figures with prefix '{args.save_figs}'...")
            for fig, name in zip(figures, figure_names):
                filename = f"{args.save_figs}_{name}.png"
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"  Saved: {filename}")
            print("All figures saved successfully!")
            
            # Close all figures
            import matplotlib.pyplot as plt
            for fig in figures:
                plt.close(fig)
        else:
            print("\nDisplaying figures interactively...")
            print("Close each figure window to see the next one.")
            import matplotlib.pyplot as plt
            for idx, (fig, name) in enumerate(zip(figures, figure_names), 1):
                print(f"  Showing stage {name[0]}/7: {name[2:]}")
                plt.show()
                plt.close(fig)
        
        print("\nVisualization complete!")
        
    except ImportError as e:
        print(f"\nWarning: Visualization unavailable - {e}", file=sys.stderr)
        print("Install matplotlib and scipy to enable visualizations.", file=sys.stderr)
    except Exception as e:
        print(f"\nError creating visualizations: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Run RS(255,223) + interleaver golden pipeline with optional visualization.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="UTF-8 text input.")
    src.add_argument("--hex", help="Hex string input (spaces allowed).")
    src.add_argument("--infile", help="Binary input file.")
    
    p.add_argument("--out", help="Write pipeline output (binary) to this file.")
    p.add_argument("--hexout", action="store_true",
                   help="Print hex of pipeline output.")
    
    # Visualization options
    p.add_argument("--visualize", action="store_true",
                   help="Generate visualization figures for all pipeline stages.")
    p.add_argument("--save-figs", metavar="PREFIX",
                   help="Save visualization figures with this prefix (e.g., 'pipeline' -> 'pipeline_1_rs.png'). Implies --visualize.")

    args = p.parse_args(argv)
    
    # --save-figs implies --visualize
    if args.save_figs:
        args.visualize = True
    
    # Set hardcoded values needed by visualizer
    args.depth = INTERLEAVER_DEPTH
    args.seed = SCRAMBLER_SEED
    args.frac_bits = FRAC_BITS
    args.no_scramble = False
    args.rrc = True

    data = _read_input(args)

    # Always pad to RS_K
    data_for_rs = pad_to_block(data, RS_K)

    # RS encode -> concatenated 255B codewords
    rs_out = rs_encode(data_for_rs)

    # Interleave per codeword with depth I=2
    interleaved = interleave(rs_out, INTERLEAVER_DEPTH)

    # Scramble (always enabled)
    scrambled = scramble_bits(interleaved, seed=SCRAMBLER_SEED)

    # Convolutional encoder
    convolved = conv_encode(scrambled)

    # Differential encoder
    diff_encoded = diff_encode(convolved)

    # QPSK modulation (always enabled)
    iq = qpsk_modulate_bytes_fixed(diff_encoded, frac_bits=FRAC_BITS, interleaved=True)
    
    # Extract separate I and Q streams
    qpsk_i = iq[0::2]
    qpsk_q = iq[1::2]
    
    # RRC pulse shaping (always enabled)
    # Upsample by SAMPLES_PER_SYMBOL
    qpsk_i_up = upsample(qpsk_i, SAMPLES_PER_SYMBOL)
    qpsk_q_up = upsample(qpsk_q, SAMPLES_PER_SYMBOL)
    
    # Pad for cascade filter delay
    group_delay_samples = (NUM_TAPS - 1) // 2
    total_delay_samples = 2 * group_delay_samples
    
    qpsk_i_up_padded = np.concatenate([qpsk_i_up, np.zeros(total_delay_samples, dtype=np.int16)])
    qpsk_q_up_padded = np.concatenate([qpsk_q_up, np.zeros(total_delay_samples, dtype=np.int16)])
    
    # TX RRC filter
    qpsk_i_rrc, qpsk_q_rrc = rrc_filter(qpsk_i_up_padded, qpsk_q_up_padded)
    
    # Convert to int16 with saturation
    qpsk_i_int = np.clip(qpsk_i_rrc, -32768, 32767).astype(np.int16)
    qpsk_q_int = np.clip(qpsk_q_rrc, -32768, 32767).astype(np.int16)
    
    # Re-interleave for output
    iq_rrc = np.empty(len(qpsk_i_int) + len(qpsk_q_int), dtype=np.int16)
    iq_rrc[0::2] = qpsk_i_int
    iq_rrc[1::2] = qpsk_q_int
    
    final_out = pack_iq_le_bytes(iq_rrc)

    # Generate visualizations if requested (before text output)
    if args.visualize:
        _generate_visualizations(data_for_rs, rs_out, interleaved, scrambled, convolved,
                                diff_encoded, iq, qpsk_i, qpsk_q, qpsk_i_rrc, qpsk_q_rrc, args)

    # Write binary output
    if args.out:
        with open(args.out, "wb") as f:
            f.write(final_out)
        print(f"Output written to {args.out} ({len(final_out)} bytes)")

    # Print hex output
    if args.hexout:
        print(final_out.hex())

    # Default summary if no output specified
    if not (args.out or args.hexout or args.visualize):
        print(f"Pipeline OK. Output: {len(final_out)} bytes "
              f"({len(iq)//2} QPSK symbols, {len(iq_rrc)//2} RRC samples)")
        print(f"Use --out to save, --hexout to print, or --visualize to see plots.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())