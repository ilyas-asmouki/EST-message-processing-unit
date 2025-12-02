#!/usr/bin/env python3
"""
Generate test vectors for RRC filter RTL verification.
"""

import sys
import os
import numpy as np
import json

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from mpu.model.rrc import rrc_filter, upsample, SAMPLES_PER_SYMBOL

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "../vectors/rrc_smoke")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Generate Random Symbols (QPSK-like)
    # Use values that fit in 16-bit signed
    # 23170 is approx 1/sqrt(2) of full scale
    np.random.seed(42)
    num_symbols = 100
    qpsk_val = 23170
    
    sym_i = np.random.choice([-qpsk_val, qpsk_val], num_symbols).astype(np.int16)
    sym_q = np.random.choice([-qpsk_val, qpsk_val], num_symbols).astype(np.int16)
    
    # 2. Generate Expected Output (Golden Model)
    # Upsample
    i_up = upsample(sym_i, SAMPLES_PER_SYMBOL)
    q_up = upsample(sym_q, SAMPLES_PER_SYMBOL)
    
    # Pad to flush the filter (RRC has memory)
    # The RTL has a shift register of length 11.
    # We need to feed enough zeros at the end to see the full response if we want to check tail.
    # But for continuous processing, we just check the valid part.
    # Let's pad the input to the Python model to match the "continuous" nature.
    # Actually, the Python model `rrc_filter` uses `fir_filter` which processes the whole array.
    # To match RTL exactly, we need to consider the padding.
    # The RTL shift register starts at 0.
    # The Python `fir_filter` starts with a zero-filled delay line.
    # This matches.
    
    # However, the Python `fir_filter` output length = input length.
    # If we feed 100 symbols, we get 400 samples.
    # The RTL will produce 400 samples.
    
    i_out, q_out = rrc_filter(i_up, q_up)
    
    # 3. Save Vectors
    # Input format: 16-bit I, 16-bit Q (Symbol Rate)
    # Output format: 16-bit I, 16-bit Q (Sample Rate)
    
    # Interleave I/Q for file storage (I0, Q0, I1, Q1...)
    # Input file
    input_data = np.zeros(2 * num_symbols, dtype=np.int16)
    input_data[0::2] = sym_i
    input_data[1::2] = sym_q
    
    with open(os.path.join(out_dir, "rrc_input.bin"), "wb") as f:
        f.write(input_data.tobytes())
        
    # Output file
    output_data = np.zeros(2 * len(i_out), dtype=np.int16)
    output_data[0::2] = i_out
    output_data[1::2] = q_out
    
    with open(os.path.join(out_dir, "rrc_output.bin"), "wb") as f:
        f.write(output_data.tobytes())
        
    # Metadata
    meta = {
        "num_symbols": num_symbols,
        "num_samples": len(i_out),
        "sps": SAMPLES_PER_SYMBOL
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"Generated vectors in {out_dir}")
    print(f"  Symbols: {num_symbols}")
    print(f"  Samples: {len(i_out)}")

if __name__ == "__main__":
    main()
