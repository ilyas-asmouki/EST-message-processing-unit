#!/usr/bin/env python3
# mpu/hdl/sim/vectors/generate_rs_vectors.py
# generate test vectors for RS encoder hdl verification
# outputs vectors in a format easily readable by verilog testbenches
#
# python generate_rs_vectors.py

import sys
import os
from pathlib import Path
import random


script_path = Path(__file__).resolve()
vectors_dir = script_path.parent                    # mpu/hdl/sim/vectors
sim_dir = vectors_dir.parent                        # mpu/hdl/sim
hdl_dir = sim_dir.parent                            # mpu/hdl
mpu_dir = hdl_dir.parent                            # mpu
project_root = mpu_dir.parent                       # parent of mpu/


sys.path.insert(0, str(project_root))


from mpu.model.reed_solomon import rs_encode_block_223, k as RS_K, n as RS_N


def bytes_to_hex_string(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def generate_test_vectors(output_file: str, num_random: int = 10):
    
    vectors = []
    
    # test case 1: all zeros
    msg_zeros = bytes([0] * RS_K)
    cw_zeros = rs_encode_block_223(msg_zeros)
    vectors.append(("all_zeros", msg_zeros, cw_zeros))
    
    # test case 2: all ones (0xFF)
    msg_ones = bytes([0xFF] * RS_K)
    cw_ones = rs_encode_block_223(msg_ones)
    vectors.append(("all_ones", msg_ones, cw_ones))
    
    # test case 3: alternating pattern (0xAA)
    msg_aa = bytes([0xAA] * RS_K)
    cw_aa = rs_encode_block_223(msg_aa)
    vectors.append(("pattern_aa", msg_aa, cw_aa))
    
    # test case 4: alternating pattern (0x55)
    msg_55 = bytes([0x55] * RS_K)
    cw_55 = rs_encode_block_223(msg_55)
    vectors.append(("pattern_55", msg_55, cw_55))
    
    # test case 5: impulse at start
    msg_impulse_start = bytes([0x80] + [0] * (RS_K - 1))
    cw_impulse_start = rs_encode_block_223(msg_impulse_start)
    vectors.append(("impulse_start", msg_impulse_start, cw_impulse_start))
    
    # test case 6: impulse at end
    msg_impulse_end = bytes([0] * (RS_K - 1) + [0x01])
    cw_impulse_end = rs_encode_block_223(msg_impulse_end)
    vectors.append(("impulse_end", msg_impulse_end, cw_impulse_end))
    
    # test case 7: counter pattern
    msg_counter = bytes([i % 256 for i in range(RS_K)])
    cw_counter = rs_encode_block_223(msg_counter)
    vectors.append(("counter", msg_counter, cw_counter))
    
    # test case 8: reverse counter
    msg_reverse = bytes([(255 - i) % 256 for i in range(RS_K)])
    cw_reverse = rs_encode_block_223(msg_reverse)
    vectors.append(("reverse_counter", msg_reverse, cw_reverse))
    
    # random test cases
    random.seed(0x52535F5645435F)  # "RS_VEC_" in ASCII-ish
    for i in range(num_random):
        msg_rand = bytes([random.randint(0, 255) for _ in range(RS_K)])
        cw_rand = rs_encode_block_223(msg_rand)
        vectors.append((f"random_{i:02d}", msg_rand, cw_rand))
    


    with open(output_file, 'w') as f:
        f.write("// RS(255,223) Encoder Test Vectors\n")
        f.write(f"// Generated from golden model (reed_solomon.py)\n")
        f.write(f"// Format: test_name | input_message (223 bytes) | expected_codeword (255 bytes)\n")
        f.write(f"// Each byte in hex, space-separated\n")
        f.write(f"//\n")
        f.write(f"// Total test vectors: {len(vectors)}\n")
        f.write("//\n\n")
        
        for name, msg, cw in vectors:
            f.write(f"// Test: {name}\n")
            f.write(f"MSG: {bytes_to_hex_string(msg)}\n")
            f.write(f"CW:  {bytes_to_hex_string(cw)}\n")
            f.write("\n")
    
    print(f"Generated {len(vectors)} test vectors")
    print(f"Written to: {output_file}")
    
    # also create a compact binary format for faster HDL loading if needed
    bin_file = output_file.replace('.txt', '.bin')
    with open(bin_file, 'wb') as f:
        # header: number of vectors (4 bytes)
        f.write(len(vectors).to_bytes(4, 'little'))
        # each vector: message (223 bytes) + codeword (255 bytes)
        for name, msg, cw in vectors:
            f.write(msg)
            f.write(cw)
    
    print(f"Binary format: {bin_file}")
    
    return vectors


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "rs_encoder_vectors.txt"
    
    print("RS Encoder Test Vector Generator")
    print("=" * 50)
    
    vectors = generate_test_vectors(str(output_file), num_random=20)
    
    print("\nTest vector summary:")
    print(f"  Input size:  {RS_K} bytes (k={RS_K})")
    print(f"  Output size: {RS_N} bytes (n={RS_N})")
    print(f"  Parity bytes: {RS_N - RS_K} bytes")
    print("\nReady for HDL verification!")

