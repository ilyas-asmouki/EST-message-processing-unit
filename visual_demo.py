#!/usr/bin/env python3
import os
import sys
import subprocess

def run_demo():
    print("="*60)
    print("      EST MPU: VISUAL VERIFICATION DEMO")
    print("="*60)
    
    user_text = input("Enter a short text string to encode (default: 'Hello World'): ").strip()
    if not user_text:
        user_text = "Hello World"
        
    print(f"\n[1] Generating Golden Model Vectors for: '{user_text}'")
    
    # 1. Generate Vectors
    # We use the existing gen_vectors.py script
    # It will pad the input to a block size (223 bytes for RS)
    # We output to mpu/hdl/vectors/demo
    
    demo_dir = os.path.abspath("mpu/hdl/vectors/demo")
    script_path = "mpu/hdl/scripts/gen_vectors.py"
    
    # Ensure PYTHONPATH is set
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    cmd = [
        sys.executable, script_path,
        "--text", user_text,
        "--stage", "rs",
        "--stage", "qpsk",
        "--out-dir", demo_dir
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE)
        print(f"    -> Vectors generated in {demo_dir}")
    except subprocess.CalledProcessError as e:
        print("Error generating vectors!")
        sys.exit(1)

    # 2. Run Simulation
    print("\n[2] Running SystemVerilog Simulation (Chain TB)...")
    print("    -> Comparing RTL output against Golden Model output")
    print("-" * 60)
    
    # We need to calculate how many blocks we generated.
    # RS block size is 223.
    # len(text) / 223 rounded up.
    # But gen_vectors pads to block.
    # Let's just assume 1 block for short text, but to be safe let's check the file size or just pass 1 if text is short.
    # Actually, gen_vectors pads to ONE block if it fits.
    num_blocks = (len(user_text.encode('utf-8')) + 222) // 223
    
    tb_dir = "mpu/hdl/tb/unit/chain"
    
    # Compile command
    # We use the Makefile's logic but manually to inject parameters easily or just use iverilog directly
    # Actually, let's use iverilog directly to be sure about parameters.
    
    # Paths relative to tb_dir
    rs_vec_param = f"chain_tb.RS_VEC_DIR=\"{demo_dir}/rs\""
    qpsk_vec_param = f"chain_tb.QPSK_VEC_DIR=\"{demo_dir}/qpsk\""
    blocks_param = f"chain_tb.TEST_BLOCKS={num_blocks}"
    
    # Source files (from Makefile)
    hdl_root = "../../../rtl/tx_chain"
    srcs = [
        f"{hdl_root}/rs/rs_encoder_pkg.sv",
        f"{hdl_root}/rs/rs_encoder.sv",
        f"{hdl_root}/interleaver/interleaver_pkg.sv",
        f"{hdl_root}/interleaver/interleaver.sv",
        f"{hdl_root}/scrambler/scrambler.sv",
        f"{hdl_root}/conv_encoder/conv_encoder.sv",
        f"{hdl_root}/diff_encoder/diff_encoder.sv",
        f"{hdl_root}/qpsk_mapper/qpsk_mapper.sv",
        "chain_tb.sv"
    ]
    
    includes = [
        f"-I{hdl_root}/rs",
        f"-I{hdl_root}/interleaver",
        f"-I{hdl_root}/scrambler",
        f"-I{hdl_root}/conv_encoder",
        f"-I{hdl_root}/diff_encoder",
        f"-I{hdl_root}/qpsk_mapper"
    ]
    
    compile_cmd = ["iverilog", "-g2012", "-Wall", "-Wno-timescale", "-DASSERT_ON"] + includes + \
                  ["-P", rs_vec_param, "-P", qpsk_vec_param, "-P", blocks_param] + \
                  ["-s", "chain_tb", "-o", "demo_chain.vvp"] + srcs
                  
    # Run compilation in the TB directory
    try:
        subprocess.run(compile_cmd, cwd=tb_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print("Compilation Failed:")
        print(e.stderr.decode())
        sys.exit(1)
        
    # Run simulation
    try:
        # We filter the output to show only our VISUAL tags and the final result
        proc = subprocess.Popen(["vvp", "demo_chain.vvp"], cwd=tb_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        
        print(f"{'BYTE IDX':<10} | {'GOLDEN I/Q':<25} | {'RTL I/Q':<25} | {'STATUS'}")
        print("-" * 80)
        
        if proc.stdout:
            for line in proc.stdout:
                line_str = line.decode().strip()
                if "[VISUAL]" in line_str:
                    # Parse: [VISUAL] Byte 00000 | Golden I: 0xa57e Q: 0xa57e | RTL I: 0xa57e Q: 0xa57e | MATCH
                    parts = line_str.split("|")
                    if len(parts) >= 4:
                        byte_idx = parts[0].split()[-1]
                        
                        # Golden I: 0xa57e Q: 0xa57e
                        golden_part = parts[1].strip()
                        golden_i = golden_part.split()[2]
                        golden_q = golden_part.split()[4]
                        golden_str = f"I:{golden_i} Q:{golden_q}"
                        
                        # RTL I: 0xa57e Q: 0xa57e
                        rtl_part = parts[2].strip()
                        rtl_i = rtl_part.split()[2]
                        rtl_q = rtl_part.split()[4]
                        rtl_str = f"I:{rtl_i} Q:{rtl_q}"
                        
                        status = parts[3].strip()
                        
                        print(f"{byte_idx:<10} | {golden_str:<25} | {rtl_str:<25} | {status}")
                elif "PASS:" in line_str:
                    print("-" * 60)
                    print(f"\nRESULT: {line_str}")
                elif "FAIL:" in line_str:
                    print("-" * 60)
                    print(f"\nRESULT: {line_str}")
                
        proc.wait()
        
    except Exception as e:
        print(f"Simulation error: {e}")
        sys.exit(1)

    print("\nDone.")

if __name__ == "__main__":
    run_demo()
