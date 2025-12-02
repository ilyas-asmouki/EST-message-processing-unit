# System Verification

This directory contains the system-level testbench for the FPGA Top module.

## Files

- `fpga_top_tb.sv`: Top-level testbench verifying the integration of `tx_chain` and `dac_interface`.
- `Makefile`: Build script for Icarus Verilog.

## Running the Simulation

To run the simulation using Icarus Verilog:

```bash
make
```

This will:
1. Compile the design and testbench.
2. Run the simulation using the `test_100_blocks` vector set.
3. Check for mismatches between the RTL output and the golden vectors.

## Test Vectors

The testbench uses vectors located in `../../vectors/test_100_blocks`.
- Input: `rs/input.hex` (Raw bytes)
- Output: `rrc/output.hex` (Expected I/Q samples)

## Mock Primitives

Since Icarus Verilog does not support Xilinx primitives natively, `../../sim/unisims_mock.sv` provides behavioral models for:
- `ODDR`: Output Double Data Rate register.
- `OBUFDS`: Differential Output Buffer.
