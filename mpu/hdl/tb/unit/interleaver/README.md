# Interleaver Unit Testbench

Drives three RS codewords from the Python golden-model vectors through `byte_interleaver` (depth = 2) while randomizing both input bubbles and output back-pressure. Every byte, SOP/last pulse, and parity flag is compared against the expected interleaver output.

## Run

From this directory:

```bash
make
```

The Makefile compiles the DUT/testbench with Icarus Verilog (`IVL`/`VVP` knobs are exposed) and emits `PASS` once all bytes match. Update `VEC_DIR`/`DEPTH` in `interleaver_tb.sv` if you want to target other vector sets or interleaver depths.
