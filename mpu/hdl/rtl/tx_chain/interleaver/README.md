# byte_interleaver

Streaming RS(255,223) block interleaver with configurable depth `DEPTH ∈ {1,2,3,4,5,8}`. Bytes arrive over an AXI-stream style interface (valid/ready, SOP/LAST, parity flag) one full 255-byte codeword at a time. Internally, two ping-pong buffers capture consecutive codewords so the stage can accept the next block while emitting the interleaved version of the previous block. The generator polynomial metadata (`s_axis_is_parity`) is preserved so downstream blocks know which bytes belonged to the parity tail.

## Ports

- `s_axis_*`: upstream stream (from RS encoder). `s_axis_last` must assert on byte 254 of every block; `s_axis_sop` tags the first byte.
- `m_axis_*`: downstream stream (to scrambler). `m_axis_sop`/`m_axis_last` transition after the permutation; `m_axis_is_parity` tracks whether each byte originated in the RS parity trailer.

Use `interleaver_pkg.sv` for permutation tables and helper functions. See `hdl/tb/unit/interleaver` for a reference testbench and `hdl/scripts/gen_vectors.py` for generating fresh stimuli from the golden model.
