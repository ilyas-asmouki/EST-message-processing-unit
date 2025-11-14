# RS Encoder Unit Testbench

Self-checking testbench that drives two RS(255,223) blocks through the HDL encoder and compares every byte against vectors emitted from the Python golden model (`hdl/vectors/rs_smoke/rs`). Randomized `m_axis_ready` stalls exercise the handshakes while scoreboard checks data, SOP/last pulses, and the parity indicator.

## Run

From this directory:

```bash
make        # compiles with Icarus and runs the TB
```

The `Makefile` exposes `IVL`, `VVP`, and `SIM_FLAGS` knobs if you need to swap simulators or pass extra defines. Test output prints `PASS` when all bytes match.
