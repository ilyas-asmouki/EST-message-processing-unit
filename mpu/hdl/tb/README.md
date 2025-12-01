# Testbenches

- `unit/`: self-checking benches for individual stages (RS encoder TB, interleaver TB, etc.), each driven by vectors emitted from the golden model.
- `system/`: end-to-end TX benches that stitch the stages together, compare against the Python reference, and optionally hook into DPI for IQ waveforms.

Keeping verification collateral here decouples it from the synthesizable RTL so we can iterate quickly without polluting the FPGA build tree.
