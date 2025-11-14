# HDL Helper Scripts

This is the staging ground for everything we need *before* any RTL is checked in:

- Python utilities that export reference vectors/coefficients from `mpu/model` (e.g., RRC taps, scrambler seeds, RS generator tables).
- Build orchestration (Make/Tcl) that can compile the RTL tree or launch regressions without manual intervention.
- Converter tools for packaging captured IQ data into memories or AXI stimulus for the testbenches.

## Available tools

### `gen_vectors.py`
Generates stage-by-stage reference vectors (RS input/output, interleaver output, etc.) driven by the golden model. Example:

```bash
source mpu/venv/bin/activate
python -m mpu.hdl.scripts.gen_vectors \
    --text "HELLO FPGA" \
    --depth 2 \
    --stage rs --stage interleaver --stage scrambler \
    --out-dir mpu/hdl/vectors/demo
```

Each requested stage gets `input.hex`, `output.hex`, and a `metadata.json` file, plus a top-level `vector_summary.json`. Testbenches can load these files directly to validate upcoming RTL implementations.
