# EST Message Processing Unit

This repository contains all components of the Message Processing Unit (MPU) project:
- **Golden model (Python)** — located in `mpu/model/`
- **HDL simulation code** — (to be added)
- **Verilog implementation** — (to be added)

Currently, only the **golden model** in `mpu/model/` is implemented and usable.

---

## Setup Instructions

Before running any Python script in the `mpu/model/` directory, set up a virtual environment and install dependencies.

### Steps

1. **Create a virtual environment** (from inside the `mpu/` directory):

   ```bash
   python -m venv venv
   ```


2. **Activate the virtual environment**:

   ```bash
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install reedsolo scikit-commpy numpy
   ```


## Golden Model (`mpu/model/`)

### Modules implemented so far

- `reed_solomon.py` — Minimal RS(255,223) encoder in GF(2^8)
- `interleaver.py` — Byte-wise block interleaver with depths I ∈ {1,2,3,4,5,8}
- `pipeline.py` — Entry point that chains modules together into a full pipeline

### Running the pipeline

From the repository root:

   ```bash
   python -m mpu.model.pipeline --text "HELLO WORLD"
   ```

### Defaults

- **Interleaver depth:** `2`  
- **Padding policy:** `zero` (pads incomplete 223-byte blocks with zeros)  
- **Printing:** enabled (prints every stage by default)  


### Example output
```
input =
72 69 76 76 79

RS output =
72 69 76 76 79 0 0 0 ... <223 data bytes + 32 parity bytes>

interleaver output (I=2) =
72 0 69 0 76 0 76 0 79 0 0 0 ... <255 bytes interleaved>
```

### Options

```bash
python -m mpu.model.pipeline [options]
```

- `--text "HELLO"` Input as UTF-8 string
- `--hex "00112233"` Input as hex string
- `--infile file.bin` Input from binary file
- `--depth N` Interleaver depth (1,2,3,4,5,8) [default=2]
- `--pad zero|none` Pad to 223-byte blocks [default=zero]
- `--out out.bin` Write binary output to file
- `--hexout` Print output in hex
- `--no-print` Suppress printing of stages
- `--check` Verify RS codeword syndromes are zero
- `--show` Print sizes, block counts, etc.

## Next Steps

- Add convolutional encoder, mapper, and filtering stages to the golden model  
- Mirror each golden model module in HDL (Verilog)  
- Integrate into the full FPGA design  
