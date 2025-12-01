# EST Message Processing Unit (MPU)

This repository contains the complete implementation of a CCSDS-compatible Message Processing Unit (MPU) for FPGA, including a bit-accurate Python golden model and SystemVerilog RTL.

## Project Structure

- **`mpu/model/`**: Python Golden Model (Reference implementation).
- **`mpu/hdl/rtl/`**: SystemVerilog RTL implementation.
- **`mpu/hdl/tb/`**: Unit and integration testbenches.
- **`mpu/hdl/scripts/`**: Vector generation tools for verification.

## Status

| Module | Python Model | SystemVerilog RTL | Verification Status |
| :--- | :---: | :---: | :--- |
| **RS Encoder** | ✅ | ✅ | **PASS** (Rigorous 500-block test) |
| **Interleaver** | ✅ | ✅ | **PASS** (Rigorous 500-block test) |
| **Scrambler** | ✅ | ✅ | **PASS** (Rigorous 500-block test) |
| **Conv Encoder** | ✅ | 🚧 | Pending |
| **Diff Encoder** | ✅ | 🚧 | Pending |
| **QPSK Mapper** | ✅ | 🚧 | Pending |

---

## Setup Instructions

### 1. Python Environment (Golden Model & Vectors)

The Python model is used to generate reference vectors for HDL verification.

```bash
cd mpu
python3 -m venv venv
source venv/bin/activate
pip install reedsolo scikit-commpy numpy scipy
```

### 2. HDL Simulation Tools

The project uses **Icarus Verilog** (`iverilog`) for simulation.

```bash
sudo apt-get install iverilog
```

---

## Running the Golden Model

You can run the full software pipeline to see how data is transformed at each stage.

```bash
# From repository root
python3 -m mpu.model.pipeline --text "HELLO WORLD" --show
```

**Options:**
- `--text "..."` / `--hex "..."` / `--infile file.bin`: Input sources.
- `--depth N`: Interleaver depth (default: 2).
- `--no-scramble`: Disable scrambling.
- `--out out.bin`: Save binary output.

---

## Verification (RTL)

We use a "rigorous verification" methodology where the Python model generates large randomized datasets (vectors), which are then loaded by SystemVerilog testbenches to verify the RTL byte-for-byte.

### Running Unit Tests

Testbenches are located in `mpu/hdl/tb/unit/`. Each module has a `Makefile`.

**Example: Testing the Scrambler**
```bash
cd mpu/hdl/tb/unit/scrambler
make
```

**Example: Testing the Full Chain (RS -> Int -> Scr)**
```bash
cd mpu/hdl/tb/unit/chain
make
```

### Generating New Vectors

To generate a new set of test vectors (e.g., for debugging or stress testing):

```bash
# Generate 500 blocks of random data
python3 -c "import os; open('random.bin', 'wb').write(os.urandom(500 * 223))"

# Generate vectors for all stages
python3 mpu/hdl/scripts/gen_vectors.py --infile random.bin --stage rs --stage interleaver --stage scrambler --out-dir mpu/hdl/vectors/rigorous_custom
```

Then update the `VEC_DIR` in the corresponding testbench to point to `mpu/hdl/vectors/rigorous_custom`.
