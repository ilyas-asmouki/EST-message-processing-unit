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
| **Conv Encoder** | ✅ | ✅ | **PASS** (Rigorous 500-block test) |
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

---

## Report Notes: Synchronization & Robustness

*These notes summarize findings from the Python model robustness testing (AWGN/CFO/STO) to be included in the final project report.*

### 1. Transmitter vs. Receiver Responsibility
*   **TX Side (Implemented)**: The Transmitter is "dumb" and open-loop. It does not need to know about channel conditions. Its only responsibility is to generate a clean signal that meets the spectral mask (via the RRC filter) and has correct timing/levels.
*   **RX Side (Future Work)**: The Receiver bears the entire burden of synchronization. It must blindly estimate and correct for all channel impairments introduced during transmission.

### 2. Critical Impairments & Solutions
Simulation with `mpu.model.pipeline` confirmed that the current "ideal" receiver fails under realistic conditions.

| Impairment | Effect on Signal | Observed Failure Mode | Required RX Block (HDL) |
| :--- | :--- | :--- | :--- |
| **CFO (Carrier Frequency Offset)** | Constellation rotates over time ("spinning phase"). | **Catastrophic**. Even small offsets (100ppm) cause the phase to rotate >45° within a packet, causing the QPSK slicer to map symbols to wrong quadrants. Differential encoding handles *static* phase ambiguity but cannot handle *spinning* phase. | **Costas Loop** (Carrier Recovery) |
| **STO (Symbol Timing Offset)** | Sampling occurs on the slope of the pulse, not the peak. | **Degradation**. Causes Inter-Symbol Interference (ISI). At 0.5 sample offset, the "eye" closes significantly, reducing noise margin. | **Gardner Loop** (Timing Recovery) |
| **AWGN (Noise)** | Adds random variance to symbol points. | **Packet Loss**. If noise pushes a symbol across the decision boundary, a bit error occurs. RS coding can fix sparse errors, but fails if synchronization loss causes a burst of errors. | **Matched Filter** (Implemented) |

### 3. Conclusion for HDL Design
For the current scope (TX implementation), the design is robust. However, any future RX implementation **must** include synchronization loops (Costas/Gardner). A simple "matched filter + slicer" architecture is insufficient for real-world hardware links where oscillators are never perfectly matched.
