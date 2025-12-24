# EST Message Processing Unit (MPU) - Project Report

**Date:** December 2025  
**Target Platform:** ZedBoard (Zynq-7000) + ADI FMCOMMS1-EBZ (AD9122 DAC)

## 1. Executive Summary

This project implements a complete CCSDS-compatible Message Processing Unit (MPU) for satellite communications. The system takes a digital message, applies a chain of channel coding and modulation algorithms, and transmits it via a high-speed DAC.

The project spans three domains:
1.  **Algorithm Design**: A bit-accurate Python golden model used for verification and robustness analysis.
2.  **RTL Design**: A synthesizable SystemVerilog implementation of the TX chain.
3.  **Hardware Implementation**: A full Zynq-based system-on-chip design running on the ZedBoard, driving an Analog Devices AD9122 DAC.

---

## 2. System Architecture

The data flow follows the CCSDS standard for telemetry:

```mermaid
graph LR
    Input[User Data] --> RS[Reed-Solomon]
    RS --> Int[Interleaver]
    Int --> Scr[Scrambler]
    Scr --> Conv[Convolutional Enc]
    Conv --> Diff[Differential Enc]
    Diff --> QPSK[QPSK Mapper]
    QPSK --> RRC[RRC Filter]
    RRC --> DAC[AD9122 DAC]
```

### 2.1 Pipeline Stages
*   **Reed-Solomon (255, 223)**: Provides burst error correction.
*   **Matrix Interleaver**: Spreads burst errors to help the Convolutional decoder.
*   **Scrambler**: Randomizes data to ensure sufficient bit transitions for clock recovery.
*   **Convolutional Encoder (k=7, r=1/2)**: Provides robust error correction for Gaussian noise.
*   **Differential Encoder**: Resolves phase ambiguity (180-degree rotation) at the receiver.
*   **QPSK Mapper**: Maps bits to complex symbols (I/Q).
*   **RRC Filter (Root Raised Cosine)**: Pulse shaping to limit bandwidth and minimize ISI.

---

## 3. Algorithm Design & Golden Model

A bit-accurate Python model (`mpu/model/`) serves as the "Golden Reference". It exactly matches the hardware behavior, including fixed-point arithmetic for the RRC filter.

### 3.1 Robustness Analysis
Simulation with channel impairments (AWGN, CFO, STO) revealed critical insights for the future Receiver (RX) design:

*   **CFO (Carrier Frequency Offset)**: Even small offsets (100ppm) cause catastrophic failure ("spinning phase"). A **Costas Loop** is mandatory in the RX.
*   **STO (Symbol Timing Offset)**: Sampling off-peak closes the eye diagram. A **Gardner Loop** is required.
*   **AWGN**: The current coding scheme is robust, but synchronization loss leads to packet loss.

---

## 4. RTL Implementation

The hardware logic is written in **SystemVerilog** (`mpu/hdl/rtl/`).

### 4.1 Verification Methodology
We employed a "Rigorous Verification" strategy:
1.  **Python** generates random input vectors and computes the expected output.
2.  **Icarus Verilog** runs the RTL with these inputs.
3.  **Testbench** compares RTL output vs. Python output byte-for-byte.

**Status:**
*   All unit tests (RS, Interleaver, Scrambler, etc.) **PASSED** with 500+ random blocks.
*   Full Chain integration test **PASSED**.

### 4.2 Vivado Compatibility (`mpu_top.v`)
To integrate with Xilinx Vivado IP Integrator, we created a Verilog wrapper (`mpu_top.v`) around the SystemVerilog `tx_chain`.
*   **Input**: AXI Stream (8-bit) from Zynq DMA.
*   **Output**: 16-bit LVDS Data + Clock/Frame for AD9122.
*   **Clocking**: Runs at **50 MHz** to meet timing constraints on the ZedBoard.

---

## 5. Hardware Implementation (Vivado)

The design was synthesized and implemented for the **ZedBoard**.

### 5.1 Block Design
*   **Zynq PS**: Runs the software driver.
*   **AXI DMA**: Fetches data from DDR memory and streams it to the FPGA logic.
*   **mpu_top**: The custom RTL core.
*   **Clocking**: PL Fabric Clock set to 50 MHz.

### 5.2 Constraints & IO
We mapped the design to the **ADI FMCOMMS1-EBZ** FMC card.
*   **Standard**: LVDS_25 (2.5V).
*   **Pinout**: Exact mapping to FMC LPC pins (LA00-LA33) derived from the ADI schematic.

---

## 6. Software Driver (Vitis)

A bare-metal C application (`mpu_app`) runs on the ARM Cortex-A9.

### 6.1 Driver Features
*   **DMA Management**: Uses `XAxiDma` to transfer data packets from DDR to the PL.
*   **Cache Coherency**: Implements `Xil_DCacheFlushRange` to ensure the DMA reads valid data from physical RAM.
*   **SDT Adaptation**: Updated to support modern Vitis System Device Tree flows (using `XPAR_XAXIDMA_0_BASEADDR`).

### 6.2 Execution Flow
1.  Initialize Platform & DMA.
2.  Generate Test Pattern (Ramp) in DDR.
3.  Flush Cache.
4.  Trigger DMA Transfer.
5.  FPGA receives stream -> Processes -> Drives DAC.

---

## 7. How to Reproduce

### 7.1 Simulation (RTL)
```bash
# Install Icarus Verilog
sudo apt-get install iverilog

# Run System Integration Test
cd mpu/hdl/tb/system
make
```

### 7.2 Hardware Build (Vivado)
1.  Create a new RTL Project for **ZedBoard**.
2.  Add sources from `mpu/hdl/rtl`.
3.  Add constraints from `mpu/hdl/constraints/constraints.xdc`.
4.  Create Block Design: Zynq + AXI DMA + `mpu_top`.
5.  **Important**: Set PL Clock to **50 MHz**.
6.  Generate Bitstream & Export Hardware (`.xsa`).

### 7.3 Software Build (Vitis)
1.  Create Platform Project from the `.xsa`.
2.  Create Application Project ("Hello World" template).
3.  Replace `main.c` with the driver code in `mpu/sw/mpu_app/src/main.c`.
4.  Build & Run on Hardware.
