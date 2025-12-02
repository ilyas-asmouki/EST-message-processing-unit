# Hardware Synthesis & Implementation Guide

This guide outlines the step-by-step process to take the verified RTL design and implement it on the **ZedBoard (Zynq-7000)** with the **Analog Devices AD9122 DAC** FMC card.

**Goal:** Create a fully functional system where the Zynq PS sends data via DMA to the FPGA logic, which processes it (RS -> Interleaver -> Scrambler -> Conv -> Diff -> QPSK -> RRC) and drives the DAC.

---

## Phase 1: Vivado Project Setup

### 1. Create Project
1.  Open **Vivado**.
2.  Click **Create Project**.
3.  Name: `mpu_zynq_system`.
4.  Type: **RTL Project** (Do not specify sources yet).
5.  **Default Part / Board**: Select **Boards** tab -> Search for **ZedBoard** -> Select it. (If missing, install board files via Vivado Store).
6.  Finish.

### 2. Import RTL Sources
1.  In **Project Manager** -> **Add Sources** -> **Add or create design sources**.
2.  Click **Add Directories** and select `mpu/hdl/rtl`.
3.  **Important**: Ensure "Copy sources into project" is **unchecked** (keep them linked to your repo).
4.  Click **Finish**.
5.  Vivado will scan the hierarchy. You should see `fpga_top` as the top module (or near the top).

### 3. Create Block Design (IP Integrator)
This is where we connect the Processor (PS) to our Logic (PL).

1.  **IP Integrator** -> **Create Block Design**. Name it `system`.
2.  **Add Zynq PS**:
    *   Right-click diagram -> **Add IP** -> Search `ZYNQ7 Processing System`.
    *   Click **Run Block Automation** (Apply Board Preset). This configures DDR, Clocks, and MIO for ZedBoard.
3.  **Add AXI DMA**:
    *   **Add IP** -> `AXI Direct Memory Access`.
    *   Double-click to configure:
        *   **Enable Scatter Gather Engine**: Uncheck (Simple mode is easier for beginners).
        *   **Enable Write Channel**: Uncheck (We only need Read Channel: Memory -> Device).
        *   **Width of Buffer Length Register**: 23 (supports large transfers).
        *   **Stream Data Width**: 8 bits (Matches our `fpga_top` input).
        *   **Max Burst Size**: 16 or 32.
4.  **Add RTL Module**:
    *   Right-click diagram -> **Add Module**.
    *   Select `fpga_top`.
5.  **Add Clocking**:
    *   The AD9122 usually requires a high-quality clock. If you are using the Zynq to generate the clock (e.g., 100MHz or 200MHz):
    *   Double-click Zynq PS -> **Clock Configuration** -> **PL Fabric Clocks**.
    *   Enable `FCLK_CLK0` and set requested frequency (e.g., 100 MHz).
    *   (Optional) If you need a specific high-speed clock (250MHz), use a **Clocking Wizard** IP connected to `FCLK_CLK0`.

### 4. Wiring the System
1.  **Run Connection Automation**:
    *   Vivado will offer to connect the AXI interfaces.
    *   Check all boxes. It will automatically add **AXI Interconnects** and **Processor System Reset** blocks.
    *   Ensure the DMA's `M_AXI_MM2S` connects to the Zynq's `S_AXI_HP0` (High Performance port) for direct DDR access. If `S_AXI_HP0` isn't enabled, enable it in Zynq PS settings.
2.  **Connect Data Stream**:
    *   Connect `axi_dma_0/M_AXIS_MM2S` (Master Stream) to `fpga_top_0/s_axis` (Slave Stream).
    *   *Note: If interface names don't match exactly, expand the interfaces (+) and connect `tdata`, `tvalid`, `tready`, `tlast` manually.*
3.  **Connect Clocks & Resets**:
    *   Ensure `fpga_top/clk` is connected to the system clock (e.g., `FCLK_CLK0`).
    *   Ensure `fpga_top/rst_n` is connected to `peripheral_aresetn` (from Processor System Reset).
4.  **Make External Ports**:
    *   Right-click on the DAC output pins of `fpga_top_0` (`dac_d_p`, `dac_d_n`, `dac_dci_p`, etc.).
    *   Select **Make External**.
    *   Rename the external ports to match your constraints file (e.g., `DAC_DATA_P`, `DAC_DATA_N`, etc.).

### 5. Validate Design
1.  Click **Validate Design** (Checkmark icon).
2.  Fix any errors (usually clock/reset mismatches).
3.  Save.

### 6. Create HDL Wrapper
1.  In **Sources** tab, right-click `system.bd`.
2.  Select **Create HDL Wrapper**.
3.  Choose **Let Vivado manage wrapper**.
4.  This wrapper becomes the new top-level file.

---

## Phase 2: Constraints (I/O Planning)

You need to tell the FPGA which physical pins connect to the FMC connector.

1.  **Add Sources** -> **Add or create constraints**.
2.  Create file: `constraints.xdc`.
3.  Add the pin mappings. You will need the **ZedBoard Schematic** and **AD9122 FMC Card Schematic**.
    *   *Example (Pseudo-code - verify with schematics!):*
    ```tcl
    # DAC Data (LVDS)
    set_property PACKAGE_PIN <PIN_LOC> [get_ports {dac_d_p[0]}]
    set_property IOSTANDARD LVDS_25 [get_ports {dac_d_p[0]}]
    
    # DAC Clock (DCI)
    set_property PACKAGE_PIN <PIN_LOC> [get_ports dac_dci_p]
    set_property IOSTANDARD LVDS_25 [get_ports dac_dci_p]
    
    # SPI Control (To configure AD9122)
    # You likely need to route Zynq SPI pins to the FMC SPI pins via EMIO
    ```

---

## Phase 3: Implementation & Bitstream

1.  **Generate Bitstream**: Click the button in the Flow Navigator.
    *   This runs Synthesis -> Implementation -> Bitstream Generation.
    *   This will take 10-30 minutes.
2.  **Export Hardware**:
    *   **File** -> **Export** -> **Export Hardware**.
    *   Select **Include Bitstream**.
    *   Save the `.xsa` file (e.g., `design_1_wrapper.xsa`).

---

## Phase 4: Software Development (Vitis)

Now we write the C code to run on the ARM processor.

### 1. Setup Vitis
1.  **Tools** -> **Launch Vitis IDE**.
2.  Select a workspace folder.
3.  **Create Platform Project**:
    *   Name: `zed_mpu_platform`.
    *   Select **Create from hardware specification (XSA)**.
    *   Browse to your exported `.xsa` file.
    *   OS: **standalone** (Bare metal, no Linux).
    *   Build the platform project.

### 2. Create Application
1.  **File** -> **New** -> **Application Project**.
2.  Select the platform you just created.
3.  Name: `mpu_app`.
4.  Domain: `standalone_ps7_cortexa9_0`.
5.  Template: **Hello World** (Good starting point).

### 3. Write the Driver Code (`helloworld.c`)

You need to implement three main things:

#### A. SPI Configuration (AD9122)
The AD9122 starts in a default state and **will not work** until configured.
*   Initialize Zynq SPI driver (`XSpiPs`).
*   Write to AD9122 registers:
    *   **0x00**: Soft Reset.
    *   **0x14**: Data Format (Twos complement).
    *   **0x08**: Input Mode (Word/Byte, DDR/SDR). **Crucial**: Match your RTL (DDR, Word interleaved).
    *   **0x16**: Clock PLL settings (if using internal PLL).
    *   **0x5F**: Power up DAC cores.

#### B. AXI DMA Transfer
*   Initialize DMA driver (`XAxiDma`).
*   Disable interrupts (for simple polling mode).
*   **Data Preparation**:
    *   Create a buffer: `u8 TxBuffer[PACKET_LEN]`.
    *   Fill it with your message (e.g., "Hello World" bytes).
*   **Flush Cache**: `Xil_DCacheFlushRange((UINTPTR)TxBuffer, PACKET_LEN);` (Vital! DMA reads from RAM, not Cache).
*   **Start Transfer**:
    *   `XAxiDma_SimpleTransfer(&DmaInst, (UINTPTR)TxBuffer, PACKET_LEN, XAXIDMA_DMA_TO_DEVICE);`
*   **Wait**:
    *   `while (XAxiDma_Busy(&DmaInst, XAXIDMA_DMA_TO_DEVICE));`

#### C. Main Loop
```c
int main() {
    init_platform();
    print("Initializing System...\n\r");
    
    setup_spi_dac(); // Configure AD9122
    setup_dma();     // Configure AXI DMA
    
    while(1) {
        send_packet_dma(); // Send data
        // Delay or wait for button press
    }
    
    cleanup_platform();
    return 0;
}
```

---

## Phase 5: Running on Hardware

1.  Connect ZedBoard via USB (JTAG/UART).
2.  Connect FMC Card.
3.  Power on.
4.  In Vitis:
    *   Right-click App Project -> **Run As** -> **Launch on Hardware (Single Application Debug)**.
    *   This will:
        1.  Program the FPGA (Bitstream).
        2.  Initialize the PS (fsbl).
        3.  Download the `.elf` (C program).
        4.  Run it.
5.  Open a Serial Terminal (115200 baud) to see your `printf` output.
6.  Connect an Oscilloscope or Spectrum Analyzer to the DAC SMA connectors to verify the RF output.

---

## Troubleshooting Common Issues

*   **No Output on DAC**:
    *   Did you configure the AD9122 via SPI? It won't output anything otherwise.
    *   Is the clock running? Check the DCI pins with a scope.
    *   Is the Reset polarity correct? (RTL uses `rst_n` active low).
*   **DMA Hangs**:
    *   Did you flush the cache (`Xil_DCacheFlushRange`)?
    *   Is the `TLAST` signal handled correctly in RTL?
*   **Garbage Output**:
    *   Check Endianness.
    *   Check Signed vs Unsigned (AD9122 expects 2's complement usually).
    *   Check I/Q swapping.
