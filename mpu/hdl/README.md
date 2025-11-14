# HDL Workspace

This tree hosts the FPGA implementation effort. It separates synthesizable RTL, verification assets, tool run scripts, and generated artifacts so each piece of the transmit pipeline can be developed and validated in isolation before integration.

```
hdl/
├── rtl/          # Synthesizable modules (organized by function)
├── tb/           # SystemVerilog/VHDL testbenches and supporting SV/UVM code
├── sim/          # Simulator-specific run scripts, waves, and config files
├── synth/        # Vendor projects (Vivado/Quartus) and build scripts
├── ip/           # Managed/generated IP cores and exported configs
├── scripts/      # Helper utilities (coeff generators, data converters, CI glue)
└── constraints/  # Timing/pin constraints and clocking definitions
```

We will bring blocks online in pipeline order (RS encoder → interleaver → scrambler → …) so that each stage can be verified against vectors from `mpu/model` before moving downstream.
