# Makefile for RRC Filter Unit Test

# Paths
# Relative to mpu/hdl/tb/unit/rrc
HDL_ROOT = ../../..
RTL_DIR = $(HDL_ROOT)/rtl/tx_chain/rrc
TB_DIR = .

# Source Files
SRCS = $(RTL_DIR)/rrc_pkg.sv \
       $(RTL_DIR)/rrc_filter.sv \
       $(TB_DIR)/rrc_tb.sv

# Tools
IV = iverilog
VVP = vvp

# Targets
all: sim

sim: rrc_tb.vvp
	$(VVP) rrc_tb.vvp

rrc_tb.vvp: $(SRCS)
	$(IV) -g2012 -o $@ $(SRCS)

clean:
	rm -f *.vvp *.vcd

.PHONY: all sim clean
