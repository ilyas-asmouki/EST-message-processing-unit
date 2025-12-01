IVL ?= iverilog
VVP ?= vvp

HDL_ROOT := ../../..
BUILD_DIR := build
VVP_OUT := $(BUILD_DIR)/interleaver_tb.vvp

RTL_SRCS := \
	$(HDL_ROOT)/rtl/tx_chain/rs/rs_encoder_pkg.sv \
	$(HDL_ROOT)/rtl/tx_chain/interleaver/interleaver_pkg.sv \
	$(HDL_ROOT)/rtl/tx_chain/interleaver/interleaver.sv

SIM_FLAGS ?= -g2012 -Wall -Wno-timescale -DASSERT_ON
INCLUDE_FLAGS := -I $(HDL_ROOT)/rtl/tx_chain/rs -I $(HDL_ROOT)/rtl/tx_chain/interleaver

.PHONY: all test clean

all: test

$(BUILD_DIR):
	@mkdir -p $@

$(VVP_OUT): $(RTL_SRCS) interleaver_tb.sv | $(BUILD_DIR)
	$(IVL) $(SIM_FLAGS) $(INCLUDE_FLAGS) -s interleaver_tb -o $@ $^

test: $(VVP_OUT)
	$(VVP) $<

clean:
	@rm -rf $(BUILD_DIR)
