IVL ?= iverilog
VVP ?= vvp

HDL_ROOT := ../../..
BUILD_DIR := build
VVP_OUT := $(BUILD_DIR)/rs_encoder_tb.vvp

RTL_SRCS := \
	$(HDL_ROOT)/rtl/tx_chain/rs/rs_encoder_pkg.sv \
	$(HDL_ROOT)/rtl/tx_chain/rs/rs_encoder.sv

SIM_FLAGS ?= -g2012 -Wall -Wno-timescale -DASSERT_ON
INCLUDE_FLAGS := -I $(HDL_ROOT)/rtl/tx_chain/rs

.PHONY: all test clean

all: test

$(BUILD_DIR):
	@mkdir -p $@

$(VVP_OUT): $(RTL_SRCS) rs_encoder_tb.sv | $(BUILD_DIR)
	$(IVL) $(SIM_FLAGS) $(INCLUDE_FLAGS) -s rs_encoder_tb -o $@ $^

test: $(VVP_OUT)
	$(VVP) $<

clean:
	@rm -rf $(BUILD_DIR)
