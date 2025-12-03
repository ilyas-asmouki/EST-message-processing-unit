IVL ?= iverilog
VVP ?= vvp

HDL_ROOT := ../..
BUILD_DIR := build
VVP_OUT := $(BUILD_DIR)/mpu_top_tb.vvp

RTL_SRCS := \
	$(HDL_ROOT)/sim/unisims_mock.sv \
	$(HDL_ROOT)/rtl/tx_chain/rs/rs_encoder_pkg.sv \
	$(HDL_ROOT)/rtl/tx_chain/rs/rs_encoder.sv \
	$(HDL_ROOT)/rtl/tx_chain/interleaver/interleaver_pkg.sv \
	$(HDL_ROOT)/rtl/tx_chain/interleaver/interleaver.sv \
	$(HDL_ROOT)/rtl/tx_chain/scrambler/scrambler.sv \
	$(HDL_ROOT)/rtl/tx_chain/conv_encoder/conv_encoder.sv \
	$(HDL_ROOT)/rtl/tx_chain/diff_encoder/diff_encoder.sv \
	$(HDL_ROOT)/rtl/tx_chain/qpsk_mapper/qpsk_mapper.sv \
	$(HDL_ROOT)/rtl/tx_chain/rrc/rrc_pkg.sv \
	$(HDL_ROOT)/rtl/tx_chain/rrc/rrc_filter.sv \
	$(HDL_ROOT)/rtl/tx_chain/tx_chain.sv \
	$(HDL_ROOT)/rtl/dac_interface/dac_interface.sv \
	$(HDL_ROOT)/rtl/mpu_top.v

SIM_FLAGS ?= -g2012 -Wall -Wno-timescale -DASSERT_ON
INCLUDE_FLAGS := \
	-I $(HDL_ROOT)/rtl/tx_chain/rs \
	-I $(HDL_ROOT)/rtl/tx_chain/interleaver \
	-I $(HDL_ROOT)/rtl/tx_chain/scrambler \
	-I $(HDL_ROOT)/rtl/tx_chain/conv_encoder \
	-I $(HDL_ROOT)/rtl/tx_chain/diff_encoder \
	-I $(HDL_ROOT)/rtl/tx_chain/qpsk_mapper \
	-I $(HDL_ROOT)/rtl/tx_chain/rrc

.PHONY: all test clean

all: test

$(BUILD_DIR):
	@mkdir -p $@

$(VVP_OUT): $(RTL_SRCS) mpu_top_tb.sv | $(BUILD_DIR)
	$(IVL) $(SIM_FLAGS) $(INCLUDE_FLAGS) -s mpu_top_tb -o $@ $^

test: $(VVP_OUT)
	$(VVP) $<

clean:
	@rm -rf $(BUILD_DIR)
