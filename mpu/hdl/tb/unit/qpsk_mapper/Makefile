IVL ?= iverilog
VVP ?= vvp

HDL_ROOT := ../../..
BUILD_DIR := build
VVP_OUT := $(BUILD_DIR)/qpsk_mapper_tb.vvp

RTL_SRCS := \
	$(HDL_ROOT)/rtl/tx_chain/qpsk_mapper/qpsk_mapper.sv

SIM_FLAGS ?= -g2012 -Wall -Wno-timescale -DASSERT_ON
INCLUDE_FLAGS := 

.PHONY: all test clean

all: test

$(BUILD_DIR):
	@mkdir -p $@

$(VVP_OUT): $(RTL_SRCS) qpsk_mapper_tb.sv | $(BUILD_DIR)
	$(IVL) $(SIM_FLAGS) $(INCLUDE_FLAGS) -s qpsk_mapper_tb -o $@ $^

test: $(VVP_OUT)
	$(VVP) $<

clean:
	@rm -rf $(BUILD_DIR)
