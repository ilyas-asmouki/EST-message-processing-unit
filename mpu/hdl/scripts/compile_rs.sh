#!/bin/bash
# mpu/hdl/scripts/compile_rs.sh
# Compile RS encoder with iverilog
#
# Usage: ./compile_rs.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}RS Encoder - iverilog compilation${NC}"
echo "=================================="

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HDL_DIR="$(dirname "$SCRIPT_DIR")"
RTL_DIR="$HDL_DIR/rtl/reed_solomon"
TB_DIR="$HDL_DIR/sim/tb"

# Output
BUILD_DIR="$HDL_DIR/build"
mkdir -p "$BUILD_DIR"

OUTPUT="$BUILD_DIR/tb_rs_encoder.vvp"

echo "RTL directory: $RTL_DIR"
echo "TB directory:  $TB_DIR"
echo "Output:        $OUTPUT"
echo ""

# Check if files exist
if [ ! -f "$RTL_DIR/rs_encoder_top.sv" ]; then
    echo -e "${RED}Error: rs_encoder_top.sv not found${NC}"
    exit 1
fi

if [ ! -f "$TB_DIR/tb_rs_encoder.sv" ]; then
    echo -e "${RED}Error: tb_rs_encoder.sv not found${NC}"
    exit 1
fi

# Compile with iverilog
echo "Compiling with iverilog..."
iverilog -g2012 \
    -o "$OUTPUT" \
    -I "$RTL_DIR" \
    "$RTL_DIR"/*.sv \
    "$TB_DIR/tb_rs_encoder.sv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Compilation successful!${NC}"
    echo ""
    echo "To run simulation:"
    echo "  cd $BUILD_DIR"
    echo "  vvp tb_rs_encoder.vvp"
    echo ""
    echo "To view waveforms:"
    echo "  code --open-url vscode://tomoki1207.pdf tb_rs_encoder.vcd"
    exit 0
else
    echo -e "${RED}Compilation failed${NC}"
    exit 1
fi
