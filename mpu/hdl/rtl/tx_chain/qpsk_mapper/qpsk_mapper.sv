// module: qpsk_mapper
//
// maps 8-bit input bytes to 4 sequential QPSK symbols.
// output format: 16-bit signed fixed-point (Q1.15).
//
// mapping (gray):
//   0 -> -23170 (0xA57E)
//   1 -> +23170 (0x5A82)
//
// input byte: [b7 b6 b5 b4 b3 b2 b1 b0]
// symbol 0: (b7, b6) -> (I, Q)
// symbol 1: (b5, b4) -> (I, Q)
// symbol 2: (b3, b2) -> (I, Q)
// symbol 3: (b1, b0) -> (I, Q)
//
// latency: 1 cycle (register input) + serialization

`timescale 1ns/1ps

module qpsk_mapper (
  input  logic        clk,
  input  logic        rst_n,

  // input stream (bytes)
  input  logic        s_axis_valid,
  output logic        s_axis_ready,
  input  logic [7:0]  s_axis_data,
  input  logic        s_axis_last,
  input  logic        s_axis_sop,     // passed through (on first symbol)
  input  logic        s_axis_is_parity, // passed through

  // output stream (16-bit I/Q symbols)
  output logic        m_axis_valid,
  input  logic        m_axis_ready,
  output logic [15:0] m_axis_i,
  output logic [15:0] m_axis_q,
  output logic        m_axis_last,    // asserted on the LAST symbol of the LAST byte
  output logic        m_axis_sop,     // asserted on the FIRST symbol of the FIRST byte
  output logic        m_axis_is_parity
);

  // constants
  localparam signed [15:0] VAL_POS = 16'sd23170; // +A
  localparam signed [15:0] VAL_NEG = -16'sd23170; // -A

  // state
  logic [7:0] data_reg;
  logic       last_reg;
  logic       sop_reg;
  logic       parity_reg;
  logic [1:0] count; // 0..3
  logic       busy;

  // internal signals
  logic       load_en;
  logic       shift_en;
  logic       symbol_valid;
  
  logic       bit_i;
  logic       bit_q;

  // --------------------------------------------------------------------------
  // control logic
  // --------------------------------------------------------------------------
  
  // we accept new data if we are not busy, or if we are on the last count and handshaking
  assign s_axis_ready = (!busy) || (shift_en && (count == 2'd3));

  assign load_en  = s_axis_valid && s_axis_ready;
  assign shift_en = busy && m_axis_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy       <= 1'b0;
      count      <= 2'd0;
      data_reg   <= 8'b0;
      last_reg   <= 1'b0;
      sop_reg    <= 1'b0;
      parity_reg <= 1'b0;
    end else begin
      if (load_en) begin
        busy       <= 1'b1;
        count      <= 2'd0;
        data_reg   <= s_axis_data;
        last_reg   <= s_axis_last;
        sop_reg    <= s_axis_sop;
        parity_reg <= s_axis_is_parity;
      end else if (shift_en) begin
        if (count == 2'd3) begin
          busy <= 1'b0; // done with this byte
        end else begin
          count <= count + 1;
        end
      end
    end
  end

  // --------------------------------------------------------------------------
  // output generation
  // --------------------------------------------------------------------------

  // mux to select bits based on count
  // count 0: bits 7,6
  // count 1: bits 5,4
  // count 2: bits 3,2
  // count 3: bits 1,0
  always_comb begin
    case (count)
      2'd0: begin bit_i = data_reg[7]; bit_q = data_reg[6]; end
      2'd1: begin bit_i = data_reg[5]; bit_q = data_reg[4]; end
      2'd2: begin bit_i = data_reg[3]; bit_q = data_reg[2]; end
      2'd3: begin bit_i = data_reg[1]; bit_q = data_reg[0]; end
      default: begin bit_i = 1'b0; bit_q = 1'b0; end
    endcase
  end

  // map bits to values
  assign m_axis_i = bit_i ? VAL_POS : VAL_NEG;
  assign m_axis_q = bit_q ? VAL_POS : VAL_NEG;

  // valid when busy
  assign m_axis_valid = busy;

  // sideband signals
  // sop only on first symbol (count 0)
  assign m_axis_sop = sop_reg && (count == 2'd0);
  
  // last only on last symbol (count 3)
  assign m_axis_last = last_reg && (count == 2'd3);
  
  // parity passed through for all symbols of the parity bytes
  assign m_axis_is_parity = parity_reg;

endmodule : qpsk_mapper
