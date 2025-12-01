// module: diff_encoder
//
// bitwise differential encoder (nrzi-style)
// y[i] = x[i] ^ y[i-1], with y[-1] = 0 (reset per block)
//
// input: 512-byte blocks (from convolutional encoder)
// output: 512-byte blocks
//
// the encoder processes 1 byte per cycle.
// since y[i] depends on y[i-1], within a byte we have a dependency chain.
// y[7] = x[7] ^ y_prev
// y[6] = x[6] ^ y[7] = x[6] ^ x[7] ^ y_prev
// ...
// y[0] = x[0] ^ ... ^ x[7] ^ y_prev
//
// this is effectively a prefix-xor operation on the byte combined with the
// accumulated xor from the previous byte.

`timescale 1ns/1ps

module diff_encoder (
  input  logic       clk,
  input  logic       rst_n,

  // input stream (512 bytes/block)
  input  logic       s_axis_valid,
  output logic       s_axis_ready,
  input  logic [7:0] s_axis_data,
  input  logic       s_axis_last,
  input  logic       s_axis_sop,
  input  logic       s_axis_is_parity, // passed through

  // output stream (512 bytes/block)
  output logic       m_axis_valid,
  input  logic       m_axis_ready,
  output logic [7:0] m_axis_data,
  output logic       m_axis_last,
  output logic       m_axis_sop,
  output logic       m_axis_is_parity
);

  // state: accumulated xor from previous bytes in the block
  // reset to 0 at start of block (sop)
  logic prev_bit_q;
  logic prev_bit_d;
  
  logic [7:0] encoded_byte;

  // --------------------------------------------------------------------------
  // combinatorial logic
  // --------------------------------------------------------------------------
  always_comb begin
    logic current_prev;
    logic [7:0] v_encoded;
    
    // if sop, we reset the history (y[-1] = 0)
    // otherwise we use the stored history from the previous byte
    current_prev = s_axis_sop ? 1'b0 : prev_bit_q;
    
    // calculate output bits (msb first: 7, 6, ..., 0)
    // y[7] = x[7] ^ prev
    // y[6] = x[6] ^ y[7]
    // ...
    
    v_encoded[7] = s_axis_data[7] ^ current_prev;
    v_encoded[6] = s_axis_data[6] ^ v_encoded[7];
    v_encoded[5] = s_axis_data[5] ^ v_encoded[6];
    v_encoded[4] = s_axis_data[4] ^ v_encoded[5];
    v_encoded[3] = s_axis_data[3] ^ v_encoded[4];
    v_encoded[2] = s_axis_data[2] ^ v_encoded[3];
    v_encoded[1] = s_axis_data[1] ^ v_encoded[2];
    v_encoded[0] = s_axis_data[0] ^ v_encoded[1];
    
    encoded_byte = v_encoded;
    
    // next state is the last bit of this byte
    prev_bit_d = v_encoded[0];
  end

  // --------------------------------------------------------------------------
  // sequential logic
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev_bit_q <= 1'b0;
    end else if (s_axis_valid && s_axis_ready) begin
      prev_bit_q <= prev_bit_d;
    end
  end

  // --------------------------------------------------------------------------
  // output assignments
  // --------------------------------------------------------------------------
  // simple pass-through with combinational processing (0 latency)
  
  assign s_axis_ready     = m_axis_ready;
  assign m_axis_valid     = s_axis_valid;
  assign m_axis_data      = encoded_byte;
  assign m_axis_last      = s_axis_last;
  assign m_axis_sop       = s_axis_sop;
  assign m_axis_is_parity = s_axis_is_parity;

endmodule : diff_encoder
