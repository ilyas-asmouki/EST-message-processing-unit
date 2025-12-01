// module: scrambler
//
// additive lfsr scrambler implementing the polynomial x^7 + x^4 + 1.
// unrolls the lfsr 8 times to process one byte per clock cycle.
//
// polynomial: x^7 + x^4 + 1
// seed:       configurable (default 0b1011101)
//
// the lfsr state is re-initialized to seed when s_axis_sop is asserted.

`timescale 1ns/1ps

module scrambler #(
  parameter logic [6:0] SEED = 7'b1011101
) (
  input  logic       clk,
  input  logic       rst_n,

  // input stream
  input  logic       s_axis_valid,
  output logic       s_axis_ready,
  input  logic [7:0] s_axis_data,
  input  logic       s_axis_last,
  input  logic       s_axis_sop,
  input  logic       s_axis_is_parity,

  // output stream
  output logic       m_axis_valid,
  input  logic       m_axis_ready,
  output logic [7:0] m_axis_data,
  output logic       m_axis_last,
  output logic       m_axis_sop,
  output logic       m_axis_is_parity
);

  // lfsr state register
  // lfsr_q[6] corresponds to s6 (msb of seed)
  // lfsr_q[0] corresponds to s0 (lsb of seed)
  logic [6:0] lfsr_q;
  logic [6:0] lfsr_next;
  logic [7:0] scrambled_data;

  // --------------------------------------------------------------------------
  // combinational unrolling (8 bits)
  // --------------------------------------------------------------------------
  always_comb begin
    logic [6:0] state;
    logic       feedback;
    logic       lfsr_out_bit;

    // if sop is asserted, we start this byte with the fresh seed.
    // otherwise, we continue from the current state.
    state = s_axis_sop ? SEED : lfsr_q;

    // process bits from msb (bit 7) to lsb (bit 0)
    // matching the python model's _bits_from_bytes order.
    for (int i = 7; i >= 0; i--) begin
      // python model:
      // lfsr_out = state[6]  (which is s0, our lfsr_q[0])
      // feedback = state[6] ^ state[3] (s0 ^ s3, our lfsr_q[0] ^ lfsr_q[3])
      // state shift: s(k) <= s(k-1), s0 <= feedback
      // in our register {s6...s0}:
      // left shift: {state[5:0], feedback}
      
      lfsr_out_bit = state[0];
      feedback     = state[0] ^ state[3];
      
      scrambled_data[i] = s_axis_data[i] ^ lfsr_out_bit;
      
      state = {state[5:0], feedback};
    end
    
    lfsr_next = state;
  end

  // --------------------------------------------------------------------------
  // sequential logic
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lfsr_q <= SEED;
    end else if (s_axis_valid && s_axis_ready) begin
      lfsr_q <= lfsr_next;
    end
  end

  // --------------------------------------------------------------------------
  // output assignments
  // --------------------------------------------------------------------------
  // simple pass-through with combinational processing.
  // latency = 0 cycles.
  
  assign s_axis_ready     = m_axis_ready;
  assign m_axis_valid     = s_axis_valid;
  assign m_axis_data      = scrambled_data;
  assign m_axis_last      = s_axis_last;
  assign m_axis_sop       = s_axis_sop;
  assign m_axis_is_parity = s_axis_is_parity;

endmodule : scrambler
