// module: rrc_filter
//
// root raised cosine filter (tx side)
// interpolates by 4 and shapes the pulse
// input: 1 symbol (I/Q)
// output: 4 samples (I/Q)

`timescale 1ns / 1ps

module rrc_filter
  import rrc_pkg::*;
(
    input  logic        clk,
    input  logic        rst_n,

    // Input Interface (Symbol Rate)
    // valid_in should be asserted for 1 cycle every 4 cycles
    input  logic        valid_in,
    input  logic signed [15:0] i_in,
    input  logic signed [15:0] q_in,

    // Output Interface (Sample Rate = 4x Symbol Rate)
    output logic        valid_out,
    output logic signed [15:0] i_out,
    output logic signed [15:0] q_out
);

  // internal state

  // shift registers for I and Q (Length = 11)
  // index 0 is the newest symbol
  logic signed [15:0] shift_reg_i [SUB_FILTER_TAPS];
  logic signed [15:0] shift_reg_q [SUB_FILTER_TAPS];

  // phase counter (0 to 3)
  logic [1:0] phase_cnt;
  logic       processing_active;

  // input processing & shift register

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int k = 0; k < SUB_FILTER_TAPS; k++) begin
        shift_reg_i[k] <= '0;
        shift_reg_q[k] <= '0;
      end
      phase_cnt <= '0;
      processing_active <= 1'b0;
    end else begin
      if (valid_in) begin
        // shift in new symbol
        // shift_reg[0] gets new data
        // shift_reg[1] gets old shift_reg[0], etc.
        for (int k = SUB_FILTER_TAPS-1; k > 0; k--) begin
          shift_reg_i[k] <= shift_reg_i[k-1];
          shift_reg_q[k] <= shift_reg_q[k-1];
        end
        shift_reg_i[0] <= i_in;
        shift_reg_q[0] <= q_in;

        // reset phase counter to start generating 4 samples for this new state
        phase_cnt <= '0;
        processing_active <= 1'b1;
      end else if (processing_active) begin
        // increment phase counter
        phase_cnt <= phase_cnt + 1;

        // if we finished 4 phases (0..3) and no new input, we could go idle
        if (phase_cnt == 2'd3) begin
          processing_active <= 1'b0;
        end
      end
    end
  end

  // fir calculation (combinational + registered output)

  // we use a wide accumulator to prevent overflow
  // input: 16 bit, coeff: 18 bit -> product: 34 bit
  // sum of 11 products -> ~38 bits. using 48 bits.
  logic signed [47:0] acc_i;
  logic signed [47:0] acc_q;

  logic signed [17:0] coeff;
  logic signed [33:0] prod_i;
  logic signed [33:0] prod_q;

  always_comb begin
    acc_i = '0;
    acc_q = '0;

    // compute dot product for the current phase
    for (int k = 0; k < SUB_FILTER_TAPS; k++) begin
      coeff = get_coeff(int'(phase_cnt), k);

      prod_i = shift_reg_i[k] * coeff;
      prod_q = shift_reg_q[k] * coeff;

      acc_i = acc_i + prod_i;
      acc_q = acc_q + prod_q;
    end
  end

  // output rounding & saturation

  // rounding: add 0.5 (1 << 15) and truncate (shift right by 16)
  // acc is Q3.31 (approx). we want Q1.15.
  // we shift right by 16.

  logic signed [47:0] acc_i_rounded;
  logic signed [47:0] acc_q_rounded;

  assign acc_i_rounded = acc_i + 48'sd32768; // + 2^(16-1)
  assign acc_q_rounded = acc_q + 48'sd32768;

  logic signed [47:0] val_i_shifted;
  logic signed [47:0] val_q_shifted;

  assign val_i_shifted = acc_i_rounded >>> 16;
  assign val_q_shifted = acc_q_rounded >>> 16;

  // saturation logic
  function automatic logic signed [15:0] saturate(input logic signed [47:0] val);
    if (val > 48'sd32767) begin
      return 16'sd32767;
    end else if (val < -48'sd32768) begin
      return -16'sd32768;
    end else begin
      return val[15:0];
    end
  endfunction

  // register the output
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      i_out <= '0;
      q_out <= '0;
      valid_out <= 1'b0;
    end else begin
      if (processing_active || valid_in) begin
        i_out <= saturate(val_i_shifted);
        q_out <= saturate(val_q_shifted);
        valid_out <= processing_active;
      end else begin
        valid_out <= 1'b0;
      end
    end
  end

endmodule
