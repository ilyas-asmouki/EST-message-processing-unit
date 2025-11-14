// SPDX-License-Identifier: MIT
// Module: rs_encoder
//
// Streaming systematic RS(255,223) encoder. Accepts one byte per cycle using a
// ready/valid handshake, forwards data immediately, and appends 32 parity bytes
// using the same GF(2^8) arithmetic as the python golden model.

`timescale 1ns/1ps

module rs_encoder #(
  parameter int unsigned BLOCK_BYTES  = rs_encoder_pkg::RS_K,
  parameter int unsigned PARITY_BYTES = rs_encoder_pkg::RS_PARITY_BYTES
) (
  input  logic clk,
  input  logic rst_n,

  // Data input stream (223 data bytes per block)
  input  logic        s_axis_valid,
  output logic        s_axis_ready,
  input  logic [7:0]  s_axis_data,
  input  logic        s_axis_last,   // assert with the final data byte of a block

  // Codeword output stream (223 data bytes + 32 parity bytes)
  output logic        m_axis_valid,
  input  logic        m_axis_ready,
  output logic [7:0]  m_axis_data,
  output logic        m_axis_last,   // asserted with the final parity byte
  output logic        m_axis_sop,    // asserted with the first data byte
  output logic        m_axis_is_parity // 0 = data portion, 1 = parity portion
);
  import rs_encoder_pkg::*;

  localparam int BLOCK_CNT_W  = $clog2(BLOCK_BYTES + 1);
  localparam int PARITY_CNT_W = $clog2(PARITY_BYTES + 1);

  typedef enum logic [1:0] {ST_IDLE, ST_DATA, ST_PARITY} state_e;

  state_e state_q, state_d;
  logic [BLOCK_CNT_W-1:0]  data_cnt_q, data_cnt_d;
  logic [PARITY_CNT_W-1:0] parity_idx_q, parity_idx_d;
  rs_byte_t parity_q [0:PARITY_BYTES-1];
  rs_byte_t parity_d [0:PARITY_BYTES-1];

  logic parity_clear;
  logic accept_byte;
  logic parity_advance;

  // ------------------------------------------------------------
  // Handshake routing (pass data through during ST_DATA)
  // ------------------------------------------------------------
  assign m_axis_valid     = (state_q == ST_PARITY) ? 1'b1 : s_axis_valid;
  assign m_axis_data      = (state_q == ST_PARITY) ? parity_q[parity_idx_q] : s_axis_data;
  assign m_axis_last      = (state_q == ST_PARITY) ? (parity_idx_q == (PARITY_BYTES-1)) : 1'b0;
  assign m_axis_sop       = (state_q != ST_PARITY) && (data_cnt_q == '0) && s_axis_valid;
  assign m_axis_is_parity = (state_q == ST_PARITY);

  assign s_axis_ready = (state_q == ST_PARITY) ? 1'b0 : m_axis_ready;

  assign accept_byte   = (state_q != ST_PARITY) && s_axis_valid && s_axis_ready;
  assign parity_advance = (state_q == ST_PARITY) && m_axis_valid && m_axis_ready;

  // ------------------------------------------------------------
  // Parity LFSR (mirrors python implementation)
  // ------------------------------------------------------------
  always_comb begin
    for (int i = 0; i < PARITY_BYTES; i++) begin
      parity_d[i] = parity_q[i];
    end
    if (parity_clear) begin
      for (int i = 0; i < PARITY_BYTES; i++) begin
        parity_d[i] = '0;
      end
    end else if (accept_byte) begin
      rs_byte_t feedback;
      feedback = s_axis_data ^ parity_q[0];
      for (int i = 0; i < PARITY_BYTES-1; i++) begin
        parity_d[i] = parity_q[i+1] ^ gf_mul(feedback, rs_gen_coeff(i));
      end
      parity_d[PARITY_BYTES-1] = gf_mul(feedback, rs_gen_coeff(PARITY_BYTES-1));
    end
  end

  // ------------------------------------------------------------
  // Control FSM
  // ------------------------------------------------------------
  always_comb begin
    state_d      = state_q;
    data_cnt_d   = data_cnt_q;
    parity_idx_d = parity_idx_q;
    parity_clear = 1'b0;

    unique case (state_q)
      ST_IDLE: begin
        if (accept_byte) begin
          state_d    = ST_DATA;
          data_cnt_d = 'd1;
        end
      end
      ST_DATA: begin
        if (accept_byte) begin
          data_cnt_d = data_cnt_q + 1;
          if (s_axis_last) begin
            state_d      = ST_PARITY;
            parity_idx_d = '0;
          end
        end
      end
      ST_PARITY: begin
        if (parity_advance) begin
          if (parity_idx_q == PARITY_BYTES-1) begin
            state_d      = ST_IDLE;
            data_cnt_d   = '0;
            parity_idx_d = '0;
            parity_clear = 1'b1;
          end else begin
            parity_idx_d = parity_idx_q + 1;
          end
        end
      end
      default: begin
        state_d = ST_IDLE;
      end
    endcase
  end

  // ------------------------------------------------------------
  // Sequential storage
  // ------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q      <= ST_IDLE;
      data_cnt_q   <= '0;
      parity_idx_q <= '0;
      for (int i = 0; i < PARITY_BYTES; i++) begin
        parity_q[i] <= '0;
      end
    end else begin
      state_q      <= state_d;
      data_cnt_q   <= data_cnt_d;
      parity_idx_q <= parity_idx_d;
      for (int i = 0; i < PARITY_BYTES; i++) begin
        parity_q[i] <= parity_d[i];
      end
    end
  end

  // ------------------------------------------------------------
  // Assertions (simple immediate checks for sim-only use)
  // ------------------------------------------------------------
`ifdef ASSERT_ON
  always @(posedge clk) begin
    if (rst_n) begin
      if (accept_byte && (data_cnt_q == BLOCK_BYTES-1) && !s_axis_last) begin
        $error("s_axis_last must assert with byte %0d of each block", BLOCK_BYTES-1);
      end
      if (accept_byte && s_axis_last && (data_cnt_q != BLOCK_BYTES-1)) begin
        $error("s_axis_last asserted early (byte %0d)", data_cnt_q);
      end
    end
  end
`endif

endmodule : rs_encoder
