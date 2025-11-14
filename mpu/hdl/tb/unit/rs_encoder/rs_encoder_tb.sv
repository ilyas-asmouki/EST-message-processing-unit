// SPDX-License-Identifier: MIT
// Unit testbench for rs_encoder

`timescale 1ns/1ps

module rs_encoder_tb;
  import rs_encoder_pkg::*;

  localparam string VEC_DIR    = "../../../vectors/rs_smoke/rs";
  localparam int    TEST_BLOCKS = 2;
  localparam int    IN_BYTES   = TEST_BLOCKS * RS_K;
  localparam int    OUT_BYTES  = TEST_BLOCKS * (RS_K + RS_PARITY_BYTES);

  // clock/reset
  logic clk;
  logic rst_n;

  // DUT connections
  logic        s_axis_valid;
  logic        s_axis_ready;
  logic [7:0]  s_axis_data;
  logic        s_axis_last;

  logic        m_axis_valid;
  logic        m_axis_ready;
  logic [7:0]  m_axis_data;
  logic        m_axis_last;
  logic        m_axis_sop;
  logic        m_axis_is_parity;

  // reference data
  rs_byte_t input_mem  [0:IN_BYTES-1];
  rs_byte_t output_mem [0:OUT_BYTES-1];

  int unsigned out_idx;
  int unsigned errors;
  logic        inputs_consumed;

  // clock generator
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  // reset pulse
  initial begin
    rst_n = 1'b0;
    repeat (5) @(posedge clk);
    rst_n = 1'b1;
  end

  // load vectors
  initial begin
    string in_path;
    string out_path;
    in_path  = {VEC_DIR, "/input.hex"};
    out_path = {VEC_DIR, "/output.hex"};
    $display("[TB] Loading %0d RS input bytes from %s", IN_BYTES, in_path);
    $readmemh(in_path, input_mem);
    $display("[TB] Loading %0d RS codeword bytes from %s", OUT_BYTES, out_path);
    $readmemh(out_path, output_mem);
  end

  // instantiate DUT
  rs_encoder dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (s_axis_valid),
    .s_axis_ready     (s_axis_ready),
    .s_axis_data      (s_axis_data),
    .s_axis_last      (s_axis_last),
    .m_axis_valid     (m_axis_valid),
    .m_axis_ready     (m_axis_ready),
    .m_axis_data      (m_axis_data),
    .m_axis_last      (m_axis_last),
    .m_axis_sop       (m_axis_sop),
    .m_axis_is_parity (m_axis_is_parity)
  );

  // Input driver
  initial begin
    s_axis_valid = 1'b0;
    s_axis_data  = '0;
    s_axis_last  = 1'b0;
    inputs_consumed = 1'b0;
    @(posedge rst_n);
    @(posedge clk);
    for (int idx = 0; idx < IN_BYTES; idx++) begin
      s_axis_data  <= input_mem[idx];
      s_axis_last  <= ((idx % RS_K) == (RS_K - 1));
      s_axis_valid <= 1'b1;
      // wait for handshake
      do @(posedge clk); while (!s_axis_ready);
    end
    @(posedge clk);
    s_axis_valid     <= 1'b0;
    s_axis_last      <= 1'b0;
    inputs_consumed  <= 1'b1;
  end

  // Ready generator w/ deterministic pseudo-random stalls
  logic [31:0] prng_q;
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prng_q       <= 32'h1ACE_B00C;
      m_axis_ready <= 1'b0;
    end else begin
      prng_q       <= {prng_q[30:0], prng_q[31] ^ prng_q[21] ^ prng_q[1] ^ prng_q[0]};
      m_axis_ready <= (prng_q[2:0] != 3'b000); // ~87% ready
    end
  end

  // Scoreboard
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_idx <= 0;
      errors  <= 0;
    end else if (m_axis_valid && m_axis_ready) begin
      if (out_idx >= OUT_BYTES) begin
        $error("[TB] More output bytes than expected (idx=%0d)", out_idx);
      end else begin
        rs_byte_t exp_byte;
        int code_pos;
        bit expect_parity;
        bit expect_sop;
        bit expect_last;
        exp_byte = output_mem[out_idx];
        if (m_axis_data !== exp_byte) begin
          errors++;
          $error("[TB] Byte mismatch @%0d: got 0x%02x, expected 0x%02x",
                 out_idx, m_axis_data, exp_byte);
        end
        code_pos = out_idx % (RS_K + RS_PARITY_BYTES);
        expect_parity = (code_pos >= RS_K);
        expect_sop    = (code_pos == 0);
        expect_last   = (code_pos == (RS_K + RS_PARITY_BYTES - 1));
        if (m_axis_is_parity !== expect_parity) begin
          errors++;
          $error("[TB] is_parity mismatch @%0d: got %0b expect %0b",
                 out_idx, m_axis_is_parity, expect_parity);
        end
        if (m_axis_sop !== expect_sop) begin
          errors++;
          $error("[TB] sop mismatch @%0d: got %0b expect %0b",
                 out_idx, m_axis_sop, expect_sop);
        end
        if (m_axis_last !== expect_last) begin
          errors++;
          $error("[TB] last mismatch @%0d: got %0b expect %0b",
                 out_idx, m_axis_last, expect_last);
        end
      end
      out_idx <= out_idx + 1;
    end
  end

  // Finish condition
  initial begin
    @(posedge rst_n);
    wait (inputs_consumed);
    wait (out_idx == OUT_BYTES);
    #20;
    if (errors == 0) begin
      $display("[TB] PASS: matched all %0d output bytes across %0d blocks",
               OUT_BYTES, TEST_BLOCKS);
    end else begin
      $error("[TB] FAIL: detected %0d mismatches", errors);
    end
    $finish;
  end

endmodule
