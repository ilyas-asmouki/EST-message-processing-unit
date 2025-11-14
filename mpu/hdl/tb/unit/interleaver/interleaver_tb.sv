// SPDX-License-Identifier: MIT
// Unit testbench for byte_interleaver

`timescale 1ns/1ps

module interleaver_tb;
  import interleaver_pkg::*;
  import rs_encoder_pkg::*;

  localparam string VEC_DIR     = "../../../vectors/interleaver_smoke/interleaver";
  localparam int    TEST_BLOCKS = 3;
  localparam int    CODE_BYTES  = interleaver_pkg::CODEWORD_BYTES;
  localparam int    TOTAL_BYTES = TEST_BLOCKS * CODE_BYTES;
  localparam int    DEPTH       = 2;

  // Clock/reset
  logic clk;
  logic rst_n;

  // DUT IO
  logic        s_axis_valid;
  logic        s_axis_ready;
  logic [7:0]  s_axis_data;
  logic        s_axis_last;
  logic        s_axis_sop;
  logic        s_axis_is_parity;

  logic        m_axis_valid;
  logic        m_axis_ready;
  logic [7:0]  m_axis_data;
  logic        m_axis_last;
  logic        m_axis_sop;
  logic        m_axis_is_parity;

  // Memories
  logic [7:0] input_mem  [0:TOTAL_BYTES-1];
  logic [7:0] output_mem [0:TOTAL_BYTES-1];

  int unsigned out_idx;
  int unsigned errors;
  logic        inputs_done;

  // Clock generation
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  // Reset
  initial begin
    rst_n = 1'b0;
    repeat (6) @(posedge clk);
    rst_n = 1'b1;
  end

  // Load vectors
  initial begin
    string in_path;
    string out_path;
    in_path  = {VEC_DIR, "/input.hex"};
    out_path = {VEC_DIR, "/output.hex"};
    $display("[TB] Loading %0d interleaver input bytes from %s", TOTAL_BYTES, in_path);
    $readmemh(in_path, input_mem);
    $display("[TB] Loading %0d interleaver expected bytes from %s", TOTAL_BYTES, out_path);
    $readmemh(out_path, output_mem);
  end

  // Instantiate DUT
  byte_interleaver #(
    .DEPTH(DEPTH)
  ) dut (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (s_axis_valid),
    .s_axis_ready     (s_axis_ready),
    .s_axis_data      (s_axis_data),
    .s_axis_last      (s_axis_last),
    .s_axis_sop       (s_axis_sop),
    .s_axis_is_parity (s_axis_is_parity),
    .m_axis_valid     (m_axis_valid),
    .m_axis_ready     (m_axis_ready),
    .m_axis_data      (m_axis_data),
    .m_axis_last      (m_axis_last),
    .m_axis_sop       (m_axis_sop),
    .m_axis_is_parity (m_axis_is_parity)
  );

  // PRNGs for stimulus/ready
  logic [31:0] prng_in;
  logic [31:0] prng_out;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prng_in  <= 32'hF00DCAFE;
      prng_out <= 32'hBADC0DED;
    end else begin
      prng_in  <= {prng_in[30:0], prng_in[31] ^ prng_in[21] ^ prng_in[1] ^ prng_in[0]};
      prng_out <= {prng_out[30:0], prng_out[31] ^ prng_out[21] ^ prng_out[1] ^ prng_out[0]};
    end
  end

  // Input driver with occasional bubbles
  int unsigned in_idx;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_axis_valid     <= 1'b0;
      s_axis_last      <= 1'b0;
      s_axis_sop       <= 1'b0;
      s_axis_is_parity <= 1'b0;
      s_axis_data      <= '0;
      in_idx           <= 0;
      inputs_done      <= 1'b0;
    end else begin
      if (s_axis_valid && s_axis_ready) begin
        in_idx <= in_idx + 1;
        s_axis_valid <= 1'b0;
        if (in_idx == TOTAL_BYTES-1) begin
          inputs_done <= 1'b1;
        end
      end
      if (!inputs_done && !s_axis_valid && (in_idx < TOTAL_BYTES)) begin
        if (prng_in[0]) begin
          s_axis_data      <= input_mem[in_idx];
          s_axis_last      <= ((in_idx % CODE_BYTES) == (CODE_BYTES-1));
          s_axis_sop       <= ((in_idx % CODE_BYTES) == 0);
          s_axis_is_parity <= ((in_idx % CODE_BYTES) >= RS_K);
          s_axis_valid     <= 1'b1;
        end
      end
    end
  end

  // Randomized ready on output
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axis_ready <= 1'b0;
    end else begin
      m_axis_ready <= prng_out[2:0] != 3'b000;
    end
  end

  // Scoreboard
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_idx <= 0;
      errors  <= 0;
    end else if (m_axis_valid && m_axis_ready) begin
      if (out_idx >= TOTAL_BYTES) begin
        errors++;
        $error("[TB] Unexpected extra output byte (idx=%0d)", out_idx);
      end else begin
        logic [7:0] exp_byte;
        int block_pos;
        bit expect_sop;
        bit expect_last;
        bit expect_parity;
        exp_byte = output_mem[out_idx];
        if (exp_byte !== m_axis_data) begin
          errors++;
          $error("[TB] Data mismatch @%0d: got 0x%02x expected 0x%02x",
                 out_idx, m_axis_data, exp_byte);
        end
        block_pos = out_idx % CODE_BYTES;
        expect_sop = (block_pos == 0);
        expect_last = (block_pos == (CODE_BYTES-1));
        expect_parity = (interleave_perm(DEPTH, block_pos) >= RS_K);
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
        if (m_axis_is_parity !== expect_parity) begin
          errors++;
          $error("[TB] parity-flag mismatch @%0d: got %0b expect %0b",
                 out_idx, m_axis_is_parity, expect_parity);
        end
      end
      out_idx <= out_idx + 1;
    end
  end

  // Finish condition
  initial begin
    @(posedge rst_n);
    wait (inputs_done);
    wait (out_idx == TOTAL_BYTES);
    #50;
    if (errors == 0) begin
      $display("[TB] PASS: interleaver output matched all %0d bytes", TOTAL_BYTES);
    end else begin
      $error("[TB] FAIL: %0d mismatches detected", errors);
    end
    $finish;
  end

endmodule
