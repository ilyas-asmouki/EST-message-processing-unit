// SPDX-License-Identifier: MIT
// Chain testbench: RS -> Interleaver -> Scrambler -> Conv -> Diff -> QPSK

`timescale 1ns/1ps

module chain_tb;
  import rs_encoder_pkg::*;
  import interleaver_pkg::*;

  // Use the same rigorous vectors
  parameter string RS_VEC_DIR    = "../../../vectors/rigorous_500/rs";
  parameter string QPSK_VEC_DIR  = "../../../vectors/rigorous_500/qpsk";
  
  parameter int    TEST_BLOCKS   = 500;
  localparam int    RS_IN_BYTES   = TEST_BLOCKS * RS_K;
  // Diff output: TEST_BLOCKS * 512 bytes.
  // QPSK output: Diff output * 4 symbols * 4 bytes/symbol = Diff output * 16 bytes.
  localparam int    FINAL_BYTES   = TEST_BLOCKS * 512 * 16; 
  
  localparam int    MAX_IN_BYTES  = 130000;
  localparam int    MAX_OUT_BYTES = 4200000; // ~4MB

  // Clock/reset
  logic clk;
  logic rst_n;

  // --- RS Encoder Inputs ---
  logic        rs_s_valid;
  logic        rs_s_ready;
  logic [7:0]  rs_s_data;
  logic        rs_s_last;

  // --- RS -> Interleaver ---
  logic        rs_m_valid;
  logic        rs_m_ready;
  logic [7:0]  rs_m_data;
  logic        rs_m_last;
  logic        rs_m_sop;
  logic        rs_m_is_parity;

  // --- Interleaver -> Scrambler ---
  logic        int_m_valid;
  logic        int_m_ready;
  logic [7:0]  int_m_data;
  logic        int_m_last;
  logic        int_m_sop;
  logic        int_m_is_parity;

  // --- Scrambler -> Conv Encoder ---
  logic        scr_m_valid;
  logic        scr_m_ready;
  logic [7:0]  scr_m_data;
  logic        scr_m_last;
  logic        scr_m_sop;
  logic        scr_m_is_parity;

  // --- Conv Encoder -> Diff Encoder ---
  logic        conv_m_valid;
  logic        conv_m_ready;
  logic [7:0]  conv_m_data;
  logic        conv_m_last;
  logic        conv_m_sop;
  logic        conv_m_is_parity;

  // --- Diff Encoder -> QPSK Mapper ---
  logic        diff_m_valid;
  logic        diff_m_ready;
  logic [7:0]  diff_m_data;
  logic        diff_m_last;
  logic        diff_m_sop;
  logic        diff_m_is_parity;

  // --- QPSK Mapper Outputs ---
  logic        qpsk_m_valid;
  logic        qpsk_m_ready;
  logic [15:0] qpsk_m_i;
  logic [15:0] qpsk_m_q;
  logic        qpsk_m_last;
  logic        qpsk_m_sop;
  logic        qpsk_m_is_parity;

  // Memories
  logic [7:0] input_mem  [0:MAX_IN_BYTES-1];
  logic [7:0] output_mem [0:MAX_OUT_BYTES-1];

  int unsigned out_byte_idx;
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
    repeat (10) @(posedge clk);
    rst_n = 1'b1;
  end

  // Load vectors
  initial begin
    string in_path;
    string out_path;
    
    in_path  = {RS_VEC_DIR, "/input.hex"};
    out_path = {QPSK_VEC_DIR, "/output.hex"};
    
    $display("[TB] Loading Chain Input (RS) from %s", in_path);
    $readmemh(in_path, input_mem);
    
    $display("[TB] Loading Chain Output (QPSK) from %s", out_path);
    $readmemh(out_path, output_mem);
    
    $display("[TB] Expecting %0d output bytes", FINAL_BYTES);
  end

  // -------------------------------------------------------------------------
  // Instantiate Modules
  // -------------------------------------------------------------------------

  // 1. RS Encoder
  rs_encoder rs_inst (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (rs_s_valid),
    .s_axis_ready     (rs_s_ready),
    .s_axis_data      (rs_s_data),
    .s_axis_last      (rs_s_last),
    .m_axis_valid     (rs_m_valid),
    .m_axis_ready     (rs_m_ready),
    .m_axis_data      (rs_m_data),
    .m_axis_last      (rs_m_last),
    .m_axis_sop       (rs_m_sop),
    .m_axis_is_parity (rs_m_is_parity)
  );

  // 2. Interleaver
  byte_interleaver #(
    .DEPTH(2)
  ) int_inst (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (rs_m_valid),
    .s_axis_ready     (rs_m_ready),
    .s_axis_data      (rs_m_data),
    .s_axis_last      (rs_m_last),
    .s_axis_sop       (rs_m_sop),
    .s_axis_is_parity (rs_m_is_parity),
    .m_axis_valid     (int_m_valid),
    .m_axis_ready     (int_m_ready),
    .m_axis_data      (int_m_data),
    .m_axis_last      (int_m_last),
    .m_axis_sop       (int_m_sop),
    .m_axis_is_parity (int_m_is_parity)
  );

  // 3. Scrambler
  scrambler scr_inst (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (int_m_valid),
    .s_axis_ready     (int_m_ready),
    .s_axis_data      (int_m_data),
    .s_axis_last      (int_m_last),
    .s_axis_sop       (int_m_sop),
    .s_axis_is_parity (int_m_is_parity),
    .m_axis_valid     (scr_m_valid),
    .m_axis_ready     (scr_m_ready),
    .m_axis_data      (scr_m_data),
    .m_axis_last      (scr_m_last),
    .m_axis_sop       (scr_m_sop),
    .m_axis_is_parity (scr_m_is_parity)
  );

  // 4. Conv Encoder
  conv_encoder conv_inst (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (scr_m_valid),
    .s_axis_ready     (scr_m_ready),
    .s_axis_data      (scr_m_data),
    .s_axis_last      (scr_m_last),
    .s_axis_sop       (scr_m_sop),
    .s_axis_is_parity (scr_m_is_parity),
    .m_axis_valid     (conv_m_valid),
    .m_axis_ready     (conv_m_ready),
    .m_axis_data      (conv_m_data),
    .m_axis_last      (conv_m_last),
    .m_axis_sop       (conv_m_sop),
    .m_axis_is_parity (conv_m_is_parity)
  );

  // 5. Diff Encoder
  diff_encoder diff_inst (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (conv_m_valid),
    .s_axis_ready     (conv_m_ready),
    .s_axis_data      (conv_m_data),
    .s_axis_last      (conv_m_last),
    .s_axis_sop       (conv_m_sop),
    .s_axis_is_parity (conv_m_is_parity),
    .m_axis_valid     (diff_m_valid),
    .m_axis_ready     (diff_m_ready),
    .m_axis_data      (diff_m_data),
    .m_axis_last      (diff_m_last),
    .m_axis_sop       (diff_m_sop),
    .m_axis_is_parity (diff_m_is_parity)
  );

  // 6. QPSK Mapper
  qpsk_mapper qpsk_inst (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (diff_m_valid),
    .s_axis_ready     (diff_m_ready),
    .s_axis_data      (diff_m_data),
    .s_axis_last      (diff_m_last),
    .s_axis_sop       (diff_m_sop),
    .s_axis_is_parity (diff_m_is_parity),
    .m_axis_valid     (qpsk_m_valid),
    .m_axis_ready     (qpsk_m_ready),
    .m_axis_i         (qpsk_m_i),
    .m_axis_q         (qpsk_m_q),
    .m_axis_last      (qpsk_m_last),
    .m_axis_sop       (qpsk_m_sop),
    .m_axis_is_parity (qpsk_m_is_parity)
  );

  // -------------------------------------------------------------------------
  // Stimulus & Checking
  // -------------------------------------------------------------------------

  // PRNGs for stimulus/ready
  logic [31:0] prng_in;
  logic [31:0] prng_out;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prng_in  <= 32'h12345678;
      prng_out <= 32'h87654321;
    end else begin
      prng_in  <= {prng_in[30:0], prng_in[31] ^ prng_in[21] ^ prng_in[1] ^ prng_in[0]};
      prng_out <= {prng_out[30:0], prng_out[31] ^ prng_out[21] ^ prng_out[1] ^ prng_out[0]};
    end
  end

  // Input Driver (Feeds RS Encoder)
  int unsigned in_idx;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rs_s_valid <= 1'b0;
      rs_s_data  <= '0;
      rs_s_last  <= 1'b0;
      in_idx     <= 0;
      inputs_done <= 1'b0;
    end else begin
      if (rs_s_valid && rs_s_ready) begin
        in_idx <= in_idx + 1;
        rs_s_valid <= 1'b0;
        if (in_idx == RS_IN_BYTES - 1) begin
          inputs_done <= 1'b1;
        end
      end

      if (!inputs_done && !rs_s_valid && (in_idx < RS_IN_BYTES)) begin
        // Randomize valid
        if (prng_in[0]) begin
          rs_s_data  <= input_mem[in_idx];
          rs_s_last  <= ((in_idx % RS_K) == (RS_K - 1));
          rs_s_valid <= 1'b1;
        end
      end
    end
  end

  // Output Ready Generator (Backpressure on QPSK Mapper)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      qpsk_m_ready <= 1'b0;
    end else begin
      qpsk_m_ready <= (prng_out[2:0] != 3'b000); // ~87% ready
    end
  end

  // Scoreboard (Checks QPSK Output)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_byte_idx <= 0;
      errors       <= 0;
    end else if (qpsk_m_valid && qpsk_m_ready) begin
      if (out_byte_idx >= FINAL_BYTES) begin
        errors++;
        $error("[TB] Unexpected extra output symbol (byte idx=%0d)", out_byte_idx);
      end else begin
        logic [15:0] exp_i;
        logic [15:0] exp_q;
        
        // Reconstruct expected 16-bit values from Little Endian bytes
        exp_i = {output_mem[out_byte_idx+1], output_mem[out_byte_idx]};
        exp_q = {output_mem[out_byte_idx+3], output_mem[out_byte_idx+2]};
        
        if (exp_i !== qpsk_m_i) begin
          errors++;
          $error("[TB] I mismatch @byte%0d: got 0x%04x expected 0x%04x",
                 out_byte_idx, qpsk_m_i, exp_i);
        end
        
        if (exp_q !== qpsk_m_q) begin
          errors++;
          $error("[TB] Q mismatch @byte%0d: got 0x%04x expected 0x%04x",
                 out_byte_idx, qpsk_m_q, exp_q);
        end

        // Visual verification for the user (first few symbols)
        if (out_byte_idx < 64) begin
             $display("[VISUAL] Byte %05d | Golden I: 0x%04x Q: 0x%04x | RTL I: 0x%04x Q: 0x%04x | %s", 
                      out_byte_idx, exp_i, exp_q, qpsk_m_i, qpsk_m_q, 
                      ((exp_i === qpsk_m_i) && (exp_q === qpsk_m_q)) ? "MATCH" : "FAIL");
        end
      end
      out_byte_idx <= out_byte_idx + 4;
    end
  end

  // Finish condition
  initial begin
    @(posedge rst_n);
    wait (inputs_done);
    // wait for all output bytes to be consumed
    fork
      begin
        wait (out_byte_idx == FINAL_BYTES);
      end
      begin
        #100000000; // timeout
        $error("[TB] Timeout waiting for outputs");
        $finish;
      end
    join_any
    
    #100;
    if (errors == 0) begin
      $display("[TB] PASS: Chain (RS->Int->Scr->Conv->Diff->QPSK) matched all %0d bytes", FINAL_BYTES);
    end else begin
      $error("[TB] FAIL: %0d mismatches detected in chain", errors);
    end
    $finish;
  end

endmodule
