// SPDX-License-Identifier: MIT
// Chain testbench: RS -> Interleaver -> Scrambler -> Conv -> Diff -> QPSK -> RRC
// Verifies the full tx_chain against Python golden model vectors.

`timescale 1ns/1ps

module chain_tb;
  import rs_encoder_pkg::*;
  import interleaver_pkg::*;

  // Vector directories
  parameter string RS_VEC_DIR    = "../../../vectors/test_100_blocks/rs";
  parameter string RRC_VEC_DIR   = "../../../vectors/test_100_blocks/rrc";
  
  parameter int    TEST_BLOCKS   = 100;
  localparam int    RS_IN_BYTES   = TEST_BLOCKS * RS_K;
  
  // RRC Output Calculation:
  // Conv Out: 512 bytes/block
  // QPSK Out: 512 * 4 symbols/block
  // RRC Out:  512 * 4 * 4 samples/block = 8192 samples/block
  // Bytes:    8192 * 4 bytes/sample (16-bit I + 16-bit Q) = 32768 bytes/block
  localparam int    FINAL_BYTES   = TEST_BLOCKS * 32768; 
  
  localparam int    MAX_IN_BYTES  = 200000;
  localparam int    MAX_OUT_BYTES = 4000000; // ~4MB

  // Clock/reset
  logic clk;
  logic rst_n;

  // --- Chain Inputs ---
  logic        s_axis_valid;
  logic        s_axis_ready;
  logic [7:0]  s_axis_data;
  logic        s_axis_last;
  logic        s_axis_sop;

  // --- Chain Outputs ---
  logic        m_axis_valid;
  logic signed [15:0] m_axis_i;
  logic signed [15:0] m_axis_q;

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
    out_path = {RRC_VEC_DIR, "/output.hex"};
    
    $display("[TB] Loading Chain Input (RS) from %s", in_path);
    $readmemh(in_path, input_mem);
    
    $display("[TB] Loading Chain Output (RRC) from %s", out_path);
    $readmemh(out_path, output_mem);
    
    $display("[TB] Expecting %0d output bytes", FINAL_BYTES);
  end

  // -------------------------------------------------------------------------
  // Instantiate Top Level
  // -------------------------------------------------------------------------

  tx_chain dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .s_axis_valid (s_axis_valid),
    .s_axis_ready (s_axis_ready),
    .s_axis_data  (s_axis_data),
    .s_axis_last  (s_axis_last),
    .s_axis_sop   (s_axis_sop),
    .m_axis_valid (m_axis_valid),
    .m_axis_i     (m_axis_i),
    .m_axis_q     (m_axis_q)
  );

  // -------------------------------------------------------------------------
  // Stimulus & Checking
  // -------------------------------------------------------------------------

  // PRNG for stimulus valid randomization
  logic [31:0] prng_in;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prng_in  <= 32'h12345678;
    end else begin
      prng_in  <= {prng_in[30:0], prng_in[31] ^ prng_in[21] ^ prng_in[1] ^ prng_in[0]};
    end
  end

  // Input Driver
  int unsigned in_idx;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      s_axis_valid <= 1'b0;
      s_axis_data  <= '0;
      s_axis_last  <= 1'b0;
      s_axis_sop   <= 1'b0; // Initialize SOP
      in_idx       <= 0;
      inputs_done  <= 1'b0;
    end else begin
      if (s_axis_valid && s_axis_ready) begin
        in_idx <= in_idx + 1;
        s_axis_valid <= 1'b0;
        s_axis_sop   <= 1'b0; // Clear SOP after acceptance
        if (in_idx == RS_IN_BYTES - 1) begin
          inputs_done <= 1'b1;
        end
      end

      if (!inputs_done && !s_axis_valid && (in_idx < RS_IN_BYTES)) begin
        // Randomize valid
        if (prng_in[0]) begin
          s_axis_data  <= input_mem[in_idx];
          s_axis_last  <= ((in_idx % RS_K) == (RS_K - 1));
          // SOP logic: First byte of block (idx % 223 == 0)
          s_axis_sop   <= ((in_idx % RS_K) == 0); 
          s_axis_valid <= 1'b1;
        end
      end
    end
  end

  // Scoreboard (Checks RRC Output)
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_byte_idx <= 0;
      errors       <= 0;
    end else if (m_axis_valid) begin
      // RRC output is valid. Check against golden model.
      // Note: Python model might have extra padding at the end.
      // We only check up to FINAL_BYTES or what we receive.
      
      if (out_byte_idx >= FINAL_BYTES) begin
        // If we receive more than expected, it might be padding or error.
        // For now, let's warn but not error if it's just a few samples.
        // But actually, we calculated FINAL_BYTES based on block count.
        // Any extra is unexpected.
        // However, RRC filter in RTL might output a few more samples due to pipeline flushing if we were flushing.
        // But we are not explicitly flushing.
        // Let's count it as error for now to be strict.
        errors++;
        $error("[TB] Unexpected extra output sample (byte idx=%0d)", out_byte_idx);
      end else begin
        logic [15:0] exp_i;
        logic [15:0] exp_q;
        
        // Reconstruct expected 16-bit values from Little Endian bytes
        // Byte order in file: I_LSB, I_MSB, Q_LSB, Q_MSB
        exp_i = {output_mem[out_byte_idx+1], output_mem[out_byte_idx]};
        exp_q = {output_mem[out_byte_idx+3], output_mem[out_byte_idx+2]};
        
        // Allow small mismatch due to rounding differences?
        // Python: np.clip(rrc_i, -32768, 32767).astype(np.int16)
        // RTL: Rounding and saturation.
        // They should match exactly if implementation is identical.
        
        if (exp_i !== m_axis_i) begin
          errors++;
          $error("[TB] I mismatch @byte%0d: got 0x%04x (%d) expected 0x%04x (%d)",
                 out_byte_idx, m_axis_i, m_axis_i, exp_i, exp_i);
        end
        
        if (exp_q !== m_axis_q) begin
          errors++;
          $error("[TB] Q mismatch @byte%0d: got 0x%04x (%d) expected 0x%04x (%d)",
                 out_byte_idx, m_axis_q, m_axis_q, exp_q, exp_q);
        end

        // Visual verification for the user (first few samples)
        if (out_byte_idx < 64) begin
             $display("[VISUAL] Byte %05d | Golden I: %d Q: %d | RTL I: %d Q: %d | %s", 
                      out_byte_idx, $signed(exp_i), $signed(exp_q), $signed(m_axis_i), $signed(m_axis_q), 
                      ((exp_i === m_axis_i) && (exp_q === m_axis_q)) ? "MATCH" : "FAIL");
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
        wait (out_byte_idx >= FINAL_BYTES);
        #1000; // Wait a bit more to ensure no extra data
      end
      begin
        #200000000; // timeout (increased for 100 blocks)
        $error("[TB] Timeout waiting for outputs. Received %0d/%0d bytes", out_byte_idx, FINAL_BYTES);
        $finish;
      end
    join_any
    
    if (errors == 0) begin
      $display("[TB] PASS: Chain (RS->...->RRC) matched all %0d bytes", FINAL_BYTES);
    end else begin
      $error("[TB] FAIL: %0d mismatches detected in chain", errors);
    end
    $finish;
  end

endmodule
