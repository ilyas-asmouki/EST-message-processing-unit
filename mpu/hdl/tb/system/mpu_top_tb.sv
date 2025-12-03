// SPDX-License-Identifier: MIT
// System Testbench for FPGA Top
// Verifies the integration of tx_chain and dac_interface.
// Uses the mock ODDR/OBUFDS primitives.

`timescale 1ns/1ps

module mpu_top_tb;

  // Parameters
  parameter string RS_VEC_DIR    = "../../vectors/test_100_blocks/rs";
  parameter string RRC_VEC_DIR   = "../../vectors/test_100_blocks/rrc";
  parameter int    TEST_BLOCKS   = 100;
  localparam int   RS_K          = 223;
  localparam int   RS_IN_BYTES   = TEST_BLOCKS * RS_K;
  localparam int   FINAL_SAMPLES = TEST_BLOCKS * 512 * 4 * 4; // 8192 samples/block

  // Signals
  logic clk;
  logic rst_n;

  // AXI Stream Input
  logic       s_axis_valid;
  logic       s_axis_ready;
  logic [7:0] s_axis_data;
  logic       s_axis_last;

  // DAC Outputs
  logic [15:0] dac_d_p;
  logic [15:0] dac_d_n;
  logic        dac_dci_p;
  logic        dac_dci_n;
  logic        dac_frame_p;
  logic        dac_frame_n;

  // Memories
  logic [7:0] input_mem  [0:RS_IN_BYTES-1];
  logic [7:0] output_mem [0:(FINAL_SAMPLES*4)-1]; // 4 bytes per sample (I+Q)

  // Test State
  int unsigned in_idx;
  int unsigned out_sample_idx;
  int unsigned errors;
  logic        inputs_done;

  // -------------------------------------------------------------------------
  // Clock Generation
  // -------------------------------------------------------------------------
  initial begin
    clk = 1'b0;
    forever #2 clk = ~clk; // 250 MHz (4ns period)
  end

  // -------------------------------------------------------------------------
  // DUT Instantiation
  // -------------------------------------------------------------------------
  mpu_top dut (
    .clk          (clk),
    .rst_n        (rst_n),
    .s_axis_valid (s_axis_valid),
    .s_axis_ready (s_axis_ready),
    .s_axis_data  (s_axis_data),
    .s_axis_last  (s_axis_last),
    .dac_d_p      (dac_d_p),
    .dac_d_n      (dac_d_n),
    .dac_dci_p    (dac_dci_p),
    .dac_dci_n    (dac_dci_n),
    .dac_frame_p  (dac_frame_p),
    .dac_frame_n  (dac_frame_n)
  );

  // -------------------------------------------------------------------------
  // Test Setup
  // -------------------------------------------------------------------------
  initial begin
    $readmemh({RS_VEC_DIR, "/input.hex"}, input_mem);
    $readmemh({RRC_VEC_DIR, "/output.hex"}, output_mem);
    
    rst_n = 1'b0;
    s_axis_valid = 1'b0;
    s_axis_data = '0;
    s_axis_last = 1'b0;
    in_idx = 0;
    out_sample_idx = 0;
    errors = 0;
    inputs_done = 1'b0;

    repeat (20) @(posedge clk);
    rst_n = 1'b1;
  end

  // -------------------------------------------------------------------------
  // Input Driver
  // -------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
        s_axis_valid <= 1'b0;
        in_idx <= 0;
        inputs_done <= 1'b0;
        s_axis_data <= '0;
        s_axis_last <= 1'b0;
    end else begin
        if (!inputs_done) begin
            if (!s_axis_valid) begin
                // Initial valid assertion
                s_axis_valid <= 1'b1;
                s_axis_data <= input_mem[in_idx];
                s_axis_last <= ((in_idx % RS_K) == (RS_K - 1));
            end else if (s_axis_ready) begin
                // Handshake occurred
                if (in_idx == RS_IN_BYTES - 1) begin
                    s_axis_valid <= 1'b0;
                    inputs_done <= 1'b1;
                end else begin
                    in_idx <= in_idx + 1;
                    s_axis_data <= input_mem[in_idx + 1];
                    s_axis_last <= (((in_idx + 1) % RS_K) == (RS_K - 1));
                end
            end
        end
    end
  end

  // -------------------------------------------------------------------------
  // Output Monitor & Checker
  // -------------------------------------------------------------------------
  // We need to reconstruct the I/Q values from the DDR LVDS outputs.
  // dac_d_p carries the data (since OBUFDS O=I).
  // Data is interleaved: I on Rising, Q on Falling.
  // We sample on edges of dac_dci_p (which is a copy of clk).
  
  logic [15:0] captured_i;
  logic [15:0] captured_q;
  logic        got_i;

  // Sample I on Rising Edge of DCI
  always @(posedge dac_dci_p) begin
    #0.1; // Small delay to resolve delta cycle race
    captured_i <= dac_d_p;
    got_i <= 1'b1;
  end

  // Sample Q on Falling Edge of DCI and Check
  always @(negedge dac_dci_p) begin
    #0.1; // Small delay to resolve delta cycle race
    if (got_i) begin
      captured_q <= dac_d_p;
      
      // Now we have a full I/Q pair. Compare with Golden.
      // Note: The pipeline has latency. We need to skip initial zeros.
      // The golden vectors don't have the pipeline latency zeros.
      // We wait for non-zero or just check stream match.
      
      check_sample(captured_i, dac_d_p); // dac_d_p is Q here
    end
  end

  task check_sample(input logic [15:0] i_val, input logic [15:0] q_val);
    logic [15:0] exp_i;
    logic [15:0] exp_q;
    int byte_offset;

    // Skip initial zeros (pipeline latency)
    if (!(out_sample_idx == 0 && i_val == 0 && q_val == 0)) begin
      if (out_sample_idx < FINAL_SAMPLES) begin
        byte_offset = out_sample_idx * 4;
        exp_i = {output_mem[byte_offset+1], output_mem[byte_offset]};
        exp_q = {output_mem[byte_offset+3], output_mem[byte_offset+2]};

        if (i_val !== exp_i || q_val !== exp_q) begin
          if (errors < 10) begin
            $error("[TB] Mismatch @ sample %0d: RTL(0x%04x, 0x%04x) != EXP(0x%04x, 0x%04x)", 
                   out_sample_idx, i_val, q_val, exp_i, exp_q);
          end
          errors++;
        end else if (out_sample_idx % 1000 == 0) begin
          // Keep alive message
          // $display("[TB] Sample %0d matched.", out_sample_idx);
        end
        
        out_sample_idx++;
      end
    end
  endtask

  // -------------------------------------------------------------------------
  // Timeout & Finish
  // -------------------------------------------------------------------------
  initial begin
    wait(inputs_done);
    wait(out_sample_idx >= FINAL_SAMPLES);
    #1000;
    
    if (errors == 0) begin
      $display("[TB] PASS: FPGA Top integration verified. %0d samples matched.", out_sample_idx);
    end else begin
      $error("[TB] FAIL: %0d mismatches.", errors);
    end
    $finish;
  end
  
  // Safety timeout
  initial begin
    #50000000; // 50ms
    $error("[TB] Timeout.");
    $finish;
  end

endmodule
