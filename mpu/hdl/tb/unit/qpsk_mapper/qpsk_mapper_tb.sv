// unit testbench for qpsk_mapper

`timescale 1ns/1ps

module qpsk_mapper_tb;

  localparam string VEC_DIR     = "../../../vectors/rigorous_500/qpsk";
  localparam int    TEST_BLOCKS = 500;
  localparam int    BLOCK_SZ    = 512; // input block size (bytes)
  
  localparam int    TOTAL_IN_BYTES = TEST_BLOCKS * BLOCK_SZ;
  // Output: 4 symbols per byte. Each symbol is 4 bytes (2 bytes I + 2 bytes Q).
  // So total output bytes = TOTAL_IN_BYTES * 4 * 4 = TOTAL_IN_BYTES * 16
  localparam int    TOTAL_OUT_BYTES = TOTAL_IN_BYTES * 16; 
  
  localparam int    MAX_IN_BYTES  = 260000;
  localparam int    MAX_OUT_BYTES = 4200000; // ~4MB

  // clock/reset
  logic clk;
  logic rst_n;

  // dut io
  logic        s_axis_valid;
  logic        s_axis_ready;
  logic [7:0]  s_axis_data;
  logic        s_axis_last;
  logic        s_axis_sop;
  logic        s_axis_is_parity;

  logic        m_axis_valid;
  logic        m_axis_ready;
  logic [15:0] m_axis_i;
  logic [15:0] m_axis_q;
  logic        m_axis_last;
  logic        m_axis_sop;
  logic        m_axis_is_parity;

  // memories
  logic [7:0] input_mem  [0:MAX_IN_BYTES-1];
  logic [7:0] output_mem [0:MAX_OUT_BYTES-1];
  
  int unsigned out_byte_idx;
  int unsigned errors;
  logic        inputs_done;

  // clock generation
  initial begin
    clk = 1'b0;
    forever #5 clk = ~clk;
  end

  // reset
  initial begin
    rst_n = 1'b0;
    repeat (6) @(posedge clk);
    rst_n = 1'b1;
  end

  // load vectors
  initial begin
    string in_path;
    string out_path;
    
    in_path  = {VEC_DIR, "/input.hex"};
    out_path = {VEC_DIR, "/output.hex"};
    
    $display("[TB] Loading qpsk_mapper vectors from %s", VEC_DIR);
    $readmemh(in_path, input_mem);
    $readmemh(out_path, output_mem);
    
    $display("[TB] Expecting %0d input bytes", TOTAL_IN_BYTES);
  end

  // instantiate dut
  qpsk_mapper dut (
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
    .m_axis_i         (m_axis_i),
    .m_axis_q         (m_axis_q),
    .m_axis_last      (m_axis_last),
    .m_axis_sop       (m_axis_sop),
    .m_axis_is_parity (m_axis_is_parity)
  );

  // prngs for stimulus/ready
  logic [31:0] prng_in;
  logic [31:0] prng_out;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prng_in  <= 32'hDEADBEEF;
      prng_out <= 32'hCAFEBABE;
    end else begin
      prng_in  <= {prng_in[30:0], prng_in[31] ^ prng_in[21] ^ prng_in[1] ^ prng_in[0]};
      prng_out <= {prng_out[30:0], prng_out[31] ^ prng_out[21] ^ prng_out[1] ^ prng_out[0]};
    end
  end

  // input driver
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
        if (in_idx == TOTAL_IN_BYTES-1) begin
          inputs_done <= 1'b1;
        end
      end
      
      if (!inputs_done && !s_axis_valid && (in_idx < TOTAL_IN_BYTES)) begin
        // randomize valid
        if (prng_in[0]) begin
          s_axis_data      <= input_mem[in_idx];
          s_axis_last      <= ((in_idx % BLOCK_SZ) == (BLOCK_SZ - 1));
          s_axis_sop       <= ((in_idx % BLOCK_SZ) == 0);
          s_axis_is_parity <= 1'b0; // don't care for now
          s_axis_valid     <= 1'b1;
        end
      end
    end
  end

  // randomized ready on output
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      m_axis_ready <= 1'b0;
    end else begin
      m_axis_ready <= prng_out[2:0] != 3'b000;
    end
  end

  // scoreboard
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_byte_idx <= 0;
      errors       <= 0;
    end else if (m_axis_valid && m_axis_ready) begin
      if (out_byte_idx >= TOTAL_OUT_BYTES) begin
        errors++;
        $error("[TB] Unexpected extra output symbol (byte idx=%0d)", out_byte_idx);
      end else begin
        logic [15:0] exp_i;
        logic [15:0] exp_q;
        
        // Reconstruct expected 16-bit values from Little Endian bytes
        exp_i = {output_mem[out_byte_idx+1], output_mem[out_byte_idx]};
        exp_q = {output_mem[out_byte_idx+3], output_mem[out_byte_idx+2]};
        
        if (exp_i !== m_axis_i) begin
          errors++;
          $error("[TB] I mismatch @byte%0d: got 0x%04x expected 0x%04x",
                 out_byte_idx, m_axis_i, exp_i);
        end
        
        if (exp_q !== m_axis_q) begin
          errors++;
          $error("[TB] Q mismatch @byte%0d: got 0x%04x expected 0x%04x",
                 out_byte_idx, m_axis_q, exp_q);
        end
        
        // Check SOP/LAST
        // We expect 4 symbols per input byte.
        // SOP should be on the FIRST symbol of the FIRST byte of a block.
        // LAST should be on the LAST symbol of the LAST byte of a block.
        
        // Current symbol index (0..N-1)
        // symbol_idx = out_byte_idx / 4
        // symbols_per_block = BLOCK_SZ * 4 = 2048
        
        // int symbol_idx = out_byte_idx / 4;
        // int block_sym_idx = symbol_idx % (BLOCK_SZ * 4);
        
        // bit expect_sop = (block_sym_idx == 0);
        // bit expect_last = (block_sym_idx == (BLOCK_SZ * 4 - 1));
        
        // if (m_axis_sop !== expect_sop) begin
        //   errors++;
        //   $error("[TB] sop mismatch @byte%0d: got %0b expect %0b", out_byte_idx, m_axis_sop, expect_sop);
        // end
        
        // if (m_axis_last !== expect_last) begin
        //   errors++;
        //   $error("[TB] last mismatch @byte%0d: got %0b expect %0b", out_byte_idx, m_axis_last, expect_last);
        // end
        
      end
      out_byte_idx <= out_byte_idx + 4;
    end
  end

  // finish condition
  initial begin
    @(posedge rst_n);
    wait (inputs_done);
    // wait for all output bytes to be consumed
    // we might need a timeout if something is wrong
    fork
      begin
        wait (out_byte_idx == TOTAL_OUT_BYTES);
      end
      begin
        #100000000; // timeout
        $error("[TB] Timeout waiting for outputs");
        $finish;
      end
    join_any
    
    #50;
    if (errors == 0) begin
      $display("[TB] PASS: qpsk_mapper output matched all symbols");
    end else begin
      $error("[TB] FAIL: %0d mismatches detected", errors);
    end
    $finish;
  end

endmodule
