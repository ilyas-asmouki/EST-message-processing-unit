// unit testbench for conv_encoder

`timescale 1ns/1ps

module conv_encoder_tb;

  localparam string VEC_DIR     = "../../../vectors/rigorous_500/conv";
  localparam int    TEST_BLOCKS = 500;
  localparam int    IN_BLOCK_SZ = 255;
  localparam int    OUT_BLOCK_SZ= 512;
  
  localparam int    IN_BYTES    = TEST_BLOCKS * IN_BLOCK_SZ;
  localparam int    OUT_BYTES   = TEST_BLOCKS * OUT_BLOCK_SZ;
  
  // large buffer for rigorous testing
  localparam int    MAX_IN_BYTES  = 130000; 
  localparam int    MAX_OUT_BYTES = 260000;

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
  logic [7:0]  m_axis_data;
  logic        m_axis_last;
  logic        m_axis_sop;
  logic        m_axis_is_parity;

  // memories
  logic [7:0] input_mem  [0:MAX_IN_BYTES-1];
  logic [7:0] output_mem [0:MAX_OUT_BYTES-1];
  
  int unsigned out_idx;
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
    
    $display("[TB] Loading conv_encoder vectors from %s", VEC_DIR);
    $readmemh(in_path, input_mem);
    $readmemh(out_path, output_mem);
    
    $display("[TB] Expecting %0d input bytes -> %0d output bytes", IN_BYTES, OUT_BYTES);
  end

  // instantiate dut
  conv_encoder dut (
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
        if (in_idx == IN_BYTES-1) begin
          inputs_done <= 1'b1;
        end
      end
      
      if (!inputs_done && !s_axis_valid && (in_idx < IN_BYTES)) begin
        // randomize valid
        if (prng_in[0]) begin
          s_axis_data      <= input_mem[in_idx];
          // 255-byte blocks
          s_axis_last      <= ((in_idx % IN_BLOCK_SZ) == (IN_BLOCK_SZ - 1));
          s_axis_sop       <= ((in_idx % IN_BLOCK_SZ) == 0);
          // parity is last 32 bytes of 255 (from RS/Interleaver/Scrambler context)
          s_axis_is_parity <= ((in_idx % IN_BLOCK_SZ) >= 223);
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
      out_idx <= 0;
      errors  <= 0;
    end else if (m_axis_valid && m_axis_ready) begin
      if (out_idx >= OUT_BYTES) begin
        errors++;
        $error("[TB] Unexpected extra output byte (idx=%0d)", out_idx);
      end else begin
        logic [7:0] exp_byte;
        int block_pos;
        bit expect_sop;
        bit expect_last;
        
        exp_byte = output_mem[out_idx];
        
        if (exp_byte !== m_axis_data) begin
          errors++;
          $error("[TB] Data mismatch @%0d: got 0x%02x expected 0x%02x",
                 out_idx, m_axis_data, exp_byte);
        end
        
        block_pos = out_idx % OUT_BLOCK_SZ;
        expect_sop = (block_pos == 0);
        expect_last = (block_pos == (OUT_BLOCK_SZ - 1));
        
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

  // finish condition
  initial begin
    @(posedge rst_n);
    wait (inputs_done);
    wait (out_idx == OUT_BYTES);
    #50;
    if (errors == 0) begin
      $display("[TB] PASS: conv_encoder output matched all %0d bytes", OUT_BYTES);
    end else begin
      $error("[TB] FAIL: %0d mismatches detected", errors);
    end
    $finish;
  end

endmodule
