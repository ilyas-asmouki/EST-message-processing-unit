// unit testbench for rrc_filter
//
// validates the rrc filter against golden model vectors
// supports randomized input timing (bubbles)

`timescale 1ns / 1ps

module rrc_tb;

  // -------------------------------------------------------------------------
  // Parameters & Configuration
  // -------------------------------------------------------------------------
  localparam string VEC_DIR = "../../../vectors/rrc_random_500/rrc";
  localparam int    MAX_BYTES = 100000; // Enough for ~20k symbols
  localparam int    SAMPLES_PER_SYMBOL = 4;
  localparam int    CLK_PERIOD = 10;

  // -------------------------------------------------------------------------
  // Signals
  // -------------------------------------------------------------------------
  logic clk;
  logic rst_n;

  logic        valid_in;
  logic signed [15:0] i_in;
  logic signed [15:0] q_in;

  logic        valid_out;
  logic signed [15:0] i_out;
  logic signed [15:0] q_out;

  // -------------------------------------------------------------------------
  // Memories
  // -------------------------------------------------------------------------
  logic [7:0] input_mem  [0:MAX_BYTES-1];
  logic [7:0] output_mem [0:MAX_BYTES-1];

  // -------------------------------------------------------------------------
  // DUT Instantiation
  // -------------------------------------------------------------------------
  rrc_filter dut (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid_in),
    .i_in(i_in),
    .q_in(q_in),
    .valid_out(valid_out),
    .i_out(i_out),
    .q_out(q_out)
  );

  // -------------------------------------------------------------------------
  // Clock Generation
  // -------------------------------------------------------------------------
  initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
  end

  // -------------------------------------------------------------------------
  // PRNG
  // -------------------------------------------------------------------------
  logic [31:0] prng_val;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prng_val <= 32'hDEADBEEF;
    end else begin
      prng_val <= {prng_val[30:0], prng_val[31] ^ prng_val[21] ^ prng_val[1] ^ prng_val[0]};
    end
  end

  // -------------------------------------------------------------------------
  // Test Logic
  // -------------------------------------------------------------------------
  int input_byte_cnt = 0;
  int output_byte_cnt = 0;
  int errors = 0;
  int input_idx = 0;
  int output_idx = 0;
  bit inputs_done = 0;

  initial begin
    // 1. Load Vectors
    string in_path, out_path;
    logic [7:0] b0, b1, b2, b3;
    bit loop_active;
    
    in_path = {VEC_DIR, "/input.hex"};
    out_path = {VEC_DIR, "/output.hex"};
    
    $readmemh(in_path, input_mem);
    $readmemh(out_path, output_mem);
    
    // 2. Reset
    rst_n = 0;
    valid_in = 0;
    i_in = 0;
    q_in = 0;
    #(CLK_PERIOD*5);
    rst_n = 1;
    #(CLK_PERIOD*5);

    // 3. Drive Inputs
    @(negedge clk);
    
    loop_active = 1;
    while (input_idx < MAX_BYTES && loop_active) begin
        // Check if we have data
        if (input_idx > 8000) begin
             loop_active = 0;
        end else if ($isunknown(input_mem[input_idx])) begin
             $display("Hit end of input data at index %0d", input_idx);
             loop_active = 0;
        end

        if (loop_active) begin
            // Construct symbol
            b0 = input_mem[input_idx];
            b1 = input_mem[input_idx+1];
            b2 = input_mem[input_idx+2];
            b3 = input_mem[input_idx+3];
            
            i_in = {b1, b0};
            q_in = {b3, b2};
            valid_in = 1;
            
            @(negedge clk);
            valid_in = 0;
            input_idx = input_idx + 4;
            
            // Wait at least 3 cycles (SPS-1)
            repeat(SAMPLES_PER_SYMBOL - 1) @(negedge clk);
            
            // Random extra delay (bubbles)
            if (prng_val[0]) begin
                repeat(prng_val[2:1]) @(negedge clk);
            end
        end
    end
    
    inputs_done = 1;
    
    // Wait for pipeline to drain
    repeat(100) @(posedge clk);
    
    $display("Test finished. Errors: %0d", errors);
    if (errors == 0) begin
        $display("STATUS: PASS");
    end else begin
        $display("STATUS: FAIL");
    end
    $finish;
  end

  // -------------------------------------------------------------------------
  // Scoreboard
  // -------------------------------------------------------------------------
  always @(posedge clk) begin
    if (rst_n && valid_out) begin : scoreboard_check
        logic [7:0] b0, b1, b2, b3;
        logic signed [15:0] exp_i, exp_q;
        
        b0 = output_mem[output_idx];
        b1 = output_mem[output_idx+1];
        b2 = output_mem[output_idx+2];
        b3 = output_mem[output_idx+3];
        
        exp_i = {b1, b0};
        exp_q = {b3, b2};
        
        if (i_out !== exp_i || q_out !== exp_q) begin
            $display("Error at output sample %0d (byte idx %0d):", output_idx/4, output_idx);
            $display("  Expected: I=%d, Q=%d", exp_i, exp_q);
            $display("  Got:      I=%d, Q=%d", i_out, q_out);
            errors++;
        end
        
        output_idx = output_idx + 4;
    end
  end

endmodule
