// module: conv_encoder
//
// rate-1/2 convolutional encoder (k=7, rate=1/2)
// polynomials: g1 = 171 (octal), g2 = 133 (octal)
//
// input: 255-byte blocks
// output: 512-byte blocks (includes tail bits and padding)
//
// the encoder implements a 2-cycle processing per input byte to generate
// 2 output bytes. it automatically inserts 6 zero tail bits and 4 zero
// padding bits at the end of each block.

`timescale 1ns/1ps

module conv_encoder (
  input  logic       clk,
  input  logic       rst_n,

  // input stream (255 bytes/block)
  input  logic       s_axis_valid,
  output logic       s_axis_ready,
  input  logic [7:0] s_axis_data,
  input  logic       s_axis_last,
  input  logic       s_axis_sop,
  input  logic       s_axis_is_parity, // passed through (conceptually, though output is larger)

  // output stream (512 bytes/block)
  output logic       m_axis_valid,
  input  logic       m_axis_ready,
  output logic [7:0] m_axis_data,
  output logic       m_axis_last,
  output logic       m_axis_sop,
  output logic       m_axis_is_parity
);

  // --------------------------------------------------------------------------
  // parameters & constants
  // --------------------------------------------------------------------------
  // g1 = 171_oct = 111_100_1_bin -> taps: 0, 3, 4, 5, 6
  // g2 = 133_oct = 101_011_1_bin -> taps: 0, 1, 3, 4, 6
  // state mapping: sr[0] is newest bit, sr[6] is oldest
  
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_PROCESS_HI,
    ST_PROCESS_LO,
    ST_TAIL_HI,
    ST_TAIL_LO
  } state_e;

  state_e state_q, state_d;

  // shift register: sr[0] = newest, sr[6] = oldest
  logic [0:6] sr_q, sr_d; 
  
  // input latch
  logic [7:0] data_latch_q, data_latch_d;
  logic       last_latch_q, last_latch_d;
  logic       sop_latch_q, sop_latch_d;
  logic       parity_latch_q, parity_latch_d;

  // output buffer
  logic [7:0] out_data_d;
  logic       out_last_d;
  logic       out_sop_d;
  logic       out_parity_d;

  // helper vars
  logic [1:0] pair;
  logic [0:6] tmp_sr;
  logic [7:0] out_byte;

  // --------------------------------------------------------------------------
  // functions
  // --------------------------------------------------------------------------
  function automatic logic [1:0] encode_bit(input logic u, input logic [0:6] state);
    logic [0:6] next_state;
    logic v1, v2;
    
    // shift in u
    next_state = {u, state[0:5]};
    
    // g1 taps: 0, 3, 4, 5, 6 (on next_state)
    v1 = next_state[0] ^ next_state[3] ^ next_state[4] ^ next_state[5] ^ next_state[6];
    
    // g2 taps: 0, 1, 3, 4, 6 (on next_state)
    v2 = next_state[0] ^ next_state[1] ^ next_state[3] ^ next_state[4] ^ next_state[6];
    
    return {v1, v2};
  endfunction

  // --------------------------------------------------------------------------
  // combinatorial logic
  // --------------------------------------------------------------------------
  always_comb begin
    state_d        = state_q;
    sr_d           = sr_q;
    data_latch_d   = data_latch_q;
    last_latch_d   = last_latch_q;
    sop_latch_d    = sop_latch_q;
    parity_latch_d = parity_latch_q;
    
    s_axis_ready   = 1'b0;
    m_axis_valid   = 1'b0;
    m_axis_data    = '0;
    m_axis_last    = 1'b0;
    m_axis_sop     = 1'b0;
    m_axis_is_parity = 1'b0;

    // default assignments for helpers to avoid latches
    pair = '0;
    tmp_sr = sr_q;
    out_byte = '0;

    case (state_q)
      ST_IDLE: begin
        s_axis_ready = 1'b1;
        if (s_axis_valid) begin
          data_latch_d   = s_axis_data;
          last_latch_d   = s_axis_last;
          sop_latch_d    = s_axis_sop;
          parity_latch_d = s_axis_is_parity;
          
          if (s_axis_sop) begin
            sr_d = '0; // reset on sop
          end
          
          state_d = ST_PROCESS_HI;
        end
      end

      ST_PROCESS_HI: begin
        // process bits 7, 6, 5, 4 (msb first)
        // we need to update sr temporarily to calculate outputs, 
        // but we only commit the final sr state for this cycle.
        
        tmp_sr = sr_q;
        
        // bit 7
        pair = encode_bit(data_latch_q[7], tmp_sr);
        out_byte[7:6] = pair;
        tmp_sr = {data_latch_q[7], tmp_sr[0:5]};
        
        // bit 6
        pair = encode_bit(data_latch_q[6], tmp_sr);
        out_byte[5:4] = pair;
        tmp_sr = {data_latch_q[6], tmp_sr[0:5]};
        
        // bit 5
        pair = encode_bit(data_latch_q[5], tmp_sr);
        out_byte[3:2] = pair;
        tmp_sr = {data_latch_q[5], tmp_sr[0:5]};
        
        // bit 4
        pair = encode_bit(data_latch_q[4], tmp_sr);
        out_byte[1:0] = pair;
        tmp_sr = {data_latch_q[4], tmp_sr[0:5]};
        
        m_axis_data      = out_byte;
        m_axis_valid     = 1'b1;
        m_axis_sop       = sop_latch_q; // sop aligns with first byte of block
        m_axis_is_parity = parity_latch_q;
        
        if (m_axis_ready) begin
          sr_d    = tmp_sr;
          state_d = ST_PROCESS_LO;
        end
      end

      ST_PROCESS_LO: begin
        // process bits 3, 2, 1, 0
        tmp_sr = sr_q;
        
        // bit 3
        pair = encode_bit(data_latch_q[3], tmp_sr);
        out_byte[7:6] = pair;
        tmp_sr = {data_latch_q[3], tmp_sr[0:5]};
        
        // bit 2
        pair = encode_bit(data_latch_q[2], tmp_sr);
        out_byte[5:4] = pair;
        tmp_sr = {data_latch_q[2], tmp_sr[0:5]};
        
        // bit 1
        pair = encode_bit(data_latch_q[1], tmp_sr);
        out_byte[3:2] = pair;
        tmp_sr = {data_latch_q[1], tmp_sr[0:5]};
        
        // bit 0
        pair = encode_bit(data_latch_q[0], tmp_sr);
        out_byte[1:0] = pair;
        tmp_sr = {data_latch_q[0], tmp_sr[0:5]};
        
        m_axis_data      = out_byte;
        m_axis_valid     = 1'b1;
        m_axis_is_parity = parity_latch_q;
        
        if (m_axis_ready) begin
          sr_d = tmp_sr;
          if (last_latch_q) begin
            state_d = ST_TAIL_HI;
          end else begin
            // check if next data is already available to save a cycle?
            // for simplicity, go back to idle. 
            // optimization: if s_axis_valid is high, we could latch and go to PROCESS_HI directly.
            // let's stick to simple first.
            state_d = ST_IDLE;
          end
        end
      end

      ST_TAIL_HI: begin
        // shift in 4 zeros
        tmp_sr = sr_q;
        
        // Unrolled loop for clarity
        // 0
        pair = encode_bit(1'b0, tmp_sr);
        out_byte[7:6] = pair;
        tmp_sr = {1'b0, tmp_sr[0:5]};
        // 0
        pair = encode_bit(1'b0, tmp_sr);
        out_byte[5:4] = pair;
        tmp_sr = {1'b0, tmp_sr[0:5]};
        // 0
        pair = encode_bit(1'b0, tmp_sr);
        out_byte[3:2] = pair;
        tmp_sr = {1'b0, tmp_sr[0:5]};
        // 0
        pair = encode_bit(1'b0, tmp_sr);
        out_byte[1:0] = pair;
        tmp_sr = {1'b0, tmp_sr[0:5]};
        
        m_axis_data      = out_byte;
        m_axis_valid     = 1'b1;
        m_axis_is_parity = 1'b1; // tail is parity-like overhead
        
        if (m_axis_ready) begin
          sr_d    = tmp_sr;
          state_d = ST_TAIL_LO;
        end
      end

      ST_TAIL_LO: begin
        // shift in 2 zeros, then append 4 padding zeros
        tmp_sr = sr_q;
        
        // 0 (tail bit 5)
        pair = encode_bit(1'b0, tmp_sr);
        out_byte[7:6] = pair;
        tmp_sr = {1'b0, tmp_sr[0:5]};
        
        // 0 (tail bit 6)
        pair = encode_bit(1'b0, tmp_sr);
        out_byte[5:4] = pair;
        tmp_sr = {1'b0, tmp_sr[0:5]};
        
        // padding (4 zeros)
        out_byte[3:0] = 4'b0000;
        
        m_axis_data      = out_byte;
        m_axis_valid     = 1'b1;
        m_axis_last      = 1'b1; // end of 512-byte block
        m_axis_is_parity = 1'b1;
        
        if (m_axis_ready) begin
          sr_d    = tmp_sr; // technically don't care, will reset on next SOP
          state_d = ST_IDLE;
        end
      end
      
      default: state_d = ST_IDLE;
    endcase
  end

  // --------------------------------------------------------------------------
  // sequential logic
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state_q        <= ST_IDLE;
      sr_q           <= '0;
      data_latch_q   <= '0;
      last_latch_q   <= '0;
      sop_latch_q    <= '0;
      parity_latch_q <= '0;
    end else begin
      state_q        <= state_d;
      sr_q           <= sr_d;
      data_latch_q   <= data_latch_d;
      last_latch_q   <= last_latch_d;
      sop_latch_q    <= sop_latch_d;
      parity_latch_q <= parity_latch_d;
    end
  end

endmodule : conv_encoder
