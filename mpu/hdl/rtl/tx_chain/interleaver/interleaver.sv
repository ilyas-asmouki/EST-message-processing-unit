// module: byte_interleaver
//
// block interleaver that re-orders each 255-byte RS codeword according to the
// configured depth (I ∈ {1,2,3,4,5,8}). it buffers complete codewords using a
// ping-pong scheme so upstream/downstream stages can run concurrently

`timescale 1ns/1ps

module byte_interleaver #(
  parameter int unsigned DEPTH = 2
) (
  input  logic        clk,
  input  logic        rst_n,

  // Input stream (one RS codeword per block)
  input  logic        s_axis_valid,
  output logic        s_axis_ready,
  input  logic [7:0]  s_axis_data,
  input  logic        s_axis_last,
  input  logic        s_axis_sop,
  input  logic        s_axis_is_parity,

  // Interleaved output stream
  output logic        m_axis_valid,
  input  logic        m_axis_ready,
  output logic [7:0]  m_axis_data,
  output logic        m_axis_last,
  output logic        m_axis_sop,
  output logic        m_axis_is_parity
);
  import interleaver_pkg::*;

  localparam int WORDS_PER_BLOCK = CODEWORD_BYTES;
  typedef enum logic {BUF0 = 1'b0, BUF1 = 1'b1} buf_sel_e;

  // storage for two ping-pong buffers
  logic [7:0] data_mem   [2][0:WORDS_PER_BLOCK-1];
  logic       parity_mem [2][0:WORDS_PER_BLOCK-1];

  // loader bookkeeping
  logic        load_active_q, load_active_d;
  buf_sel_e    load_sel_q, load_sel_d;
  logic [8:0]  load_idx_q, load_idx_d;

  // sender bookkeeping
  logic        send_active_q, send_active_d;
  buf_sel_e    send_sel_q, send_sel_d;
  logic [8:0]  send_idx_q, send_idx_d;

  // ready flags for completed buffers
  logic [1:0] buf_ready_q, buf_ready_d;

  logic input_fire;
  logic output_fire;

  // oarameter guard
  initial begin
    if (!depth_supported(DEPTH)) begin
      $error("byte_interleaver: unsupported DEPTH=%0d (allowed: 1,2,3,4,5,8)", DEPTH);
    end
  end

  // io handshakes
  assign s_axis_ready = load_active_q;
  assign m_axis_valid = send_active_q;
  assign m_axis_sop   = send_active_q && (send_idx_q == 0);
  assign m_axis_last  = send_active_q && (send_idx_q == WORDS_PER_BLOCK-1);

  assign input_fire  = s_axis_valid && s_axis_ready;
  assign output_fire = m_axis_valid && m_axis_ready;

  // combinatorial control: producer/consumer state machines
  always_comb begin
    load_active_d = load_active_q;
    load_sel_d    = load_sel_q;
    load_idx_d    = load_idx_q;

    send_active_d = send_active_q;
    send_sel_d    = send_sel_q;
    send_idx_d    = send_idx_q;

    buf_ready_d   = buf_ready_q;

    // advance sender if currently streaming
    if (send_active_q && output_fire) begin
      if (send_idx_q == WORDS_PER_BLOCK-1) begin
        send_active_d = 1'b0;
        send_idx_d    = '0;
      end else begin
        send_idx_d = send_idx_q + 1;
      end
    end

    // consume input bytes into current load buffer
    if (load_active_q && input_fire) begin
      load_idx_d = load_idx_q + 1;
      if (s_axis_last) begin
        buf_ready_d[load_sel_q] = 1'b1;
        load_active_d = 1'b0;
        load_idx_d    = '0;
      end
    end

    // start sending if idle and a buffer is ready
    if (!send_active_d) begin
      if (buf_ready_d[BUF0]) begin
        send_active_d    = 1'b1;
        send_sel_d       = BUF0;
        send_idx_d       = '0;
        buf_ready_d[BUF0]= 1'b0;
      end else if (buf_ready_d[BUF1]) begin
        send_active_d    = 1'b1;
        send_sel_d       = BUF1;
        send_idx_d       = '0;
        buf_ready_d[BUF1]= 1'b0;
      end
    end

    // assign a buffer for loading when idle
    if (!load_active_d) begin
      if (!buf_ready_d[BUF0] && !(send_active_d && (send_sel_d == BUF0))) begin
        load_active_d = 1'b1;
        load_sel_d    = BUF0;
        load_idx_d    = '0;
      end else if (!buf_ready_d[BUF1] && !(send_active_d && (send_sel_d == BUF1))) begin
        load_active_d = 1'b1;
        load_sel_d    = BUF1;
        load_idx_d    = '0;
      end
    end
  end

  // sequential updates
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      load_active_q <= 1'b1;
      load_sel_q    <= BUF0;
      load_idx_q    <= '0;
      send_active_q <= 1'b0;
      send_sel_q    <= BUF0;
      send_idx_q    <= '0;
      buf_ready_q   <= '0;
    end else begin
      load_active_q <= load_active_d;
      load_sel_q    <= load_sel_d;
      load_idx_q    <= load_idx_d;
      send_active_q <= send_active_d;
      send_sel_q    <= send_sel_d;
      send_idx_q    <= send_idx_d;
      buf_ready_q   <= buf_ready_d;
    end
  end

  // data storage
  always_ff @(posedge clk) begin
    if (load_active_q && input_fire) begin
      data_mem[load_sel_q][load_idx_q]   <= s_axis_data;
      parity_mem[load_sel_q][load_idx_q] <= s_axis_is_parity;
    end
  end

  // output selection
  logic [7:0] data_word;
  logic       parity_word;

  always_comb begin
    data_word   = '0;
    parity_word = 1'b0;
    if (send_active_q) begin
      int unsigned src_idx;
      src_idx = interleave_perm(DEPTH, send_idx_q);
      data_word   = data_mem[send_sel_q][src_idx];
      parity_word = parity_mem[send_sel_q][src_idx];
    end
  end

  assign m_axis_data      = data_word;
  assign m_axis_is_parity = parity_word;

  // simple assertions
`ifdef ASSERT_ON
  always @(posedge clk) begin
    if (rst_n && load_active_q && input_fire) begin
      if ((load_idx_q == WORDS_PER_BLOCK-1) && !s_axis_last) begin
        $error("interleaver: s_axis_last must assert on byte %0d", WORDS_PER_BLOCK-1);
      end
      if (s_axis_sop && (load_idx_q != 0)) begin
        $error("interleaver: s_axis_sop asserted mid-block (idx=%0d)", load_idx_q);
      end
      if (s_axis_last && (load_idx_q != WORDS_PER_BLOCK-1)) begin
        $error("interleaver: s_axis_last early (idx=%0d)", load_idx_q);
      end
    end
  end
`endif

endmodule : byte_interleaver
