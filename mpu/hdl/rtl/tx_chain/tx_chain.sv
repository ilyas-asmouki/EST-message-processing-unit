// module: tx_chain
//
// top-level transmit chain module
// instantiates and connects:
// 1. RS Encoder
// 2. Byte Interleaver
// 3. Scrambler
// 4. Convolutional Encoder
// 5. Differential Encoder
// 6. QPSK Mapper
// 7. RRC Filter
//
// interface:
// - input: AXI-Stream (Bytes)
// - output: AXI-Stream (Samples, I/Q 16-bit)
//
// note: The RRC filter interpolates by 4. The input rate is throttled to 1/4 of the clock rate.

`timescale 1ns/1ps

module tx_chain (
  input  logic        clk,
  input  logic        rst_n,

  // input stream (bytes)
  input  logic        s_axis_valid,
  output logic        s_axis_ready,
  input  logic [7:0]  s_axis_data,
  input  logic        s_axis_last,
  input  logic        s_axis_sop,

  // output stream (samples)
  output logic        m_axis_valid,
  output logic signed [15:0] m_axis_i,
  output logic signed [15:0] m_axis_q
);

  // --------------------------------------------------------------------------
  // internal Signals
  // --------------------------------------------------------------------------

  // RS -> interleaver
  logic        rs_m_valid;
  logic        rs_m_ready;
  logic [7:0]  rs_m_data;
  logic        rs_m_last;
  logic        rs_m_sop;
  logic        rs_m_is_parity;

  // interleaver -> scrambler
  logic        int_m_valid;
  logic        int_m_ready;
  logic [7:0]  int_m_data;
  logic        int_m_last;
  logic        int_m_sop;
  logic        int_m_is_parity;

  // scrambler -> conv encoder
  logic        scr_m_valid;
  logic        scr_m_ready;
  logic [7:0]  scr_m_data;
  logic        scr_m_last;
  logic        scr_m_sop;
  logic        scr_m_is_parity;

  // conv encoder -> diff encoder
  logic        conv_m_valid;
  logic        conv_m_ready;
  logic [7:0]  conv_m_data;
  logic        conv_m_last;
  logic        conv_m_sop;
  logic        conv_m_is_parity;

  // diff encoder -> qpsk mapper
  logic        diff_m_valid;
  logic        diff_m_ready;
  logic [7:0]  diff_m_data;
  logic        diff_m_last;
  logic        diff_m_sop;
  logic        diff_m_is_parity;

  // qpsk mapper -> rrc filter
  logic        qpsk_m_valid;
  logic        qpsk_m_ready;
  logic [15:0] qpsk_m_i;
  logic [15:0] qpsk_m_q;
  logic        qpsk_m_last;
  logic        qpsk_m_sop;
  logic        qpsk_m_is_parity;

  // rrc throttling
  logic [1:0]  rrc_throttle_cnt;
  logic        rrc_ready_strobe;

  // --------------------------------------------------------------------------
  // instantiations
  // --------------------------------------------------------------------------

  // 1. rs encoder
  rs_encoder rs_inst (
    .clk              (clk),
    .rst_n            (rst_n),
    .s_axis_valid     (s_axis_valid),
    .s_axis_ready     (s_axis_ready),
    .s_axis_data      (s_axis_data),
    .s_axis_last      (s_axis_last),
    .m_axis_valid     (rs_m_valid),
    .m_axis_ready     (rs_m_ready),
    .m_axis_data      (rs_m_data),
    .m_axis_last      (rs_m_last),
    .m_axis_sop       (rs_m_sop),
    .m_axis_is_parity (rs_m_is_parity)
  );

  // 2. interleaver
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

  // 3. scrambler
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

  // 4. conv encoder
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

  // 5. diff encoder
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

  // 6. qpsk mapper
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

  // 7. rrc filter
  // throttling logic: rrc interpolates by 4, so we feed it once every 4 cycles.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rrc_throttle_cnt <= '0;
    end else begin
      rrc_throttle_cnt <= rrc_throttle_cnt + 1;
    end
  end

  assign rrc_ready_strobe = (rrc_throttle_cnt == 2'd3);
  assign qpsk_m_ready     = rrc_ready_strobe;

  rrc_filter rrc_inst (
    .clk       (clk),
    .rst_n     (rst_n),
    .valid_in  (qpsk_m_valid && qpsk_m_ready),
    .i_in      (qpsk_m_i),
    .q_in      (qpsk_m_q),
    .valid_out (m_axis_valid),
    .i_out     (m_axis_i),
    .q_out     (m_axis_q)
  );

endmodule : tx_chain
