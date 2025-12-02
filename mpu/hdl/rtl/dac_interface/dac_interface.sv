// module: dac_interface
//
// interface to Analog Devices AD9122 DAC
//
// features:
// - accepts 16-bit I/Q samples via AXI-stream
// - drives AD9122 LVDS interface (D[15:0]P/N, DCI, FRAME)
// - uses xilinx ODDR primitives for DDR output (I on rising, Q on falling)
// - uses xilinx OBUFDS primitives for LVDS signaling
//
// notes: This module targets xilinx 7-Series FPGAs (ZedBoard)

`timescale 1ns/1ps

module dac_interface (
  input  logic        clk,       // high-speed DAC clock (e.g. 250 MHz)
  input  logic        rst_n,

  // AXI-Stream input (samples)
  // we expect continuous data. if valid drops, we output zero
  input  logic        s_axis_valid,
  output logic        s_axis_ready,
  input  logic [15:0] s_axis_i,
  input  logic [15:0] s_axis_q,

  // physical interface (to FMC pins)
  output logic [15:0] dac_d_p,
  output logic [15:0] dac_d_n,
  output logic        dac_dci_p,
  output logic        dac_dci_n,
  output logic        dac_frame_p,
  output logic        dac_frame_n
);

  // --------------------------------------------------------------------------
  // internal signals
  // --------------------------------------------------------------------------
  logic [15:0] data_i_reg;
  logic [15:0] data_q_reg;
  logic [15:0] ddr_out; // output of ODDRs (single-ended)
  logic        dci_out; // output of ODDR for clock
  logic        frame_out; // output of ODDR for frame

  // always ready to consume
  // if the upstream cannot keep up, we will just transmit zeros (handled below)
  assign s_axis_ready = 1'b1;

  // --------------------------------------------------------------------------
  // input registering & zero stuffing
  // --------------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      data_i_reg <= '0;
      data_q_reg <= '0;
    end else begin
      if (s_axis_valid) begin
        data_i_reg <= s_axis_i;
        data_q_reg <= s_axis_q;
      end else begin
        // if no data available, send 0 (mid-scale for signed 2's complement)
        data_i_reg <= '0;
        data_q_reg <= '0;
      end
    end
  end

  // --------------------------------------------------------------------------
  // DDR Output Primitives (Data)
  // --------------------------------------------------------------------------
  // AD9122 mode: word interleaved (or similar DDR mode)
  // we drive I on rising edge, Q on falling edge of DCI
  // note: DCI is usually edge-aligned or center-aligned depending on config
  // here we generate DCI aligned with data

  genvar i;
  generate
    for (i = 0; i < 16; i++) begin : gen_data_ddr
      ODDR #(
        .DDR_CLK_EDGE ("SAME_EDGE"), // capture D1/D2 on same rising edge of clk
        .INIT         (1'b0),
        .SRTYPE       ("SYNC")
      ) oddr_data (
        .Q  (ddr_out[i]),
        .C  (clk),
        .CE (1'b1),
        .D1 (data_i_reg[i]), // rising edge data (I)
        .D2 (data_q_reg[i]), // falling edge data (Q)
        .R  (1'b0),
        .S  (1'b0)
      );

      OBUFDS obufds_data (
        .I  (ddr_out[i]),
        .O  (dac_d_p[i]),
        .OB (dac_d_n[i])
      );
    end
  endgenerate

  // --------------------------------------------------------------------------
  // DCI (data clock input) generation
  // --------------------------------------------------------------------------
  // we send a copy of the clock
  // D1=1, D2=0 -> high on rise, low on fall -> Copy of CLK
  
  ODDR #(
    .DDR_CLK_EDGE ("SAME_EDGE"),
    .INIT         (1'b0),
    .SRTYPE       ("SYNC")
  ) oddr_dci (
    .Q  (dci_out),
    .C  (clk),
    .CE (1'b1),
    .D1 (1'b1),
    .D2 (1'b0),
    .R  (1'b0),
    .S  (1'b0)
  );

  OBUFDS obufds_dci (
    .I  (dci_out),
    .O  (dac_dci_p),
    .OB (dac_dci_n)
  );

  // --------------------------------------------------------------------------
  // FRAME Generation
  // --------------------------------------------------------------------------
  // frame can be used to distinguish I/Q
  // usually high for I, low for Q (or similar pattern)
  
  ODDR #(
    .DDR_CLK_EDGE ("SAME_EDGE"),
    .INIT         (1'b0),
    .SRTYPE       ("SYNC")
  ) oddr_frame (
    .Q  (frame_out),
    .C  (clk),
    .CE (1'b1),
    .D1 (1'b1), // frame high during I
    .D2 (1'b0), // frame low during Q
    .R  (1'b0),
    .S  (1'b0)
  );

  OBUFDS obufds_frame (
    .I  (frame_out),
    .O  (dac_frame_p),
    .OB (dac_frame_n)
  );

endmodule
