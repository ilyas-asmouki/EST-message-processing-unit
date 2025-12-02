// module: fpga_top
//
// top-level wrapper for the FPGA programmable logic (PL).
// connects the zynq PS (via AXI DMA -> AXI stream) to the TX Chain and DAC
//
// architecture:
// [Zynq PS] --(AXI Stream)--> [FIFO] --(AXI Stream)--> [TX Chain] --(AXI Stream)--> [DAC Interface] --(LVDS)--> [AD9122]
//
// note: The FIFO is instantiated here to handle potential backpressure or clock domain crossing
// if the DMA clock differs from the DAC clock. for this RTL, we assume a single clock domain
// for simplicity, but a FIFO is good practice

`timescale 1ns/1ps

module fpga_top (
  input  logic        clk,       // sys clk (e.g. 100MHz or 250MHz from PLL)
  input  logic        rst_n,

  // AXI-stream slave interface (from zynq DMA)
  input  logic        s_axis_valid,
  output logic        s_axis_ready,
  input  logic [7:0]  s_axis_data,
  input  logic        s_axis_last,
  // note: DMA usually provides TLAST. we use it to mark block boundaries if needed
  // we generate SOP locally or assume TLAST implies next is SOP
  
  // physical interface to AD9122 (FMC)
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
  
  // SOP generation logic
  // we need to generate 'sop' for the tx_chain based on 'last' from the previous transaction
  logic axis_sop;
  logic prev_last;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prev_last <= 1'b1; // assume we start fresh, so next valid is SOP
    end else begin
      if (s_axis_valid && s_axis_ready) begin
        prev_last <= s_axis_last;
      end
    end
  end
  
  assign axis_sop = prev_last; // if previous was last, this one is start

  // TX chain outputs
  logic        chain_m_valid;
  logic signed [15:0] chain_m_i;
  logic signed [15:0] chain_m_q;
  
  // DAC interface ready (always 1, but good to have signal)
  logic        dac_ready;

  // --------------------------------------------------------------------------
  // TX chain instantiation
  // --------------------------------------------------------------------------
  tx_chain chain_inst (
    .clk          (clk),
    .rst_n        (rst_n),
    
    // input from DMA
    .s_axis_valid (s_axis_valid),
    .s_axis_ready (s_axis_ready),
    .s_axis_data  (s_axis_data),
    .s_axis_last  (s_axis_last),
    .s_axis_sop   (axis_sop),
    
    // output to DAC interface
    .m_axis_valid (chain_m_valid),
    .m_axis_i     (chain_m_i),
    .m_axis_q     (chain_m_q)
    // m_axis_ready is implicit inside tx_chain (throttled by RRC)
  );
  
  // --------------------------------------------------------------------------
  // DAC interface instantiation
  // --------------------------------------------------------------------------
  dac_interface dac_inst (
    .clk          (clk),
    .rst_n        (rst_n),
    
    .s_axis_valid (chain_m_valid),
    .s_axis_ready (dac_ready), // ignored by tx_chain as it pushes, but good for debug
    .s_axis_i     (chain_m_i),
    .s_axis_q     (chain_m_q),
    
    .dac_d_p      (dac_d_p),
    .dac_d_n      (dac_d_n),
    .dac_dci_p    (dac_dci_p),
    .dac_dci_n    (dac_dci_n),
    .dac_frame_p  (dac_frame_p),
    .dac_frame_n  (dac_frame_n)
  );

endmodule
