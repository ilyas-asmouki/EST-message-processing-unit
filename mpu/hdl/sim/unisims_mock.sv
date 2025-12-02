// Mock Xilinx Primitives for Icarus Verilog Simulation

`timescale 1ns/1ps

module ODDR #(
    parameter DDR_CLK_EDGE = "OPPOSITE_EDGE",
    parameter INIT = 1'b0,
    parameter SRTYPE = "SYNC"
)(
    output reg Q,
    input C,
    input CE,
    input D1,
    input D2,
    input R,
    input S
);
    reg d2_latched;

    initial Q = INIT;

    // Simplified model for SAME_EDGE
    // Captures D1 and D2 on rising edge.
    // Drives D1 on rising edge.
    // Drives D2 on falling edge.

    always @(posedge C) begin
        if (R) begin
            Q <= 0;
        end else if (S) begin
            Q <= 1;
        end else if (CE) begin
            Q <= D1;
            d2_latched <= D2;
        end
    end

    always @(negedge C) begin
        if (R) begin
            Q <= 0;
        end else if (S) begin
            Q <= 1;
        end else if (CE) begin
            Q <= d2_latched;
        end
    end

endmodule

module OBUFDS (
    output O,
    output OB,
    input I
);
    assign O = I;
    assign OB = ~I;
endmodule
