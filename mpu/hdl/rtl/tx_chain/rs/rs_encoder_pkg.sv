// package: reed-solomon encoder helpers/parameters
//
// captures the GF(2^8) constants shared by the RTL/TBs so that the HDL
// implementation stays aligned with the python golden model

package rs_encoder_pkg;

  // reed–solomon code parameters (RS(255,223))
  localparam int RS_SYM_BITS      = 8;
  localparam int RS_N             = 255;
  localparam int RS_K             = 223;
  localparam int RS_PARITY_BYTES  = RS_N - RS_K;  // 32 parity symbols
  localparam logic [7:0] RS_PRIM_POLY = 8'h87;    // x^8 + x^7 + x^2 + x + 1 (w/o x^8 term)

  typedef logic [RS_SYM_BITS-1:0] rs_byte_t;

  // helper that returns generator coefficients re-ordered for the LFSR
  // idx corresponds to _GEN[RS_PARITY_BYTES-1-idx] in the python model
  function automatic rs_byte_t rs_gen_coeff (input int idx);
    unique case (idx)
      0:  return 8'd149;
      1:  return 8'd063;
      2:  return 8'd158;
      3:  return 8'd151;
      4:  return 8'd242;
      5:  return 8'd060;
      6:  return 8'd185;
      7:  return 8'd229;
      8:  return 8'd153;
      9:  return 8'd181;
      10: return 8'd171;
      11: return 8'd055;
      12: return 8'd177;
      13: return 8'd254;
      14: return 8'd013;
      15: return 8'd130;
      16: return 8'd133;
      17: return 8'd013;
      18: return 8'd084;
      19: return 8'd242;
      20: return 8'd076;
      21: return 8'd232;
      22: return 8'd172;
      23: return 8'd197;
      24: return 8'd251;
      25: return 8'd179;
      26: return 8'd219;
      27: return 8'd205;
      28: return 8'd119;
      29: return 8'd135;
      30: return 8'd182;
      31: return 8'd059;
      default: return '0;
    endcase
  endfunction

  // GF(2^8) multiply using the primitive polynomial above. this mirrors the
  // helper used in the python model and is synthesizable (pure combinational)
  function automatic rs_byte_t gf_mul(rs_byte_t a, rs_byte_t b);
    rs_byte_t product;
    rs_byte_t aa;
    rs_byte_t bb;
    logic carry;
    product = '0;
    aa = a;
    bb = b;
    for (int i = 0; i < RS_SYM_BITS; i++) begin
      if (bb[0]) begin
        product ^= aa;
      end
      carry = aa[RS_SYM_BITS-1];
      aa = aa << 1;
      if (carry) begin
        aa ^= RS_PRIM_POLY;
      end
      bb = bb >> 1;
    end
    return product;
  endfunction

endpackage : rs_encoder_pkg
