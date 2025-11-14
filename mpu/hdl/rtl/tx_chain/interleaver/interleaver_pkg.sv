// SPDX-License-Identifier: MIT
// Package: RS interleaver helpers (depth validation + permutation logic).

package interleaver_pkg;

  localparam int CODEWORD_BYTES = 255;

  function automatic bit depth_supported(int depth);
    case (depth)
      1,2,3,4,5,8: return 1'b1;
      default:     return 1'b0;
    endcase
  endfunction

  // Return the original byte index that produces output position 'idx' for
  // the requested interleaver depth.
  function automatic int unsigned interleave_perm(input int depth, input int idx);
    int rows;
    int total;
    int full_cols;
    int extra_rows;
    int cols;
    int r;
    int c;

    rows = depth;
    total = CODEWORD_BYTES;
    full_cols = total / rows;
    extra_rows = total % rows;
    cols = full_cols + ((extra_rows != 0) ? 1 : 0);

    if (idx < rows * full_cols) begin
      c = idx / rows;
      r = idx % rows;
    end else begin
      c = full_cols;
      r = idx - (rows * full_cols);
    end

    return r * cols + c;
  endfunction

endpackage : interleaver_pkg
