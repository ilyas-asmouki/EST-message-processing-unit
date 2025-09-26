# mpu/model/interleaver.py
# Byte-wise block interleaver (per RS(255,223) codeword) with depths I in {1,2,3,4,5,8}
# For each 255-byte codeword, we write bytes row-wise into an I*ceil(255/I) matrix
# then read them out column-wise (skipping empty cells)
# This spreads bursts of errors
# while staying simple and deterministic for the golden model
#
# Interleave/deinterleave are exact inverses per block. For I=1, it's a nop

from typing import List
import math

POSSIBLE_DEPTHS = {1, 2, 3, 4, 5, 8}
CODEWORD_BYTES = 255

def _perm_indices_for_block(block_len: int, I: int) -> List[int]:
    # builds a permutation that maps original byte positions -> interleaved positions
    # construction:
    # - rows = I, cols = ceil(block_len / I)
    # - write row-wise, read column-wise (skip out-of-range)
    if I == 1:
        return list(range(block_len))

    rows = I
    cols = math.ceil(block_len / rows)

    # fill row wise with original indices, pad with -1 to mark empty cells.
    grid: List[List[int]] = [[-1] * cols for _ in range(rows)]
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < block_len:
                grid[r][c] = idx
                idx += 1

    # read column-wise to get output order
    order: List[int] = []
    for c in range(cols):
        for r in range(rows):
            v = grid[r][c]
            if v != -1:
                order.append(v)

    # order[k] = source_index that moves to position k
    # We want perm such that out[k] = in[perm[k]]
    perm = order
    assert len(perm) == block_len
    return perm

def _invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv

def interleave(data: bytes, I: int, block_len: int = CODEWORD_BYTES) -> bytes:
    # 'data' lengrh must be a multiple of block_len
    # returns interleaved bytes
    if len(data) % block_len != 0:
        raise ValueError(f"Input length must be a multiple of {block_len} bytes (got {len(data)})")
    perm = _perm_indices_for_block(block_len, I)
    out = bytearray(len(data))
    for off in range(0, len(data), block_len):
        block = data[off:off + block_len]
        # out[off + k] = block[perm[k]]
        for k, src in enumerate(perm):
            out[off + k] = block[src]
    return bytes(out)

def deinterleave(data: bytes, I: int, block_len: int = CODEWORD_BYTES) -> bytes:
    # inverse of interleave. restore original codeword
    if len(data) % block_len != 0:
        raise ValueError(f"Input length must be a multiple of {block_len} bytes (got {len(data)})")
    perm = _perm_indices_for_block(block_len, I)
    inv = _invert_perm(perm)
    out = bytearray(len(data))
    for off in range(0, len(data), block_len):
        block = data[off:off + block_len]
        # original[k] = interleaved[inv[k]]
        for k, src in enumerate(inv):
            out[off + k] = block[src]
    return bytes(out)

if __name__ == "__main__":
    import os
    for I in sorted(POSSIBLE_DEPTHS):
        # buf = os.urandom(2 * CODEWORD_BYTES)
        buf = bytes(list(range(CODEWORD_BYTES)) * 2)
        enc = interleave(buf, I)
        dec = deinterleave(enc, I)
        # print(f"\n=== Depth I={I} ===")
        # print(f"\n\n\n\n\n\n")
        # print("buf:", str(buf))
        # print(f"\n\n\n\n\n\n")
        # print("enc:", str(enc))
        # print(f"\n\n\n\n\n\n")
        # print("dec:", str(dec))
        assert buf == dec, f"round-trip failed for I={I}"
    
    buf = bytes(list(range(CODEWORD_BYTES)) * 2)
    enc = interleave(buf, 2)
    dec = deinterleave(enc, 2)
    print(f"\n=== Depth I={I} ===")
    print(f"\n\n\n\n\n\n")
    print("buf:", " ".join(str(b) for b in buf))
    print(f"\n\n\n\n\n\n")
    print("enc:", " ".join(str(b) for b in enc))
    print(f"\n\n\n\n\n\n")
    print("dec:", " ".join(str(b) for b in dec))



    print("interleaver.py: self-test OK")
