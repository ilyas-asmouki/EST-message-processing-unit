# mpu/model/conv_encoder.py
# rate-1/2 convolutional encoder (non-systematic, feed-forward)
# L=7 (m=6), generators G1=171_o, G2=133_o
# conventions:
# - bitstream is MSB-first per byte (same as scrambler)
# - shift register state is [s6, s5, s4, s3, s2, s1, s0] with s6 = newest bit
# - on each input bit u: state <- [u, s6, s5, s4, s3, s2, s1], drop old s0
# - outputs per input bit: emit v1 (G1) then v2 (G2)
# - encoder is reset (all zeros) at the start of EACH 255-byte codeword,
#   and after the data bits of that block we append 6 zero tail bits
#
# this matches the classic CCSDS/NASA (171,133)_oct encoder.

from typing import Iterable, List
from mpu.model.interleaver import CODEWORD_BYTES  # = 255

# code parameters
L = 7       # contraint length
M = L - 1   # memory
G1_OCT = 0o171
G2_OCT = 0o133

# tap indices for G1 and G2 when state = [s6, s5, s4, s3, s2, s1, s0]
# 171_o -> 0b1111001  -> taps at [6,5,4,3,0]
# 133_o -> 0b1011011  -> taps at [6,4,3,1,0]
G1_TAPS = (6, 5, 4, 3, 0)
G2_TAPS = (6, 4, 3, 1, 0)

def _bits_from_bytes(data: bytes) -> List[int]:
    # msb-first bits
    out: List[int] = []
    for b in data:
        for i in range(8):
            out.append((b >> (7 - i)) & 1)
    return out

def _bytes_from_bits(bits: Iterable[int]) -> bytes:
    # pack bits msb-first per byte (pad last byte with zeros if needed)
    out = bytearray()
    acc = 0
    n = 0
    for bit in bits:
        acc = (acc << 1) | (bit & 1)
        n += 1
        if n == 8:
            out.append(acc)
            acc = 0
            n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)

def _parity_from_taps(state: List[int], taps: Iterable[int]) -> int:
    v = 0
    for i in taps:
        v ^= state[i]
    return v

def conv_encode_block_bits(bits: List[int]) -> List[int]:
    # encode a single block of bits with per-block reset and 6 zero tail bits
    # returns the list of encoded bits (alternating v1, v2)

    state = [0] * L  # [s6..s0], all zeros at block start
    out_bits: List[int] = []

    # process data bits
    for u in bits:
        # shift in u at s6 (newest)
        state = [u, state[0], state[1], state[2], state[3], state[4], state[5]]
        v1 = _parity_from_taps(state, G1_TAPS)
        v2 = _parity_from_taps(state, G2_TAPS)
        out_bits.append(v1)
        out_bits.append(v2)

    # append 6 zero tail bits to return to all-zero state
    for _ in range(M):
        u = 0
        state = [u, state[0], state[1], state[2], state[3], state[4], state[5]]
        v1 = _parity_from_taps(state, G1_TAPS)
        v2 = _parity_from_taps(state, G2_TAPS)
        out_bits.append(v1)
        out_bits.append(v2)

    return out_bits

def conv_encode_bytes_per_codeword(data: bytes, codeword_bytes: int = CODEWORD_BYTES) -> bytes:
    # treat 'data' as a concatenation of 255-byte codewords
    # for each 255B chunk:
    # - reset encoder state to zeros
    # - encode all its bits
    # - append 6 zero tail bits
    # output is packed msb-first

    if len(data) % codeword_bytes != 0:
        raise ValueError(f"Convolutional encoder expects multiples of {codeword_bytes} bytes, got {len(data)}.")

    encoded_stream_bits: List[int] = []

    for off in range(0, len(data), codeword_bytes):
        block = data[off:off + codeword_bytes]
        block_bits = _bits_from_bytes(block)  # 255 * 8 bits
        encoded_stream_bits.extend(conv_encode_block_bits(block_bits))

    return _bytes_from_bits(encoded_stream_bits)


def conv_encode(data: bytes) -> bytes:
    # encode a post-scrambler byte stream that consists of whole 255-byte
    # interleaved+scrambled codewords. per-codeword reset + 6 zero tails
    return conv_encode_bytes_per_codeword(data, CODEWORD_BYTES)

if __name__ == "__main__":
    import os
    test = bytes(range(CODEWORD_BYTES))
    out = conv_encode(test)
    # for one block: input bits = 255*8 = 2040, plus 6 tail bits => 2046 input steps
    # output bits = 2 * 2046 = 4092 -> bytes = ceil(4092/8) = 512 +  (4092-4096=-4) -> 512 bytes exactly? No, 4096 would be 512 bytes, 4092 is 511.5 -> 512 bytes after padding
    expected_bits = 2 * ((CODEWORD_BYTES * 8) + M)
    expected_bytes = (expected_bits + 7) // 8
    assert len(out) == expected_bytes, f"Unexpected output length: {len(out)} vs {expected_bytes}"
    print("conv_encoder.py: self-test OK")
