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
    # - encode all its bits (4092 bits)
    # - pad **per block** with 4 zero bits to reach a byte boundary (4096 bits = 512 B)
    # output is a concatenation of 512-byte encoded blocks

    if len(data) % codeword_bytes != 0:
        raise ValueError(f"Convolutional encoder expects multiples of {codeword_bytes} bytes, got {len(data)}.")

    out_bytes = bytearray()

    for off in range(0, len(data), codeword_bytes):
        block = data[off:off + codeword_bytes]
        block_bits = _bits_from_bytes(block)             # 255*8 = 2040
        enc_bits   = conv_encode_block_bits(block_bits)  # 4092 bits (with 6 zero tails)

        # pad **per block** to a byte boundary: add 4 zero bits → 4096 bits (512 bytes)
        pad = (-len(enc_bits)) % 8
        if pad:
            enc_bits.extend([0] * pad)                   # pad = 4

        out_bytes.extend(_bytes_from_bits(enc_bits))     # append exactly 512 bytes for this block

    return bytes(out_bytes)



def conv_encode(data: bytes) -> bytes:
    # encode a post-scrambler byte stream that consists of whole 255-byte
    # interleaved+scrambled codewords. per-codeword reset + 6 zero tails
    return conv_encode_bytes_per_codeword(data, CODEWORD_BYTES)



if __name__ == "__main__":
    from commpy.channelcoding.convcode import Trellis, conv_encode as commpy_conv_encode
    import numpy as np
    import random

    def commpy_encode(data: bytes) -> bytes:
        trellis = Trellis(np.array([L-1]), np.array([[G1_OCT, G2_OCT]], dtype=int))
        out_bits = []
        if len(data) % CODEWORD_BYTES != 0:
            raise ValueError("Input must be multiples of 255 bytes")
        for off in range(0, len(data), CODEWORD_BYTES):
            block = data[off:off+CODEWORD_BYTES]
            bits = np.array(_bits_from_bytes(block), dtype=int)
            enc = commpy_conv_encode(bits, trellis, termination='term')
            out_bits.extend(int(b) for b in enc.tolist())
        return _bytes_from_bits(out_bits)

    def compare(name: str, blk: bytes):
        ours = conv_encode(blk)          # our implementation
        ref  = commpy_encode(blk)        # oracle
        ok   = (ours == ref)
        status = "good" if ok else "bad"
        print(f"[{name}] {status} (len={len(ours)})")
        if not ok:
            for i,(a,b) in enumerate(zip(ours, ref)):
                if a != b:
                    print(f"  first diff at byte {i}: ours={a}, ref={b}")
                    break

    random.seed(0xBEEF)

    blk_zero = bytes([0]*CODEWORD_BYTES)
    blk_ff   = bytes([0xFF]*CODEWORD_BYTES)
    blk_imp  = bytes([0x80] + [0]*(CODEWORD_BYTES-1))
    blk_rand = bytes(random.getrandbits(8) for _ in range(CODEWORD_BYTES))

    compare("all zeros", blk_zero)
    compare("all 0xFF",  blk_ff)
    compare("impulse",   blk_imp)
    for i in range(500):
        blk_rand = bytes(random.getrandbits(8) for _ in range(CODEWORD_BYTES))
        compare(f"random {i}", blk_rand)

    big_block = blk_zero + blk_ff + blk_rand
    compare("3-block mixed", big_block)
