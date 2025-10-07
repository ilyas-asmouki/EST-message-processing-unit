# mpu/model/scrambler.py
# LFSR scrambler: p(x) = x^7 + x^4 + 1, seed = 0b1011101 (MSB first)
# Bitwise: out_bit = in_bit XOR lfsr_out
# feedback = s6 ^ s3, shift left, insert feedback at s0
from typing import Tuple
from mpu.model.helpers import bits_from_bytes, bytes_from_bits

DEFAULT_POLY_TAPS = (6, 3)  # zero-based indices of x^7 and x^4 terms in 7-bit state s6..s0
DEFAULT_SEED = 0b1011101    # '1011101' -> s6..s0


def scramble_bits(data: bytes, seed: int = DEFAULT_SEED, taps: Tuple[int,int] = DEFAULT_POLY_TAPS) -> bytes:
    # LFSR state s6..s0 in bits of 'seed' (MSB at s6)
    state = [(seed >> (6 - i)) & 1 for i in range(7)]
    out_bits = []
    for bit in bits_from_bytes(data):
        lfsr_out = state[6]                 # choose MSB as whitening bit (convention)
        out_bits.append(bit ^ lfsr_out)
        feedback = state[taps[0]] ^ state[taps[1]]  # s6 ^ s3
        # shift left: s6 gets s5, ..., s1 gets s0, s0 gets feedback
        state = state[1:] + [feedback]
    return bytes_from_bits(out_bits)

def descramble_bits(data: bytes, seed: int = DEFAULT_SEED, taps: Tuple[int,int] = DEFAULT_POLY_TAPS) -> bytes:
    # same as scrambling for this additive LFSR (self-inverse) 
    return scramble_bits(data, seed, taps)


if __name__ == "__main__":
    data = bytes([0xac])
    bits = list(bits_from_bytes(data))
    print(bits)