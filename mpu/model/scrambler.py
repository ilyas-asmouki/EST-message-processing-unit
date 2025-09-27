# mpu/model/scrambler.py
# LFSR scrambler: p(x) = x^7 + x^4 + 1, seed = 0b1011101 (MSB first)
# Bitwise: out_bit = in_bit XOR lfsr_out
# feedback = s6 ^ s3, shift left, insert feedback at s0
from typing import Iterable, Tuple

DEFAULT_POLY_TAPS = (6, 3)  # zero-based indices of x^7 and x^4 terms in 7-bit state s6..s0
DEFAULT_SEED = 0b1011101    # '1011101' -> s6..s0

def _bits_from_bytes(data: bytes) -> list[int]:
    bits = []
    for b in data:
        for i in range(8):
            bits.append((b >> (7 - i)) & 1)
    return bits


def _bytes_from_bits(bits: Iterable[int]) -> bytes:
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
    if n:  # pad last byte with zeros if not multiple of 8 (shouldnt happen in our pipeline)
        out.append(acc << (8 - n))
    return bytes(out)

def scramble_bits(data: bytes, seed: int = DEFAULT_SEED, taps: Tuple[int,int] = DEFAULT_POLY_TAPS) -> bytes:
    # LFSR state s6..s0 in bits of 'seed' (MSB at s6)
    state = [(seed >> (6 - i)) & 1 for i in range(7)]
    out_bits = []
    for bit in _bits_from_bytes(data):
        lfsr_out = state[6]                 # choose MSB as whitening bit (convention)
        out_bits.append(bit ^ lfsr_out)
        feedback = state[taps[0]] ^ state[taps[1]]  # s6 ^ s3
        # shift left: s6 gets s5, ..., s1 gets s0, s0 gets feedback
        state = state[1:] + [feedback]
    return _bytes_from_bits(out_bits)

def descramble_bits(data: bytes, seed: int = DEFAULT_SEED, taps: Tuple[int,int] = DEFAULT_POLY_TAPS) -> bytes:
    # Same as scrambling for this additive LFSR (self-inverse)
    return scramble_bits(data, seed, taps)


if __name__ == "__main__":
    data = bytes([0xac])
    bits = list(_bits_from_bytes(data))
    print(bits)
