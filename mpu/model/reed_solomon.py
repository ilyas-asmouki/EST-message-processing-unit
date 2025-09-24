# mpu/model/reed_solomon.py
# Minimal, dependency-free Reed–Solomon (255,223) encoder for GF(2^8)
# Field: p(x) = x^8 + x^7 + x^2 + x + 1  -> 0x187 (or 0x87 without the x^8 bit)
# Systematic codeword: [data (223 bytes)] || [parity (32 bytes)]
# Consecutive roots: alpha^(B+i), i=0..(2t-1) with B=1 (common choice)

# 0b1_1000_0111 = 0x187

from typing import List

# code parameters
m = 8                       # num of bits per symbol
n = 255                     # num of symbols per codeword
k = 223                     # num of data symbols per codeword
two_t = n - k               # num of parity symbols
t = two_t // 2              # num of correctable symbols
prim_poly = 0b110000111     # p(x) = x^8 + x^7 + x^2 + x + 1
alpha = 0x02                # generator element (primitive in this field)
first_consec_root = 1       # B = 1

# GF(2^8) tables (log/antilog)
exp = [0] * 2**m * 2   # exp table duplicated to avoid mod 255 during mul
log = [0] * 2**m

def _gf_init():
    x = 1
    for i in range(n):
        exp[i] = x
        log[x] = i
        # multiply by primitive element (x -> x*2 in polynomial basis)
        x <<= 1
        if x & 0x100:  # if degree 8 term appears, reduce by primitive polynomial
            # reduce mod p(x): subtract (XOR) prim_poly
            x ^= prim_poly
    # duplicate
    for i in range(255, 512):
        exp[i] = exp[i - 255]

_gf_init()

def gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return exp[log[a] + log[b]]

def gf_poly_mul(p: List[int], q: List[int]) -> List[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a == 0:
            continue
        for j, b in enumerate(q):
            if b == 0:
                continue
            out[i + j] ^= gf_mul(a, b)
    return out

# generator polynomial g(x) = prod_{i=0}^{2*t-1} (x - alpha^{B+i})
def rs_generator_poly(nsym: int = two_t, B: int = first_consec_root) -> List[int]:
    g = [1]
    for i in range(nsym):
        root = exp[(log[alpha] * (B + i)) % 255]  # alpha^(B+i)
        # (x - root) == (1, root) in coefficients [1, root]
        g = gf_poly_mul(g, [1, root])
    return g

_GEN = rs_generator_poly(two_t, first_consec_root)

# systematic encoder: compute parity as remainder of x^two_t * m(x) mod g(x)
# uses Linear Feedback Shift Register
def rs_encode_block_223(data: bytes, nsym: int = two_t) -> bytes:
    if len(data) != k:
        raise ValueError(f"RS(255,223) block must be {k} bytes, got {len(data)}")
    # remainder register (parity), start at all zeros
    parity = [0] * nsym
    for b in data:
        # feedback = incoming_byte XOR top of remainder
        feedback = b ^ parity[0]
        # shift left by 1 (drop parity[0], append 0 at end)
        parity = parity[1:] + [0]
        if feedback != 0:
            # parity ^= feedback * (g(x) without the leading 1), align degrees
            # g = [1, g1, g2, ..., g_ns]  -> apply to parity positions 0..nsym-1
            for i in range(nsym):
                parity[i] ^= gf_mul(_GEN[i + 1], feedback)
    return data + bytes(parity)

# convenience for streaming multiple blocks (exact multiples of 223)
# raises if data length is not a multiple of 223 (no padding here to keep it simple)
def rs_encode(data: bytes, block_bytes: int = k) -> bytes:
    if len(data) % block_bytes != 0:
        raise ValueError("Input length must be a multiple of 223 bytes (no padding in golden model)")
    out = bytearray()
    for i in range(0, len(data), block_bytes):
        out += rs_encode_block_223(data[i:i+block_bytes])
    return bytes(out)

# simple syndrome helper for testing pipelines later
# compute syndromes S_j = c(alpha^{B+j}), j=0..nsym-1
# all zeros means 'valid codeword' if no erasures/shortening
def rs_syndromes(codeword: bytes, nsym: int = two_t, B: int = first_consec_root) -> List[int]:
    S = []
    for j in range(nsym):
        x = exp[(log[alpha] * (B + j)) % 255]  # alpha^{B+j}
        acc = 0
        for c in codeword:
            acc = gf_mul(acc, x) ^ c
        S.append(acc)
    return S


if __name__ == "__main__":
    # 1. encode a single zero block (223 bytes of 0)
    m = bytes([0] * k)
    cw = rs_encode_block_223(m)
    print(f"Codeword length = {len(cw)} (expect 255)")
    # 2) check syndromes are all-zero for the produced codeword
    S = rs_syndromes(cw)
    print("All-zero syndromes? ", all(s == 0 for s in S))
