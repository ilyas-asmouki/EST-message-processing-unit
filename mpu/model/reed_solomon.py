# mpu/model/reed_solomon.py
# Minimal Reed–Solomon (255,223) encoder for GF(2^8)
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
        if x & (1 << m):  # if degree 8 term appears, reduce by primitive polynomial
            # reduce mod p(x): subtract (XOR) prim_poly
            x ^= prim_poly
    # duplicate
    for i in range(2**m - 1, 2 * (2**m - 1)):
        exp[i] = exp[i - (2**m - 1)]

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
        root = exp[(log[alpha] * (B + i)) % (2**m - 1)]  # alpha^(B+i)
        # (x - root) == (1, root) in coefficients [root, 1]
        g = gf_poly_mul(g, [root, 1])

    return g

_GEN = rs_generator_poly(two_t, first_consec_root)

# systematic encoder: compute parity as remainder of x^two_t * m(x) mod g(x)
# uses Linear Feedback Shift Register
def rs_encode_block_223(data: bytes, nsym: int = two_t) -> bytes:
    if len(data) != k:
        raise ValueError(f"RS({n},{k}) block must be {k} bytes, got {len(data)}")
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
                parity[i] ^= gf_mul(_GEN[nsym - 1 - i], feedback)
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
        x = exp[(log[alpha] * (B + j)) % (2**m - 1)]  # alpha^{B+j}
        acc = 0
        # codeword[0] = x^(n-1), codeword[1] = x^(n-2), etc.
        for i, c in enumerate(codeword):
            if c != 0:
                # c * x^(n-1-i)
                power = ((n - 1 - i) * log[x]) % (2**m - 1)
                acc ^= gf_mul(c, exp[power])
        S.append(acc)
    return S



if __name__ == "__main__":
    import random
    import reedsolo  # pip install reedsolo
    random.seed(69)

    RS = reedsolo.RSCodec(
        two_t,                 # 32 parity bytes
        nsize=n,               # codeword length
        c_exp=m,               # GF(2^8)
        generator=alpha,       # primitive element alpha = 0x02
        fcr=first_consec_root, # first consecutive root = 1
        prim=prim_poly,        # primitive polynomial 0x187
    )
    print("[info] reedsolo oracle configured (prim=0x%X, gen=%d, fcr=%d)" % (prim_poly, alpha, first_consec_root))

    total = 0
    bad = 0

    def check_equal(name: str, want: bytes, got: bytes):
        global total, bad
        total += 1
        if want != got:
            bad += 1
            first_bad = next((i for i,(a,b) in enumerate(zip(want,got)) if a!=b), None)
            print(f"[FAIL] {name}: mismatch at byte {first_bad if first_bad is not None else 'n/a'}")
        else:
            print(f"[PASS] {name}")

    # Edge patterns
    edge_msgs = [
        bytes([0]*k),
        bytes([0xFF]*k),
        bytes([i % 256 for i in range(k)]),
        bytes([(255 - i) % 256 for i in range(k)]),
        bytes([0xAA]*k),
        bytes([0x55]*k),
        bytes([0]*(k-1) + [1]),                   # one-hot at end
        bytes([1] + [0]*(k-1)),                   # one-hot at start
        bytes([0]*100 + [0xFF]*23 + [0]*(k-123)), # blocky structure
        bytes([(i*37) & 0xFF for i in range(k)]), # LCG-ish
    ]

    # random pool (increase count if you want more hammering)
    random_msgs = [bytes(random.randrange(256) for _ in range(k)) for __ in range(500)]

    # 1. edge vectors
    for idx, mblk in enumerate(edge_msgs, 1):
        got = rs_encode_block_223(mblk)
        ref = RS.encode(mblk)        # oracle
        print(f"msg = {mblk.hex()}")
        print(f"ref = {ref.hex()}")
        print(f"got = {got.hex()}")
        check_equal(f"EDGE#{idx}", ref, got)

    # 2. random vectors
    for rix, mblk in enumerate(random_msgs, 1):
        got = rs_encode_block_223(mblk)
        ref = RS.encode(mblk)        # oracle
        print(f"msg = {mblk.hex()}")
        print(f"ref = {ref.hex()}")
        print(f"got = {got.hex()}")
        check_equal(f"RAND#{rix}", ref, got)

    # summary
    print("\n=== SUMMARY ===")
    print(f"Total oracle equality checks: {total}, mismatches: {bad}")
    if bad == 0:
        print("All codewords match the reedsolo oracle")
    else:
        print("Mismatches found")
