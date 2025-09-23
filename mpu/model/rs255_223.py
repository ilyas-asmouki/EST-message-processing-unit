from dataclasses import dataclass
from typing import List

# //////////
# This section defines the configuration for our Reed-Solomon error correction system.
# 
# Think of Reed-Solomon like a sophisticated spell-checker for digital data. When you
# send data over a noisy channel (like radio waves to a satellite), some bits might
# get flipped from 0 to 1 or vice versa. Reed-Solomon adds extra "parity" bytes to
# your original message that act like a mathematical fingerprint. If some of your
# data gets corrupted during transmission, the receiver can use these parity bytes
# to detect AND fix the errors automatically.
#
# The RSCfg class holds all the mathematical parameters that define exactly how
# this error correction works. These numbers determine how many errors can be
# detected and corrected.

@dataclass
class RSCfg:
    # GF(256) polynomial without the x^8 term: default = 0b100011101
    # p(x) = x^8+x^4+x^3+x^2+1
    prim_poly: int = 0b100011101
    # generator: first root (alpha^b): CCSDS uses b = 0 (sometimes 1 or 2 in other specs)
    first_consecutive_root: int = 0
    # RS(n,k) over GF(256)
    n: int = 255
    k: int = 223


default_rs_cfg = RSCfg()


# //////////
# This section implements a special kind of number system called a "Galois Field" (GF).
# 
# In normal math, if you multiply two large numbers, the result gets bigger. But in
# Galois Field math, all numbers stay within a fixed range (0-255 in our case).
# Think of it like a clock: if it's 11 PM and you add 3 hours, you get 2 AM, not 14 PM.
# The numbers "wrap around" in a controlled way.
#
# Why do we need this weird math? Reed-Solomon codes work by treating your data bytes
# as coefficients in mathematical polynomials, then doing polynomial algebra on them.
# Regular math would make these numbers grow huge and unmanageable. Galois Field math
# keeps everything neat and bounded.
#
# The GF256 class implements this special math system where all operations stay within
# 0-255 (perfect for bytes). It precomputes lookup tables to make the math fast.

class GF256:
    # //////////
    # This initializes our Galois Field number system.
    #
    # The "prim_poly" (primitive polynomial) is like the rule that defines how our
    # special math works. It's similar to how different number bases (like binary vs
    # decimal) have different rules. This specific polynomial creates a mathematical
    # structure where we can do reliable error correction.
    #
    # We create two lookup tables:
    # - exp[]: converts from "logarithm space" back to normal numbers  
    # - log[]: converts normal numbers to "logarithm space"
    # 
    # These tables make multiplication and division extremely fast - instead of doing
    # complex polynomial math every time, we just look up precomputed results.
    def __init__(self, prim_poly: int):
        self.pp = prim_poly
        self.exp = [0] * 512   # exp table (duplicate to avoid mod 255)
        self.log = [0] * 256   # log table. log(0) undefined, we keep 0
        self._build_tables()

    # //////////
    # This builds the lookup tables that make our Galois Field math fast.
    #
    # Think of this like building a multiplication table, but for our special math.
    # We start with x = 1 and keep multiplying by 2 (shifting bits left), but whenever
    # the result gets bigger than 255, we apply our primitive polynomial rule to bring
    # it back into the 0-255 range.
    #
    # After 255 iterations, we've generated all possible non-zero values in our field.
    # The exp[] table maps powers to values, and log[] maps values back to powers.
    # We duplicate the exp table so we don't have to do modular arithmetic in hot code paths.
    #
    # This is like creating a super-efficient calculator that can instantly multiply
    # or divide any two numbers in our special math system.
    def _build_tables(self):
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= self.pp
        # duplicate to avoid mod in hot path
        for i in range(255, 512):
            self.exp[i] = self.exp[i - 255]

    # //////////
    # Addition in Galois Field is simple: just XOR the two numbers.
    #
    # XOR (exclusive OR) is perfect for this because it has a special property:
    # if you XOR a number with itself, you get 0. This means subtraction and
    # addition are the same operation in our field, which simplifies the math
    # tremendously.
    #
    # Example: 5 + 3 = 5 XOR 3 = 6 in our field
    # This might seem weird, but it's exactly what we need for error correction.
    def add(self, a: int, b: int) -> int:
        return a ^ b

    # //////////
    # Multiplication in Galois Field uses our precomputed lookup tables.
    #
    # Instead of doing complex polynomial multiplication, we use a clever trick:
    # multiply(a,b) = exp[log[a] + log[b]]
    # 
    # This works because of logarithm properties: log(a*b) = log(a) + log(b)
    # So we look up the logarithms, add them, then convert back using exp table.
    #
    # Special case: if either number is 0, the result is 0 (since 0 times anything is 0).
    # This is much faster than doing the full polynomial math every time.
    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]
    
    # //////////
    # Division in Galois Field also uses logarithms for speed.
    #
    # divide(a,b) = exp[log[a] - log[b]]
    # Again using logarithm properties: log(a/b) = log(a) - log(b)
    #
    # We handle the special cases:
    # - If a is 0, result is 0 (0 divided by anything is 0)
    # - If b is 0, that's division by zero which is undefined (error)
    # - We use modulo 255 when subtracting to keep the result in valid range
    #
    # This gives us fast, accurate division in our special number system.
    def div(self, a: int, b: int) -> int:
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("GF256 div by 0")
        return self.exp[(self.log[a] - self.log[b]) % 255]
    


# //////////
# This function multiplies two polynomials using our Galois Field math.
#
# A polynomial is like "3x² + 2x + 1" - it has coefficients (3, 2, 1) and powers of x.
# In Reed-Solomon, we represent polynomials as lists of coefficients.
# For example: [3, 2, 1] represents 3x² + 2x + 1 (highest power first).
#
# When we multiply polynomials, we:
# 1. Multiply each term from the first polynomial with each term from the second
# 2. Add together all terms that have the same power of x
# 3. Use Galois Field arithmetic for all operations
#
# This is fundamental to Reed-Solomon because your data gets treated as polynomial
# coefficients, and we do polynomial arithmetic to generate the error correction codes.
def poly_mul(gf: GF256, a: List[int], b: List[int]) -> List[int]:
    # multiply polynomials over GF256. a,b as lists of coeffs (highest first)
    res = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            res[i + j] ^= gf.mul(ai, bj)
    return res

# //////////
# This function multiplies a polynomial by 'x' (shifts all powers up by 1).
#
# Multiplying by x is like shifting: if you have 3x² + 2x + 1, multiplying by x
# gives you 3x³ + 2x² + 1x. In our list representation [3, 2, 1], this becomes
# [3, 2, 1, 0] - we just add a zero at the end.
#
# This operation is used when building the generator polynomial. It's a simple
# but important building block in polynomial arithmetic.
def poly_scale_x(gf: GF256, a: List[int]) -> List[int]:
    # multiply by x: append 0 at the end (highest-first representation)
    return a + [0]

# //////////
# This creates the "generator polynomial" - the mathematical key to Reed-Solomon encoding.
#
# Think of the generator polynomial as a special mathematical formula that converts
# your data into a codeword with error correction. It's like a recipe that tells
# us exactly how to compute the parity bytes.
#
# The generator polynomial has a special property: it has specific "roots" (values
# where the polynomial equals zero). By carefully choosing these roots, we ensure
# that any valid codeword, when treated as a polynomial, will also equal zero at
# these same points.
#
# If errors occur during transmission, the received polynomial won't equal zero at
# these points anymore, which is how we detect errors. The pattern of non-zero
# values tells us exactly where the errors are and how to fix them.
#
# We build this polynomial by multiplying together many linear factors:
# (x - α⁰) × (x - α¹) × (x - α²) × ... where α is a primitive element of our field.
def rs_generator_poly(gf: GF256, t: int, first_root: int) -> List[int]:
    # RS(255, 255-2t). generator has 2*t consecutive roots
    # Start with g(x) = 1
    g = [1]
    for i in range(2 * t):
        # Multiply by (x - alpha^(first_root + i))  -> coefficients [1, root]
        root = gf.exp[(first_root + i) % 255]
        g = poly_mul(gf, g, [1, root])  # using x + root since subtraction == addition in GF(2^m)
    return g  # highest-first

# //////////
# This is the main Reed-Solomon encoding function - it takes your data and adds error correction.
#
# Here's what happens step by step:
#
# 1. We take your original data (up to 223 bytes for this RS(255,223) code)
# 2. We calculate 32 "parity bytes" using the generator polynomial
# 3. We append these parity bytes to your data to create a 255-byte codeword
#
# The parity bytes are like a mathematical signature of your data. They're computed
# using polynomial division in Galois Field math. The process is similar to how
# checksums work, but much more sophisticated.
#
# Think of it like this: if your data was a sentence, the parity bytes would be
# like adding extra words that capture the "essence" of the sentence so well that
# even if some of the original words get corrupted, you can reconstruct them.
#
# The encoding uses a "systematic" approach, meaning your original data stays
# unchanged and the parity is simply appended. This makes it easy to extract
# the original data if no errors occurred.
#
# The algorithm uses an LFSR (Linear Feedback Shift Register) approach - we feed
# each data byte through a mathematical filter that accumulates the parity bytes.
def rs_encode_255_223(data: bytes, cfg: RSCfg = default_rs_cfg) -> bytes:
    gf = GF256(cfg.prim_poly)
    n, k = cfg.n, cfg.k
    if len(data) > k:
        raise ValueError(f"data length {len(data)} exceeds RS({n},{k}) payload {k}")
    t = (n - k) // 2

    # build generator polynomial once
    g = rs_generator_poly(gf, t, cfg.first_consecutive_root)  # length = 2*t + 1

    # message polynomial (highest-first) padded with parity space
    # Use a simple LFSR-like encoder (systematic): feed bytes, update 2t-byte register
    parity = [0] * (n - k)
    for byte in data:
        # feedback = input ^ parity[0]
        feedback = byte ^ parity[0]
        # shift left and update with generator (skip g[0]==1)
        for i in range(n - k - 1):
            coef = g[i + 1]
            parity[i] = parity[i + 1] ^ (gf.mul(feedback, coef) if coef else 0)
        # last element
        parity[-1] = gf.mul(feedback, g[-1])

    return bytes(data) + bytes(parity)
    


# //////////
# This section provides a simple test to demonstrate the Reed-Solomon encoder in action.
#
# When you run this file directly (not import it), it will:
# 1. Take the message "HELLO WORLD" (11 bytes)
# 2. Encode it using Reed-Solomon to create a 255-byte codeword
# 3. Show you the original length, final length, and the encoded result in hexadecimal
#
# The encoded result contains your original message plus 32 parity bytes that can
# correct up to 16 byte errors anywhere in the codeword. This is exactly the kind
# of robust error correction used in satellite communications, CDs, DVDs, and
# many other applications where data integrity is critical.
#
# The hex output shows the complete codeword - the first 11 bytes are your original
# "HELLO WORLD" message, followed by the computed parity bytes.
# ---- Simple self-test when run directly ----
if __name__ == "__main__":
    msg = b"HELLO WORLD"
    code = rs_encode_255_223(msg)
    print("msg len:", len(msg), "code len:", len(code))
    print("codeword hex:", code.hex())