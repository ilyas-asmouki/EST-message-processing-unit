# mpu/model/diff_encoder.py
# bitwise differential encoder/decoder used after the CC stage
# - msb-first per byte
# - reset state at the start of EACH 512-byte convolutional block
#   (per 255-byte RS/interleaver codeword after CC+pad)
#
# encode (NRZI-style):
#   y[i] = x[i] ^ y[i-1], with y[-1] = 0     (reset per block)
# Decode:
#   x[i] = y[i] ^ y[i-1], with y[-1] = 0     (reset per block)
#
# note: encode is a prefix-XOR of x: y[i] = x[0] ^ x[1] ^ ... ^ x[i]

from typing import Iterable, List

ENC_BYTES_PER_BLOCK = 512  # 4096 bits per CC-encoded block in this pipeline


def _bits_from_bytes(data: bytes) -> List[int]:
    out: List[int] = []
    for b in data:
        for i in range(8):
            out.append((b >> (7 - i)) & 1)
    return out


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
    if n:
        out.append(acc << (8 - n))
    return bytes(out)


def _encode_block_bits(bits: List[int]) -> List[int]:
    # y[i] = x[i] ^ y[i-1], y[-1] = 0
    prev_y = 0
    out: List[int] = []
    for x in bits:
        y = x ^ prev_y
        out.append(y)
        prev_y = y
    return out


def _decode_block_bits(bits: List[int]) -> List[int]:
    # x[i] = y[i] ^ y[i-1], y[-1] = 0
    prev_y = 0
    out: List[int] = []
    for y in bits:
        x = y ^ prev_y
        out.append(x)
        prev_y = y
    return out


def diff_encode(data: bytes, block_bytes: int = ENC_BYTES_PER_BLOCK) -> bytes:
    # differentially encode a stream composed of 512B CC blocks
    if len(data) % block_bytes != 0:
        raise ValueError(f"differential encoder expects multiples of {block_bytes} bytes, got {len(data)}")
    out = bytearray()
    for off in range(0, len(data), block_bytes):
        blk = data[off:off + block_bytes]
        bits = _bits_from_bytes(blk)          # 4096 bits
        enc  = _encode_block_bits(bits)       # reset per block
        out.extend(_bytes_from_bits(enc))     # 512 bytes
    return bytes(out)


def diff_decode(data: bytes, block_bytes: int = ENC_BYTES_PER_BLOCK) -> bytes:
    # Differentially decode a stream composed of 512B CC blocks
    if len(data) % block_bytes != 0:
        raise ValueError(f"differential decoder expects multiples of {block_bytes} bytes, got {len(data)}.")
    out = bytearray()
    for off in range(0, len(data), block_bytes):
        blk = data[off:off + block_bytes]
        bits = _bits_from_bytes(blk)          # 4096 bits
        dec  = _decode_block_bits(bits)       # reset per block
        out.extend(_bytes_from_bits(dec))     # 512 bytes
    return bytes(out)


# oracle implementations

def _oracle_prefix_xor_encode(bits: List[int]) -> List[int]:
    # reference/oracle: y[i] = XOR_{k=0..i} x[k] (prefix XOR)
    acc = 0
    out: List[int] = []
    for x in bits:
        acc ^= x
        out.append(acc)
    return out


def _oracle_decode_from_encoded(bits_enc: List[int]) -> List[int]:
    # reference/oracle decode: x[i] = y[i] XOR y[i-1], y[-1] = 0
    prev_y = 0
    out: List[int] = []
    for y in bits_enc:
        out.append(y ^ prev_y)
        prev_y = y
    return out


if __name__ == "__main__":
    import random
    random.seed(1234)

    def check_roundtrip(num_blocks: int) -> None:
        src = bytes(random.getrandbits(8) for _ in range(num_blocks * ENC_BYTES_PER_BLOCK))

        # our encode/decode
        enc = diff_encode(src)
        dec = diff_decode(enc)

        assert dec == src, "our decode did not restore original"

        # bitwise oracle comparisons per block
        for off in range(0, len(src), ENC_BYTES_PER_BLOCK):
            blk = src[off:off + ENC_BYTES_PER_BLOCK]
            bits = _bits_from_bytes(blk)

            # oracle encode (prefix XOR)
            y_oracle = _oracle_prefix_xor_encode(bits)

            # our encode on same block
            y_ours = _encode_block_bits(bits)
            assert y_ours == y_oracle, "encode mismatch vs oracle (prefix XOR)"

            # oracle decode should restore x
            x_oracle = _oracle_decode_from_encoded(y_oracle)
            assert x_oracle == bits, "oracle decode failed to restore bits"

            # cross-check: decoding our encoded bits with oracle
            x_from_ours = _oracle_decode_from_encoded(y_ours)
            assert x_from_ours == bits, "oracle decode failed on our encoded bits"

    # edge patterns in one block
    def bits_of_byte(b: int) -> List[int]:
        return [(b >> (7 - i)) & 1 for i in range(8)]

    # 1 block edge tests
    zero_block = bytes([0x00] * ENC_BYTES_PER_BLOCK)
    ff_block   = bytes([0xFF] * ENC_BYTES_PER_BLOCK)
    imp_block  = bytes([0x80] + [0x00]*(ENC_BYTES_PER_BLOCK - 1))

    for blk in (zero_block, ff_block, imp_block):
        enc = diff_encode(blk)
        dec = diff_decode(enc)
        assert dec == blk, "edge-case roundtrip failed"

        # spot-check first few bytes against oracle
        bbits = _bits_from_bytes(blk[:4])
        y_or  = _oracle_prefix_xor_encode(bbits)
        y_our = _encode_block_bits(bbits)
        assert y_or == y_our, "edge-case oracle mismatch"

    # random multi-block hammering
    for blks in (1, 2, 3, 5, 8, 16):
        check_roundtrip(blks)


    for _ in range(200):
        blks = random.randint(1, 6)
        check_roundtrip(blks)

    print("[diff_encoder] all oracle and roundtrip tests passed")
