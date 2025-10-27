# mpu/model/helpers.py
# common helper functions used across multiple modules in the pipeline
# provides bit/byte conversion utilities

from typing import Iterable, List


def bits_from_bytes(data: bytes) -> List[int]:
    out: List[int] = []
    for b in data:
        for i in range(8):
            out.append((b >> (7 - i)) & 1)
    return out


def bytes_from_bits(bits: Iterable[int]) -> bytes:
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
        # pad last byte with zeros on the right
        out.append(acc << (8 - n))
    return bytes(out)


def pad_to_block(data: bytes, block_size: int) -> bytes:
    rem = len(data) % block_size
    if rem == 0:
        return data
    return data + bytes(block_size - rem)


def chunk_bytes(data: bytes, chunk_size: int) -> List[bytes]:
    if len(data) % chunk_size != 0:
        raise ValueError(f"Data length {len(data)} is not a multiple of chunk size {chunk_size}")
    
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    return chunks


def group_bits(bits: List[int], group_size: int) -> List[tuple]:
    # pad if necessary
    if len(bits) % group_size != 0:
        padding = group_size - (len(bits) % group_size)
        bits = bits + [0] * padding
    
    groups = []
    for i in range(0, len(bits), group_size):
        groups.append(tuple(bits[i:i + group_size]))
    return groups


def hex_dump(data: bytes, bytes_per_line: int = 16) -> str:
    lines = []
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i + bytes_per_line]
        hex_str = ' '.join(f'{b:02X}' for b in chunk)
        lines.append(f'{i:08X}: {hex_str}')
    return '\n'.join(lines)


def hamming_distance_bytes(a: bytes, b: bytes) -> int:
    if len(a) != len(b):
        raise ValueError(f"Sequences must have equal length: {len(a)} != {len(b)}")
    
    distance = 0
    for byte_a, byte_b in zip(a, b):
        xor = byte_a ^ byte_b
        # count set bits in XOR
        distance += bin(xor).count('1')
    return distance


def bit_error_rate(original: bytes, received: bytes) -> float:
    total_bits = len(original) * 8
    if total_bits == 0:
        return 0.0
    
    errors = hamming_distance_bytes(original, received)
    return errors / total_bits

