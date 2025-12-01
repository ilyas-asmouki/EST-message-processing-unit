# mpu/model/helpers.py
# common helper functions used across multiple modules in the pipeline
# provides bit/byte conversion utilities

from typing import Iterable, List, Optional, Tuple

import numpy as np


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


def add_awgn(
    i_stream: np.ndarray,
    q_stream: np.ndarray,
    snr_db: Optional[float] = None,
    noise_std: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Inject AWGN into the I/Q streams.

    Args:
        i_stream: Transmit I samples (int16 expected).
        q_stream: Transmit Q samples (int16 expected).
        snr_db: Target complex SNR in dB (mutually exclusive with noise_std).
        noise_std: Standard deviation per real dimension (same units as samples).
        seed: Optional RNG seed to make the noise repeatable.

    Returns:
        (i_noisy, q_noisy, metrics)
    """
    if len(i_stream) != len(q_stream):
        raise ValueError("I and Q streams must be the same length for AWGN injection")

    if snr_db is None and noise_std is None:
        return i_stream.copy(), q_stream.copy(), {
            "enabled": False,
            "noise_std": 0.0,
            "measured_snr_db": float("inf"),
        }

    if snr_db is not None and noise_std is not None:
        raise ValueError("Specify either snr_db or noise_std, not both.")

    rng = np.random.default_rng(seed)
    i_float = i_stream.astype(np.float64)
    q_float = q_stream.astype(np.float64)

    signal_power = float(np.mean(i_float**2 + q_float**2))
    if signal_power <= 0:
        signal_power = 1e-12  # avoid division by zero for all-zero signals

    if snr_db is not None:
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        component_var = noise_power / 2.0
    else:
        component_var = float(noise_std) ** 2
        noise_power = 2.0 * component_var

    component_std = float(np.sqrt(component_var))
    noise_i = rng.normal(0.0, component_std, size=i_float.shape)
    noise_q = rng.normal(0.0, component_std, size=q_float.shape)

    i_noisy = np.clip(np.rint(i_float + noise_i), -32768, 32767).astype(np.int16)
    q_noisy = np.clip(np.rint(q_float + noise_q), -32768, 32767).astype(np.int16)

    measured_noise_power = float(np.mean(noise_i**2 + noise_q**2))
    if measured_noise_power <= 0:
        measured_snr = float("inf")
    else:
        measured_snr = float(10 * np.log10(signal_power / measured_noise_power))

    metrics = {
        "enabled": True,
        "signal_power": signal_power,
        "target_snr_db": snr_db,
        "noise_std": component_std,
        "measured_noise_power": measured_noise_power,
        "measured_snr_db": measured_snr,
        "seed": seed,
    }

    return i_noisy, q_noisy, metrics
