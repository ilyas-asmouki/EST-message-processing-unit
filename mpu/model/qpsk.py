# mpu/model/qpsk.py
# fixed-point QPSK modulator/demod, matching commpy.QAMModem(4) labeling:
#   dibit (b1 b0), MSB-first per symbol and byte:
#       00 -> (-A, -A)
#       01 -> (-A, +A)
#       11 -> (+A, +A)
#       10 -> (+A, -A)
# where A = round((2^FRAC_BITS - 1)/sqrt(2)). With FRAC_BITS=15, A ~= 23170
#
# I/Q outputs are int16 (Q1.15), either interleaved [I0,Q0,I1,Q1,...] or shape (N,2)
# hard-decision demod maps sign(I)->b1, sign(Q)->b0 using negative->0, non-negative->1

from typing import Dict, Tuple
import numpy as np
from mpu.model.helpers import bits_from_bytes, bytes_from_bits, group_bits

FRAC_BITS_DEFAULT = 15  # Q1.15
INT16_MIN = -32768
INT16_MAX =  32767


# mapping
def _saturate_int16(x: int) -> int:
    if x > INT16_MAX: return INT16_MAX
    if x < INT16_MIN: return INT16_MIN
    return x

def _qpsk_lut(frac_bits: int) -> Dict[Tuple[int, int], Tuple[int, int]]:
    # scale so +1.0 -> 2^frac_bits - 1 (e.g., 32767 for Q1.15)
    scale = (1 << frac_bits) - 1
    a = int(round(scale / np.sqrt(2.0)))  # component magnitude
    # commpy convention: 0 -> -a, 1 -> +a
    # gray map (b1,b0): 00,01,11,10 -> (-,-),(-,+),(+,+),(+,-)
    return {
        (0, 0): (-a, -a),
        (0, 1): (-a, +a),
        (1, 1): (+a, +a),
        (1, 0): (+a, -a),
    }


def qpsk_modulate_bytes_fixed(
    data: bytes,
    *,
    frac_bits: int = FRAC_BITS_DEFAULT,
    interleaved: bool = True
) -> np.ndarray:
    # map bytes -> fixed-point qpsk I/Q (int16)
    # returns:
    # if interleaved=True, shape (2*Nsym,) as [I0, Q0, I1, Q1, ...]
    # else: shape (Nsym, 2) with columns [I, Q]

    bits = bits_from_bytes(data)
    dibits = group_bits(bits, 2)
    lut = _qpsk_lut(frac_bits)

    I = np.empty(len(dibits), dtype=np.int16)
    Q = np.empty(len(dibits), dtype=np.int16)
    for i, db in enumerate(dibits):
        ii, qq = lut[db]
        I[i] = np.int16(_saturate_int16(ii))
        Q[i] = np.int16(_saturate_int16(qq))

    if interleaved:
        out = np.empty(2 * len(dibits), dtype=np.int16)
        out[0::2] = I
        out[1::2] = Q
        return out
    else:
        return np.stack([I, Q], axis=1)

def qpsk_demod_hard_fixed(
    iq: np.ndarray,
    *,
    interleaved: bool = True
) -> bytes:
    # hard-decision demod from fixed-point I/Q back to packed bytes (msb-first)
    # uses the same convention: neg -> 0, pos -> 1
    if interleaved:
        if iq.ndim != 1 or (iq.size % 2) != 0:
            raise ValueError("Interleaved input must be 1D with even length")
        I = iq[0::2]
        Q = iq[1::2]
    else:
        if iq.ndim != 2 or iq.shape[1] != 2:
            raise ValueError("Non-interleaved input must be shape (N,2)")
        I = iq[:, 0]
        Q = iq[:, 1]

    bits = []
    for ii, qq in zip(I, Q):
        b1 = 0 if ii < 0 else 1  # I sign
        b0 = 0 if qq < 0 else 1  # Q sign
        bits.extend((b1, b0))
    return bytes_from_bits(bits)

def pack_iq_le_bytes(iq_interleaved: np.ndarray) -> bytes:
    # pack interleaved int16 I/Q to little-endian bytes for file/FIFO/DAC
    # input must be 1D int16 [I0, Q0, I1, Q1, ...]
    if iq_interleaved.dtype != np.int16 or iq_interleaved.ndim != 1:
        raise ValueError("iq_interleaved must be 1D np.int16.")
    return iq_interleaved.astype('<i2', copy=False).tobytes()



if __name__ == "__main__":
    import random
    random.seed(0xD00D)

    from commpy.modulation import QAMModem  # pip install commpy

    def to_float_complex(iq_interleaved: np.ndarray, frac_bits: int) -> np.ndarray:
        scale = float((1 << frac_bits) - 1)
        I = iq_interleaved[0::2].astype(np.float64) / scale
        Q = iq_interleaved[1::2].astype(np.float64) / scale
        return (I + 1j * Q).astype(np.complex128)

    # edge vectors
    edge_msgs = [
        bytes([0x00]*32),
        bytes([0xFF]*32),
        bytes([0xAA]*32),  # 1010...
        bytes([0x55]*32),  # 0101...
        bytes([0x80] + [0x00]*31),
        bytes(list(range(32))),
    ]
    # random blobs with awkward sizes
    rand_msgs = [bytes(random.getrandbits(8) for _ in range(L)) for L in (1, 2, 3, 7, 31, 33, 64, 255, 512)]

    # 1) roundtrip hard-decision on fixed-point
    for idx, msg in enumerate(edge_msgs + rand_msgs, 1):
        iq = qpsk_modulate_bytes_fixed(msg, frac_bits=FRAC_BITS_DEFAULT, interleaved=True)
        rec = qpsk_demod_hard_fixed(iq, interleaved=True)
        if rec != msg:
            a = bits_from_bytes(msg)
            b = bits_from_bytes(rec[:len(msg)])
            first = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None)
            raise AssertionError(f"[FAIL] fixed hard roundtrip mismatch at bit {first}")
    print("[info] fixed hard-decision roundtrips passed")

    # 2) avg symbol energy sanity: should be ~1.0
    msg = bytes(random.getrandbits(8) for _ in range(4096))
    iq = qpsk_modulate_bytes_fixed(msg, frac_bits=FRAC_BITS_DEFAULT, interleaved=True)
    sym = to_float_complex(iq, FRAC_BITS_DEFAULT)
    Es = np.mean((sym.real**2 + sym.imag**2))
    if abs(Es - 1.0) > 1e-3:
        raise AssertionError(f"[FAIL] avg energy {Es} not ~1.0")
    print("[info] average symbol energy ~ 1.0")

    # 3) oracle comparison vs commpy (unit-average-energy constellation, gray)
    modem = QAMModem(4)  # 4-QAM ≡ 45 deg-rotated QPSK, gray-labeled like our mapping
    for idx, msg in enumerate(edge_msgs + rand_msgs, 1):
        bits = np.array(bits_from_bytes(msg), dtype=int)
        if bits.size % 2 != 0:
            bits = np.concatenate([bits, np.zeros(1, dtype=int)])
        ref_syms = modem.modulate(bits)             # complex128, Es=1
        # normalize oracle to unit-average energy if needed (some commpy builds dont)
        Es_ref = np.mean((ref_syms.real**2 + ref_syms.imag**2))
        if not np.isclose(Es_ref, 1.0, atol=1e-9):
            ref_syms = ref_syms / np.sqrt(Es_ref)

        our_syms = to_float_complex(
            qpsk_modulate_bytes_fixed(msg, frac_bits=FRAC_BITS_DEFAULT, interleaved=True),
            FRAC_BITS_DEFAULT
        )
        if ref_syms.shape != our_syms.shape:
            raise AssertionError(f"[FAIL] oracle shape mismatch {ref_syms.shape} vs {our_syms.shape}")
        if not (np.allclose(ref_syms.real, our_syms.real, atol=1e-6) and
                np.allclose(ref_syms.imag, our_syms.imag, atol=1e-6)):
            for i, (x, y) in enumerate(zip(our_syms, ref_syms)):
                if not (np.isclose(x.real, y.real, atol=1e-6) and np.isclose(x.imag, y.imag, atol=1e-6)):
                    raise AssertionError(f"[FAIL] oracle#{idx} first diff at {i}: ours={x}, ref={y}")
        print(f"[PASS] oracle#{idx}")

    print("[qpsk] all tests passed")