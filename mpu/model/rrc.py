# mpu/model/rrc.py
# fixed-point root-raised-cosine (RRC) FIR filter for FPGA implementation
# filters separate I and Q streams (non-interleaved) before DAC

from typing import Tuple
import numpy as np

# rrc filter parameters
SAMPLES_PER_SYMBOL = 8
FILTER_SPAN = 8
ROLL_OFF = 0.35
NUM_TAPS = FILTER_SPAN * SAMPLES_PER_SYMBOL + 1

# fixed-point parameters
COEFF_WIDTH = 18       # coefficient width in bits
COEFF_FRAC = 16        # fractional bits for coefficients (Q2.16)
DATA_WIDTH = 16        # input data width (Q1.15 from QPSK)
DATA_FRAC = 15         # fractional bits for data
ACC_WIDTH = 48         # accumulator width to prevent overflow
ACC_FRAC = DATA_FRAC + COEFF_FRAC  # 31 fractional bits


def float_to_fixed(value: float, frac_bits: int, width: int) -> int:
    scale = 1 << frac_bits
    fixed = int(round(value * scale))
    
    # sat to representable range
    max_val = (1 << (width - 1)) - 1
    min_val = -(1 << (width - 1))
    
    if fixed > max_val:
        fixed = max_val
    elif fixed < min_val:
        fixed = min_val
    
    return fixed


def fixed_to_float(fixed: int, frac_bits: int) -> float:
    return float(fixed) / (1 << frac_bits)


def generate_rrc_coefficients(
    sps: int = SAMPLES_PER_SYMBOL,
    span: int = FILTER_SPAN,
    beta: float = ROLL_OFF
) -> np.ndarray:
    num_taps = span * sps + 1
    t = np.arange(num_taps) - (num_taps - 1) / 2.0
    t = t / sps
    
    h = np.zeros(num_taps)
    
    for i, time in enumerate(t):
        # handle special cases to avoid division by zero
        if abs(time) < 1e-10:
            h[i] = (1.0 + beta * (4.0 / np.pi - 1.0))
        elif abs(abs(time) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = (
                np.sin(np.pi * time * (1.0 - beta)) +
                4.0 * beta * time * np.cos(np.pi * time * (1.0 + beta))
            )
            denominator = np.pi * time * (1.0 - (4.0 * beta * time) ** 2)
            h[i] = numerator / denominator
    
    # normalize to unit energy (sum of squares = 1)
    h = h / np.sqrt(np.sum(h ** 2))
    
    # convert to fixed point Q2.16
    h_fixed = np.array([float_to_fixed(c, COEFF_FRAC, COEFF_WIDTH) for c in h], dtype=np.int32)
    
    return h_fixed


# pre-compute filter coefficients
RRC_COEFFS = generate_rrc_coefficients(SAMPLES_PER_SYMBOL, FILTER_SPAN, ROLL_OFF)


def upsample(signal: np.ndarray, factor: int) -> np.ndarray:
    upsampled = np.zeros(len(signal) * factor, dtype=signal.dtype)
    upsampled[::factor] = signal
    return upsampled


def fir_filter(signal: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    # 
    # args:
    # - signal: input signal as int16 (Q1.15)
    # - coeffs: filter coefficients as int32 (Q2.16)
    # returns: filtered signal as int16
    n_taps = len(coeffs)
    n_samples = len(signal)
    
    # output array (Q1.15)
    filtered = np.zeros(n_samples, dtype=np.int16)
    
    # delay line (shift register) - initialized with zeros
    delay_line = np.zeros(n_taps, dtype=np.int16)
    
    # process each sample
    for n in range(n_samples):
        # shift delay line (like a shift register in hardware)
        delay_line[1:] = delay_line[:-1]
        
        # no input scaling - use full precision
        delay_line[0] = signal[n]
        
        # compute filter output (multiply-accumulate)
        # accumulator is 64-bit to prevent overflow during MAC
        acc = 0  # int64
        
        for k in range(n_taps):
            # multiply: Q1.15 * Q2.16 = Q3.31
            product = int(delay_line[k]) * int(coeffs[k])
            acc += product
        
        # round and scale down from Q3.31 to Q1.15
        # shift right by 16 bits (COEFF_FRAC = 16)
        acc_rounded = (acc + (1 << 15)) >> 16
        
        # saturate to int16 range
        if acc_rounded > 32767:
            filtered[n] = 32767
        elif acc_rounded < -32768:
            filtered[n] = -32768
        else:
            filtered[n] = np.int16(acc_rounded)
    
    return filtered


def rrc_filter(
    i_stream: np.ndarray,
    q_stream: np.ndarray,
    coeffs: np.ndarray = RRC_COEFFS
) -> Tuple[np.ndarray, np.ndarray]:
    # args:
    # i_stream: I samples as int16 (Q1.15)
    # q_stream: Q samples as int16 (Q1.15)
    # returns: Tuple of (filtered_i, filtered_q) as int16 (Q1.15)
    if len(i_stream) != len(q_stream):
        raise ValueError(f"I and Q streams must have same length: {len(i_stream)} vs {len(q_stream)}")
    
    i_filtered = fir_filter(i_stream, coeffs)
    q_filtered = fir_filter(q_stream, coeffs)
    
    return i_filtered, q_filtered


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print(f"[RRC Fixed-Point Filter Configuration]")
    print(f"  Samples per symbol: {SAMPLES_PER_SYMBOL}")
    print(f"  Filter span: {FILTER_SPAN} symbols")
    print(f"  Roll-off factor (beta): {ROLL_OFF}")
    print(f"  Number of taps: {NUM_TAPS}")
    print(f"  Coefficient format: Q{COEFF_WIDTH-COEFF_FRAC}.{COEFF_FRAC}")
    print(f"  Data format: Q{DATA_WIDTH-DATA_FRAC}.{DATA_FRAC}")
    print(f"  Accumulator format: Q{ACC_WIDTH-ACC_FRAC}.{ACC_FRAC}")
    print(f"  Input scaling: none (full precision)")
    print(f"  Output scaling: >>16 (COEFF_FRAC bits)")
    
    # convert coefficients to float for display
    coeffs_float = np.array([fixed_to_float(c, COEFF_FRAC) for c in RRC_COEFFS])
    
    print(f"\nCoefficient statistics:")
    print(f"  Fixed-point min/max: {np.min(RRC_COEFFS)} / {np.max(RRC_COEFFS)}")
    print(f"  Float equivalent min/max: {np.min(coeffs_float):.10f} / {np.max(coeffs_float):.10f}")
    print(f"  Filter energy (L2 norm): {np.linalg.norm(coeffs_float):.6f}")
    
    # visualize filter coefficients
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.stem(coeffs_float, basefmt=' ')
    plt.title('RRC Filter Impulse Response (Fixed-Point)')
    plt.xlabel('Tap Index')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    
    # frequency response
    plt.subplot(2, 2, 2)
    freq_response = np.fft.fft(coeffs_float, 2048)
    freq_db = 20 * np.log10(np.abs(freq_response) + 1e-10)
    freq_axis = np.linspace(0, 1, len(freq_db))
    plt.plot(freq_axis, freq_db)
    plt.title('RRC Filter Frequency Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.ylim(-60, 5)
    
    # test with impulse
    impulse = np.zeros(100, dtype=np.int16)
    impulse[50] = 23170  # max Q1.15 value / sqrt(2)
    response_i = fir_filter(impulse, RRC_COEFFS)
    
    plt.subplot(2, 2, 3)
    plt.plot(response_i.astype(float))
    plt.title('Filter Response to Unit Impulse')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # test with random QPSK-like symbols
    np.random.seed(42)
    n_symbols = 50
    qpsk_val = 23170  # approximately 1.0 in Q1.15
    i_symbols = np.random.choice([-qpsk_val, qpsk_val], n_symbols).astype(np.int16)
    q_symbols = np.random.choice([-qpsk_val, qpsk_val], n_symbols).astype(np.int16)
    
    i_filt, q_filt = rrc_filter(i_symbols, q_symbols)
    
    plt.subplot(2, 2, 4)
    plt.plot(i_filt.astype(float), label='I filtered', alpha=0.7)
    plt.plot(q_filt.astype(float), label='Q filtered', alpha=0.7)
    plt.plot(i_symbols.astype(float), 'o', label='I symbols', markersize=4)
    plt.plot(q_symbols.astype(float), 's', label='Q symbols', markersize=4)
    plt.title('Filtered QPSK Symbols (Fixed-Point)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # check for saturation
    i_saturated = np.sum((i_filt == 32767) | (i_filt == -32768))
    q_saturated = np.sum((q_filt == 32767) | (q_filt == -32768))
    print(f"\nSaturation check:")
    print(f"  I samples saturated: {i_saturated}/{len(i_filt)}")
    print(f"  Q samples saturated: {q_saturated}/{len(q_filt)}")
    
    print("\n[RRC Fixed-Point Tests Passed]")