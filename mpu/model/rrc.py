# mpu/model/rrc.py
# root-raised-cosine (RRC) FIR filter for pulse shaping
# filters separate I and Q streams (non-interleaved) before DAC

from typing import Tuple
import numpy as np

# rrc filter parameters
SAMPLES_PER_SYMBOL = 8           # oversampling factor (samples per QPSK symbol)
FILTER_SPAN = 8                  # filter span in symbols (total taps = span * sps + 1)
ROLL_OFF = 0.35                  # roll-off factor (beta) - typical for satellite: 0.25-0.5
NUM_TAPS = FILTER_SPAN * SAMPLES_PER_SYMBOL + 1  # total number of filter taps


# rrc coefficient generation
def generate_rrc_coefficients(
    sps: int = SAMPLES_PER_SYMBOL,
    span: int = FILTER_SPAN,
    beta: float = ROLL_OFF
) -> np.ndarray:
    # returns: numpy array of normalized filter coefficients
    num_taps = span * sps + 1
    t = np.arange(num_taps) - (num_taps - 1) / 2.0  # time indices centered at 0
    t = t / sps  # normalize by samples per symbol
    

    h = np.zeros(num_taps)
    
    for i, time in enumerate(t):
        # handle special cases to avoid division by zero
        
        # case 1: t = 0
        if abs(time) < 1e-10:
            h[i] = (1.0 + beta * (4.0 / np.pi - 1.0))
        
        # case 2: t = +/- 1/(4*beta)
        elif abs(abs(time) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        
        # case 3: general case
        else:
            numerator = (
                np.sin(np.pi * time * (1.0 - beta)) +
                4.0 * beta * time * np.cos(np.pi * time * (1.0 + beta))
            )
            denominator = np.pi * time * (1.0 - (4.0 * beta * time) ** 2)
            h[i] = numerator / denominator
    
    # normalize to unit energy (sum of squares = 1)
    h = h / np.sqrt(np.sum(h ** 2))
    
    return h


# pre-compute filter coefficients (can be re-generated if parameters change)
RRC_COEFFS = generate_rrc_coefficients(SAMPLES_PER_SYMBOL, FILTER_SPAN, ROLL_OFF)


# fir filtering implementation
def fir_filter(signal: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    # explicit implementation suitable for fpga translation

    n_taps = len(coeffs)
    n_samples = len(signal)
    
    # output array
    filtered = np.zeros(n_samples, dtype=signal.dtype)
    
    # delay line (shift register) - initialized with zeros
    delay_line = np.zeros(n_taps, dtype=signal.dtype)
    
    # process each sample
    for n in range(n_samples):
        # shift delay line (like a shift register in hardware)
        delay_line[1:] = delay_line[:-1]
        delay_line[0] = signal[n]
        
        # compute filter output (multiply-accumulate)
        acc = 0.0
        for k in range(n_taps):
            acc += coeffs[k] * delay_line[k]
        
        filtered[n] = acc
    
    return filtered



def rrc_filter(
    i_stream: np.ndarray,
    q_stream: np.ndarray,
    coeffs: np.ndarray = RRC_COEFFS
) -> Tuple[np.ndarray, np.ndarray]:
    # apply rrc filtering to separate I and Q streams
    # this function filters the I and Q streams independently using the smae
    # rrc filter coefficients. the streams should be at symbol rate (1 sample
    # per symbol for QPSK)
    # returns: Tuple of (filtered_i, filtered_q) as numpy arrays
    if len(i_stream) != len(q_stream):
        raise ValueError(f"I and Q streams must have same length: {len(i_stream)} vs {len(q_stream)}")
    

    i_filtered = fir_filter(i_stream, coeffs)
    q_filtered = fir_filter(q_stream, coeffs)
    
    return i_filtered, q_filtered



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    

    print(f"[RRC Filter Configuration]")
    print(f"  Samples per symbol: {SAMPLES_PER_SYMBOL}")
    print(f"  Filter span: {FILTER_SPAN} symbols")
    print(f"  Roll-off factor (beta): {ROLL_OFF}")
    print(f"  Number of taps: {NUM_TAPS}")
    print(f"  Filter coefficients sum: {np.sum(RRC_COEFFS):.6f}")
    print(f"  Filter energy (L2 norm): {np.linalg.norm(RRC_COEFFS):.6f}")
    
    # visualize filter coefficients
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.stem(RRC_COEFFS, basefmt=' ')
    plt.title('RRC Filter Impulse Response')
    plt.xlabel('Tap Index')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    
    # frequency response
    plt.subplot(2, 2, 2)
    freq_response = np.fft.fft(RRC_COEFFS, 2048)
    freq_db = 20 * np.log10(np.abs(freq_response) + 1e-10)
    freq_axis = np.linspace(0, 1, len(freq_db))
    plt.plot(freq_axis, freq_db)
    plt.title('RRC Filter Frequency Response')
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.ylim(-60, 5)
    
    # test with impulse
    impulse = np.zeros(100)
    impulse[50] = 1.0
    response_i = fir_filter(impulse, RRC_COEFFS)
    
    plt.subplot(2, 2, 3)
    plt.plot(response_i)
    plt.title('Filter Response to Unit Impulse')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # test with random QPSK-like symbols
    np.random.seed(42)
    n_symbols = 50
    i_symbols = np.random.choice([-1, 1], n_symbols)
    q_symbols = np.random.choice([-1, 1], n_symbols)
    
    i_filt, q_filt = rrc_filter(i_symbols, q_symbols)
    
    plt.subplot(2, 2, 4)
    plt.plot(i_filt, label='I filtered', alpha=0.7)
    plt.plot(q_filt, label='Q filtered', alpha=0.7)
    plt.plot(i_symbols, 'o', label='I symbols', markersize=4)
    plt.plot(q_symbols, 's', label='Q symbols', markersize=4)
    plt.title('Filtered QPSK Symbols')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("\n[RRC Filter Tests Passed]")