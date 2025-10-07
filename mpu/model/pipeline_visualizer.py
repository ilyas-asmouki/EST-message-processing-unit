#!/usr/bin/env python3
"""
Visualization tool for the communication pipeline
Generates separate focused figures for each processing stage
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple, Dict

from mpu.model.reed_solomon import rs_encode, rs_syndromes, k as RS_K
from mpu.model.interleaver import interleave, deinterleave, CODEWORD_BYTES
from mpu.model.scrambler import scramble_bits, descramble_bits, DEFAULT_SEED
from mpu.model.conv_encoder import conv_encode
from mpu.model.diff_encoder import diff_encode, diff_decode
from mpu.model.qpsk import qpsk_modulate_bytes_fixed, qpsk_demod_hard_fixed, FRAC_BITS_DEFAULT
from mpu.model.rrc import (rrc_filter, upsample, fir_filter, SAMPLES_PER_SYMBOL, 
                           FILTER_SPAN, ROLL_OFF, NUM_TAPS, RRC_COEFFS,
                           fixed_to_float, COEFF_FRAC)
from mpu.model.helpers import bits_from_bytes, pad_to_block



def plot_byte_histogram(ax, data: bytes, title: str, color='steelblue'):
    """Plot histogram of byte values"""
    values = np.frombuffer(data, dtype=np.uint8)
    ax.hist(values, bins=256, range=(0, 256), alpha=0.7, edgecolor='black', color=color)
    ax.set_xlabel('Byte Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_val = np.mean(values)
    std_val = np.std(values)
    ax.text(0.98, 0.98, f'μ={mean_val:.1f}\nσ={std_val:.1f}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_bit_transitions(ax, bits: np.ndarray, title: str, max_bits: int = 2000):
    """Plot bit transitions over time"""
    plot_bits = bits[:min(len(bits), max_bits)]
    ax.step(range(len(plot_bits)), plot_bits, where='post', linewidth=0.8)
    ax.set_xlabel('Bit Index')
    ax.set_ylabel('Bit Value')
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    
    # Calculate transitions
    transitions = np.sum(np.abs(np.diff(plot_bits)))
    ax.text(0.98, 0.98, f'Transitions: {transitions}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def plot_autocorrelation(ax, data: bytes, title: str, max_lag: int = 100):
    """Plot autocorrelation of byte sequence"""
    values = np.frombuffer(data, dtype=np.uint8).astype(float)
    if len(values) < 2:
        return
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Handle constant sequences (zero std deviation)
    if std_val < 1e-10:
        ax.text(0.5, 0.5, 'Constant sequence\n(zero variance)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.set_xlabel('Lag (bytes)')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        return
    
    values_norm = (values - mean_val) / std_val
    
    lags = range(max_lag)
    acf = []
    for lag in lags:
        if lag == 0:
            acf.append(1.0)
        elif lag >= len(values):
            acf.append(0.0)
        else:
            acf.append(np.corrcoef(values_norm[:-lag], values_norm[lag:])[0, 1])
    
    ax.stem(lags, acf, basefmt=' ')
    ax.set_xlabel('Lag (bytes)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)

def plot_qpsk_constellation(ax, iq: np.ndarray, title: str, 
                           frac_bits: int = FRAC_BITS_DEFAULT,
                           max_points: int = 5000):
    """Plot QPSK constellation diagram"""
    scale = float((1 << frac_bits) - 1)
    I = iq[0::2].astype(float) / scale
    Q = iq[1::2].astype(float) / scale
    
    # Plot subset for visibility
    n_points = min(len(I), max_points)
    if n_points < len(I):
        indices = np.random.choice(len(I), n_points, replace=False)
    else:
        indices = np.arange(len(I))
    
    ax.scatter(I[indices], Q[indices], alpha=0.2, s=2, color='steelblue')
    ax.set_xlabel('In-Phase (I)')
    ax.set_ylabel('Quadrature (Q)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    
    # Add expected constellation points
    ideal_val = 1.0 / np.sqrt(2)
    ax.plot([-ideal_val, -ideal_val, ideal_val, ideal_val],
            [-ideal_val, ideal_val, ideal_val, -ideal_val],
            'rx', markersize=10, markeredgewidth=2, label='Ideal')
    ax.legend()


def plot_eye_diagram(ax, signal: np.ndarray, sps: int, title: str,
                     n_traces: int = 200):
    """Plot eye diagram"""
    samples_per_trace = 2 * sps
    
    for i in range(min(n_traces, len(signal) // sps)):
        start = i * sps
        if start + samples_per_trace > len(signal):
            break
        trace = signal[start:start + samples_per_trace]
        ax.plot(trace, 'b-', alpha=0.05, linewidth=0.5)
    
    # Mark symbol centers
    ax.axvline(sps, color='r', linestyle='--', alpha=0.5, label='Symbol center')
    
    ax.set_xlabel('Sample Index (2 symbol periods)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_psd(ax, signal: np.ndarray, fs: float, title: str):
    """Plot power spectral density"""
    from scipy import signal as sp_signal
    f, psd = sp_signal.welch(signal, fs=fs, nperseg=min(2048, len(signal)//4))
    
    ax.semilogy(f, psd)
    ax.set_xlabel('Frequency (normalized to symbol rate)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_waveform(ax, signal: np.ndarray, title: str, max_samples: int = 2000):
    """Plot time-domain waveform"""
    plot_signal = signal[:min(len(signal), max_samples)]
    ax.plot(plot_signal, linewidth=0.5)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_comparison_histogram(ax, data1: bytes, data2: bytes, 
                              label1: str, label2: str, title: str):
    """Plot overlaid histograms for comparison"""
    values1 = np.frombuffer(data1, dtype=np.uint8)
    values2 = np.frombuffer(data2, dtype=np.uint8)
    
    ax.hist(values1, bins=256, range=(0, 256), alpha=0.5, 
            label=label1, color='steelblue', edgecolor='black')
    ax.hist(values2, bins=256, range=(0, 256), alpha=0.5, 
            label=label2, color='coral', edgecolor='black')
    
    ax.set_xlabel('Byte Value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def create_rs_interleaver_figure(data: bytes, rs_out: bytes, 
                                  interleaved: bytes, args: argparse.Namespace):
    """Figure 1: RS Encoding and Interleaving"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Input data
    ax = fig.add_subplot(gs[0, 0])
    plot_byte_histogram(ax, data, 'Input Data Distribution', 'steelblue')
    
    ax = fig.add_subplot(gs[0, 1])
    plot_bit_transitions(ax, bits_from_bytes(data[:256]), 'Input Bit Pattern')
    
    ax = fig.add_subplot(gs[0, 2])
    plot_autocorrelation(ax, data, 'Input Autocorrelation')
    
    # RS encoded
    ax = fig.add_subplot(gs[1, 0])
    plot_comparison_histogram(ax, data, rs_out, 'Input', 'RS Encoded',
                              'RS Encoding Effect')
    
    ax = fig.add_subplot(gs[1, 1])
    plot_bit_transitions(ax, bits_from_bytes(rs_out[:256]), 
                        f'RS Encoded Bit Pattern (Parity Added)')
    
    ax = fig.add_subplot(gs[1, 2])
    plot_autocorrelation(ax, interleaved, f'After Interleaving (depth={args.depth})')
    
    fig.suptitle('Stage 1: Reed-Solomon Encoding & Interleaving', 
                 fontsize=14, fontweight='bold')
    return fig


def create_scrambler_figure(interleaved: bytes, scrambled: bytes, 
                            rs_out: bytes, args: argparse.Namespace):
    """Figure 2: Scrambling"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Before scrambling
    ax = fig.add_subplot(gs[0, 0])
    plot_byte_histogram(ax, interleaved, 'Before Scrambling', 'steelblue')
    
    ax = fig.add_subplot(gs[0, 1])
    plot_bit_transitions(ax, bits_from_bytes(interleaved[:256]), 
                        'Interleaved Bit Pattern')
    
    ax = fig.add_subplot(gs[0, 2])
    plot_autocorrelation(ax, interleaved, 'Before Scrambling ACF')
    
    # After scrambling
    ax = fig.add_subplot(gs[1, 0])
    if not args.no_scramble:
        plot_comparison_histogram(ax, interleaved, scrambled, 
                                 'Before', 'After', 'Scrambling Effect')
    else:
        plot_byte_histogram(ax, scrambled, 'No Scrambling (disabled)', 'gray')
    
    ax = fig.add_subplot(gs[1, 1])
    plot_bit_transitions(ax, bits_from_bytes(scrambled[:256]), 
                        'Scrambled Bit Pattern' if not args.no_scramble else 'Unchanged')
    
    ax = fig.add_subplot(gs[1, 2])
    plot_autocorrelation(ax, scrambled, 'After Scrambling ACF')
    
    title = 'Stage 2: Scrambling' if not args.no_scramble else 'Stage 2: Scrambling (DISABLED)'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    return fig


def create_conv_encoder_figure(scrambled: bytes, convolved: bytes):
    """Figure 3: Convolutional Encoding"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Before conv encoding
    ax = fig.add_subplot(gs[0, 0])
    plot_byte_histogram(ax, scrambled, 'Before Conv Encoding', 'steelblue')
    
    ax = fig.add_subplot(gs[0, 1])
    plot_bit_transitions(ax, bits_from_bytes(scrambled[:128]), 
                        'Input Bit Pattern (255 bytes)')
    
    ax = fig.add_subplot(gs[0, 2])
    plot_autocorrelation(ax, scrambled, 'Input ACF')
    
    # After conv encoding (now 512 bytes per block)
    ax = fig.add_subplot(gs[1, 0])
    plot_byte_histogram(ax, convolved, 'After Conv Encoding (rate 1/2)', 'coral')
    
    ax = fig.add_subplot(gs[1, 1])
    plot_bit_transitions(ax, bits_from_bytes(convolved[:256]), 
                        'Encoded Bit Pattern (512 bytes)')
    
    ax = fig.add_subplot(gs[1, 2])
    plot_autocorrelation(ax, convolved, 'Encoded ACF')
    
    fig.suptitle('Stage 3: Convolutional Encoding (171,133)o, L=7', 
                 fontsize=14, fontweight='bold')
    return fig


def create_diff_encoder_figure(convolved: bytes, diff_encoded: bytes):
    """Figure 4: Differential Encoding"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Before diff encoding
    ax = fig.add_subplot(gs[0, 0])
    plot_byte_histogram(ax, convolved, 'Before Diff Encoding', 'steelblue')
    
    ax = fig.add_subplot(gs[0, 1])
    plot_bit_transitions(ax, bits_from_bytes(convolved[:256]), 
                        'Conv Encoded Pattern')
    
    ax = fig.add_subplot(gs[0, 2])
    plot_autocorrelation(ax, convolved, 'Before Diff Encoding ACF')
    
    # After diff encoding
    ax = fig.add_subplot(gs[1, 0])
    plot_comparison_histogram(ax, convolved, diff_encoded,
                             'Before', 'After', 'Differential Encoding Effect')
    
    ax = fig.add_subplot(gs[1, 1])
    plot_bit_transitions(ax, bits_from_bytes(diff_encoded[:256]), 
                        'Diff Encoded Pattern (NRZI-style)')
    
    ax = fig.add_subplot(gs[1, 2])
    plot_autocorrelation(ax, diff_encoded, 'After Diff Encoding ACF')
    
    fig.suptitle('Stage 4: Differential Encoding', 
                 fontsize=14, fontweight='bold')
    return fig


def create_qpsk_figure(diff_encoded: bytes, iq: np.ndarray, 
                       qpsk_i: np.ndarray, qpsk_q: np.ndarray,
                       args: argparse.Namespace):
    """Figure 5: QPSK Modulation"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Input
    ax = fig.add_subplot(gs[0, 0])
    plot_byte_histogram(ax, diff_encoded, 'Input to QPSK', 'steelblue')
    
    ax = fig.add_subplot(gs[0, 1])
    plot_waveform(ax, qpsk_i.astype(float), 'I Channel (baseband)', max_samples=1000)
    
    ax = fig.add_subplot(gs[0, 2])
    plot_waveform(ax, qpsk_q.astype(float), 'Q Channel (baseband)', max_samples=1000)
    
    # QPSK outputs
    ax = fig.add_subplot(gs[1, 0])
    plot_qpsk_constellation(ax, iq, f'QPSK Constellation (Q1.{args.frac_bits})', 
                           args.frac_bits)
    
    ax = fig.add_subplot(gs[1, 1])
    # I vs sample
    ax.scatter(range(min(500, len(qpsk_i))), 
              qpsk_i[:500].astype(float), s=1, alpha=0.5)
    ax.set_xlabel('Symbol Index')
    ax.set_ylabel('I Amplitude')
    ax.set_title('I Symbol Sequence')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[1, 2])
    # Q vs sample
    ax.scatter(range(min(500, len(qpsk_q))), 
              qpsk_q[:500].astype(float), s=1, alpha=0.5, color='coral')
    ax.set_xlabel('Symbol Index')
    ax.set_ylabel('Q Amplitude')
    ax.set_title('Q Symbol Sequence')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Stage 5: QPSK Modulation', 
                 fontsize=14, fontweight='bold')
    return fig


def create_rrc_figure(qpsk_i: np.ndarray, qpsk_q: np.ndarray,
                      qpsk_i_rrc: np.ndarray, qpsk_q_rrc: np.ndarray):
    """Figure 6: RRC Pulse Shaping"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Filter characteristics
    ax = fig.add_subplot(gs[0, 0])
    coeffs_float = np.array([fixed_to_float(c, COEFF_FRAC) for c in RRC_COEFFS])
    ax.stem(coeffs_float, basefmt=' ')
    ax.set_xlabel('Tap Index')
    ax.set_ylabel('Coefficient')
    ax.set_title(f'RRC Impulse Response ({NUM_TAPS} taps)')
    ax.grid(True, alpha=0.3)
    
    ax = fig.add_subplot(gs[0, 1])
    freq_response = np.fft.fft(coeffs_float, 4096)
    freq_db = 20 * np.log10(np.abs(freq_response) + 1e-10)
    freq_axis = np.linspace(0, 1, len(freq_db))
    ax.plot(freq_axis, freq_db)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(f'Frequency Response (β={ROLL_OFF})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-60, 5)
    
    ax = fig.add_subplot(gs[0, 2])
    ax.text(0.5, 0.5, f'RRC Filter Parameters:\n\n'
                      f'Samples/Symbol: {SAMPLES_PER_SYMBOL}\n'
                      f'Span: {FILTER_SPAN} symbols\n'
                      f'Roll-off (β): {ROLL_OFF}\n'
                      f'Taps: {NUM_TAPS}\n'
                      f'Format: Q2.16',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.axis('off')
    
    # Before/after waveforms - I channel
    ax = fig.add_subplot(gs[1, 0])
    plot_waveform(ax, qpsk_i.astype(float), 
                 'I Channel Before RRC', max_samples=200)
    
    ax = fig.add_subplot(gs[1, 1])
    plot_waveform(ax, qpsk_i_rrc.astype(float), 
                 'I Channel After RRC (upsampled)', max_samples=1600)
    
    ax = fig.add_subplot(gs[1, 2])
    plot_eye_diagram(ax, qpsk_i_rrc.astype(float), SAMPLES_PER_SYMBOL,
                    'Eye Diagram - I Channel', n_traces=100)
    
    # Q channel
    ax = fig.add_subplot(gs[2, 0])
    plot_waveform(ax, qpsk_q.astype(float), 
                 'Q Channel Before RRC', max_samples=200)
    
    ax = fig.add_subplot(gs[2, 1])
    plot_waveform(ax, qpsk_q_rrc.astype(float), 
                 'Q Channel After RRC (upsampled)', max_samples=1600)
    
    ax = fig.add_subplot(gs[2, 2])
    # PSD
    signal_norm = qpsk_i_rrc.astype(float) / (2**15)
    plot_psd(ax, signal_norm, fs=SAMPLES_PER_SYMBOL, 
            title='Power Spectral Density')
    
    fig.suptitle('Stage 6: Root Raised Cosine Pulse Shaping', 
                 fontsize=14, fontweight='bold')
    return fig


def create_decoder_figure(diff_encoded: bytes, iq: np.ndarray,
                         qpsk_i_rrc: np.ndarray, qpsk_q_rrc: np.ndarray,
                         args: argparse.Namespace):
    """Figure 7: Receiver/Decoder Path"""
    from commpy.channelcoding.convcode import Trellis, viterbi_decode
    import reedsolo
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # RRC receiver (matched filter + downsample)
    if args.rrc:
        # Matched filtering
        i_matched = fir_filter(qpsk_i_rrc, RRC_COEFFS)
        q_matched = fir_filter(qpsk_q_rrc, RRC_COEFFS)
        
        ax = fig.add_subplot(gs[0, 0])
        plot_waveform(ax, i_matched.astype(float), 
                     'After Matched Filter (I)', max_samples=1600)
        
        # Downsample
        group_delay = (NUM_TAPS - 1) // 2
        total_delay = 2 * group_delay
        i_down = i_matched[total_delay::SAMPLES_PER_SYMBOL]
        q_down = q_matched[total_delay::SAMPLES_PER_SYMBOL]
        
        # Trim and interleave
        num_symbols = len(iq) // 2
        i_recovered = np.clip(i_down[:num_symbols], -32768, 32767).astype(np.int16)
        q_recovered = np.clip(q_down[:num_symbols], -32768, 32767).astype(np.int16)
        iq_recovered = np.empty(len(i_recovered) * 2, dtype=np.int16)
        iq_recovered[0::2] = i_recovered
        iq_recovered[1::2] = q_recovered
        
        ax = fig.add_subplot(gs[0, 1])
        plot_qpsk_constellation(ax, iq_recovered, 
                               'Recovered Constellation', args.frac_bits)
        
        # Hard demod
        diff_encoded_recovered = qpsk_demod_hard_fixed(iq_recovered, interleaved=True)
        
        ax = fig.add_subplot(gs[0, 2])
        errors = sum(a != b for a, b in zip(diff_encoded, diff_encoded_recovered))
        ax.text(0.5, 0.5, f'QPSK Hard Demod:\n\n'
                          f'Total bytes: {len(diff_encoded)}\n'
                          f'Bit errors: {errors * 8}\n'
                          f'BER: {errors * 8 / (len(diff_encoded) * 8):.2e}',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', 
                                      facecolor='lightgreen' if errors == 0 else 'lightcoral'))
        ax.axis('off')
        
        start_bytes = diff_encoded_recovered
    else:
        start_bytes = diff_encoded
        
        ax = fig.add_subplot(gs[0, :])
        ax.text(0.5, 0.5, 'RRC disabled - using direct QPSK symbols',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
        ax.axis('off')
    
    # Differential decode
    de_diff = diff_decode(start_bytes)
    
    ax = fig.add_subplot(gs[1, 0])
    plot_comparison_histogram(ax, start_bytes, de_diff,
                             'Diff Encoded', 'Decoded', 'Differential Decoding')
    
    # Viterbi decode
    L = 7
    G1_OCT = 0o171
    G2_OCT = 0o133
    DATA_BITS_PER_BLOCK = CODEWORD_BYTES * 8
    TAIL_BITS = L - 1
    ENC_BITS_PER_BLOCK = 2 * (DATA_BITS_PER_BLOCK + TAIL_BITS)
    ENC_BYTES_PER_BLOCK = (ENC_BITS_PER_BLOCK + 7) // 8
    
    def _bits_from_bytes(data):
        out = []
        for b in data:
            for i in range(8):
                out.append((b >> (7 - i)) & 1)
        return out
    
    def _bytes_from_bits(bits):
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
    
    trellis = Trellis(np.array([L-1]), np.array([[G1_OCT, G2_OCT]], dtype=int))
    viterbi_out_bits = []
    for off in range(0, len(de_diff), ENC_BYTES_PER_BLOCK):
        blk_bytes = de_diff[off:off + ENC_BYTES_PER_BLOCK]
        bits_full = _bits_from_bytes(blk_bytes)
        bits = bits_full[:ENC_BITS_PER_BLOCK]
        dec = viterbi_decode(np.array(bits, dtype=float), trellis,
                            tb_depth=5*(L-1), decoding_type='hard')
        viterbi_out_bits.extend(int(b) for b in dec[:DATA_BITS_PER_BLOCK])
    
    viterbi_bytes = _bytes_from_bits(viterbi_out_bits)
    
    ax = fig.add_subplot(gs[1, 1])
    plot_byte_histogram(ax, viterbi_bytes, 'After Viterbi Decode', 'steelblue')
    
    # Descramble
    descrambled = viterbi_bytes if args.no_scramble else descramble_bits(viterbi_bytes, seed=args.seed)
    
    ax = fig.add_subplot(gs[1, 2])
    plot_autocorrelation(ax, descrambled, 'After Descrambling')
    
    # Deinterleave
    deintl = deinterleave(descrambled, args.depth)
    
    ax = fig.add_subplot(gs[2, 0])
    plot_byte_histogram(ax, deintl, 'After Deinterleaving', 'coral')
    
    # RS decode
    RS = reedsolo.RSCodec(32, nsize=255, c_exp=8, generator=2, fcr=1, prim=0x187)
    restored = bytearray()
    rs_errors = 0
    for off in range(0, len(deintl), CODEWORD_BYTES):
        cw = deintl[off:off + CODEWORD_BYTES]
        try:
            msg, _, errata = RS.decode(cw)
            rs_errors += len(errata) if errata else 0
            restored.extend(msg)
        except Exception as e:
            ax = fig.add_subplot(gs[2, 1:])
            ax.text(0.5, 0.5, f'RS Decode Error:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightcoral'))
            ax.axis('off')
            fig.suptitle('Stage 7: Receiver/Decoder Path (RRC RX -> Diff Decode -> Viterbi -> Descramble -> Deinterleave -> RS)', 
                         fontsize=14, fontweight='bold')
            return fig
    
    ax = fig.add_subplot(gs[2, 1])
    plot_byte_histogram(ax, bytes(restored), 'Final Decoded Output', 'green')
    
    # Final comparison
    ax = fig.add_subplot(gs[2, 2])
    try:
        decoded_text = restored.decode('utf-8', errors='replace')
        success_text = f'[OK] Decode Success!\n\n'
        success_text += f'RS corrections: {rs_errors}\n'
        success_text += f'Output length: {len(restored)} bytes\n\n'
        success_text += f'Text preview:\n"{decoded_text[:100]}..."'
        ax.text(0.5, 0.5, success_text,
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen'),
               family='monospace')
    except:
        ax.text(0.5, 0.5, f'Decoded {len(restored)} bytes\n(binary data)',
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))
    ax.axis('off')
    
    fig.suptitle('Stage 7: Receiver/Decoder Path (RRC RX -> Diff Decode -> Viterbi -> Descramble -> Deinterleave -> RS)', 
                 fontsize=14, fontweight='bold')
    return fig


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description="Visualize the communication pipeline stages (generates multiple figures)"
    )
    
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="UTF-8 text input.")
    src.add_argument("--hex", help="Hex string input (spaces allowed).")
    src.add_argument("--infile", help="Binary input file.")
    
    p.add_argument("--depth", type=int, default=2, 
                  help="Interleaver depth I (default: 2)")
    p.add_argument("--no-scramble", action="store_true",
                  help="Disable scrambler (enabled by default).")
    p.add_argument("--seed", type=lambda s: int(s, 0), default=DEFAULT_SEED,
                  help="Override scrambler seed (int).")
    p.add_argument("--frac-bits", type=int, default=FRAC_BITS_DEFAULT,
                  help="Fractional bits for QPSK (default: 15).")
    p.add_argument("--rrc", action="store_true", default=True,
                  help="Apply RRC pulse shaping (default: enabled).")
    p.add_argument("--no-rrc", dest="rrc", action="store_false",
                  help="Disable RRC pulse shaping.")
    p.add_argument("--save", help="Save figures to files with this prefix (e.g., 'pipeline' -> 'pipeline_1_rs.png').")
    p.add_argument("--dpi", type=int, default=150,
                  help="DPI for saved figures (default: 150).")
    p.add_argument("--skip-decoder", action="store_true",
                  help="Skip decoder figure (faster for encoder-only analysis).")
    
    args = p.parse_args(argv)
    
    # Read input
    if args.text is not None:
        data = args.text.encode("utf-8")
    elif args.hex is not None:
        s = args.hex.replace(" ", "").replace("\n", "")
        try:
            data = bytes.fromhex(s)
        except ValueError:
            print("Error: --hex contains non-hex characters.", file=sys.stderr)
            return 2
    elif args.infile is not None:
        with open(args.infile, "rb") as f:
            data = f.read()
    else:
        # Default test message
        data = b"HELLO WORLD! This is a test message for the communication pipeline visualization. " \
               b"The message should be long enough to show interesting patterns in the various stages. " \
               b"Let's add some more text to make it even more interesting!"
    
    # Prepare data
    data_padded = pad_to_block(data, RS_K)
    
    print(f"Processing {len(data_padded)} bytes through pipeline...")
    print(f"Configuration: depth={args.depth}, scramble={not args.no_scramble}, RRC={args.rrc}")
    
    try:
        # Process through pipeline stages
        print("  - RS encoding...")
        rs_out = rs_encode(data_padded)
        
        print("  - Interleaving...")
        interleaved = interleave(rs_out, args.depth)
        
        print("  - Scrambling...")
        if args.no_scramble:
            scrambled = interleaved
        else:
            scrambled = scramble_bits(interleaved, seed=args.seed)
        
        print("  - Convolutional encoding...")
        convolved = conv_encode(scrambled)
        
        print("  - Differential encoding...")
        diff_encoded = diff_encode(convolved)
        
        print("  - QPSK modulation...")
        iq = qpsk_modulate_bytes_fixed(diff_encoded, frac_bits=args.frac_bits, 
                                       interleaved=True)
        qpsk_i = iq[0::2]
        qpsk_q = iq[1::2]
        
        # RRC filtering if enabled
        qpsk_i_rrc = None
        qpsk_q_rrc = None
        if args.rrc:
            print("  - RRC pulse shaping...")
            # Upsample
            qpsk_i_up = upsample(qpsk_i, SAMPLES_PER_SYMBOL)
            qpsk_q_up = upsample(qpsk_q, SAMPLES_PER_SYMBOL)
            
            # Pad for filter delay
            group_delay = (NUM_TAPS - 1) // 2
            total_delay = 2 * group_delay
            qpsk_i_up_padded = np.concatenate([qpsk_i_up, 
                                               np.zeros(total_delay, dtype=np.int16)])
            qpsk_q_up_padded = np.concatenate([qpsk_q_up, 
                                               np.zeros(total_delay, dtype=np.int16)])
            
            # Filter
            qpsk_i_rrc, qpsk_q_rrc = rrc_filter(qpsk_i_up_padded, qpsk_q_up_padded)
        
        print("\nGenerating figures...")
        
        # Generate all figures
        figures = []
        figure_names = []
        
        if not args.skip_decoder:
            print("  [7/7] Decoder Path...")
            fig7 = create_decoder_figure(diff_encoded, iq, 
                                         qpsk_i_rrc if args.rrc else qpsk_i,
                                         qpsk_q_rrc if args.rrc else qpsk_q,
                                         args)
            figures.append(fig7)
            figure_names.append("7_decoder")
        
        if args.rrc:
            print("  [6/7] RRC Pulse Shaping...")
            fig6 = create_rrc_figure(qpsk_i, qpsk_q, qpsk_i_rrc, qpsk_q_rrc)
            figures.append(fig6)
            figure_names.append("6_rrc")
        
        print("  [5/7] QPSK Modulation...")
        fig5 = create_qpsk_figure(diff_encoded, iq, qpsk_i, qpsk_q, args)
        figures.append(fig5)
        figure_names.append("5_qpsk")
        
        print("  [4/7] Differential Encoding...")
        fig4 = create_diff_encoder_figure(convolved, diff_encoded)
        figures.append(fig4)
        figure_names.append("4_diff_encoder")
        
        print("  [3/7] Convolutional Encoding...")
        fig3 = create_conv_encoder_figure(scrambled, convolved)
        figures.append(fig3)
        figure_names.append("3_conv_encoder")
        
        print("  [2/7] Scrambling...")
        fig2 = create_scrambler_figure(interleaved, scrambled, rs_out, args)
        figures.append(fig2)
        figure_names.append("2_scrambler")
        
        print("  [1/7] RS & Interleaving...")
        fig1 = create_rs_interleaver_figure(data_padded, rs_out, interleaved, args)
        figures.append(fig1)
        figure_names.append("1_rs_interleaver")
        
        # Save or show
        if args.save:
            print(f"\nSaving figures with prefix '{args.save}'...")
            for fig, name in zip(figures, figure_names):
                filename = f"{args.save}_{name}.png"
                fig.savefig(filename, dpi=args.dpi, bbox_inches='tight')
                print(f"  Saved: {filename}")
            print("All figures saved successfully!")
            
            # Close all figures
            for fig in figures:
                plt.close(fig)
        else:
            print("\nDisplaying figures interactively...")
            print("Close each figure window to see the next one.")
            for idx, (fig, name) in enumerate(zip(figures, figure_names), 1):
                print(f"  Showing figure {idx}/{len(figures)}: {name}")
                plt.show()
                plt.close(fig)
        
        print("\n[OK] Visualization complete!")
        
    except Exception as e:
        print(f"\nError creating visualizations: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())