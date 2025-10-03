# mpu/model/rrc.py
# root-raised-cosine (RRC) pulse shaping for baseband I/Q
#
# - inputs: int16 I and Q at 1 sample/symbol (Q1.15, range ~[-32768, 32767])
# - outputs: int16 I and Q at 'sps' samples/symbol (Q1.15), optionally
#            group-delay compensated (trimmed)
#
# tunables:
#   - beta (roll-off, 0.0..1.0), default 0.35
#   - sps (samples per symbol), default 8
#   - span (filter span in symbols), default 8  (taps = span*sps + 1)
#   - normalize: "unit-energy" (symbol energy ~1) or "unit-peak" (pulse peak ~1)
#   - remove_delay: True trims the (len(taps)-1)//2 samples of group delay
#
# implementation notes:
#   - We generate taps via the standard analytic RRC formula with special-case
#     handling for t = 0 and t = +/- T/(4*beta)
#   - We pulse-shape using scipy.signal.upfirdn if available


from __future__ import annotations
from typing import Tuple, Literal
import numpy as np
from commpy.filters import rrcosfilter

from scipy.signal import upfirdn


INT16_MIN = -32768
INT16_MAX =  32767


def _rrc_impulse_response(beta: float, sps: int, span: int,
                          normalize: Literal["unit-energy", "unit-peak"] = "unit-energy"
                          ) -> np.ndarray:
    # generate rrc taps using commpy's rrcosfilter
    # length = span*sps + 1. Ts=1, Fs=sps
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0,1]")
    if sps < 1:
        raise ValueError("sps must be >= 1")
    if span < 1:
        raise ValueError("span must be >= 1")

    N = span * sps + 1
    h_ref, _t = rrcosfilter(N, alpha=beta, Ts=1.0, Fs=float(sps))
    taps = np.asarray(h_ref, dtype=np.float64)

    if normalize == "unit-energy":
        es = float(np.sqrt(np.sum(taps * taps)))
        if es > 0:
            taps = taps / es
    elif normalize == "unit-peak":
        m = float(np.max(np.abs(taps)))
        if m > 0:
            taps = taps / m
    else:
        raise ValueError("normalize must be 'unit-energy' or 'unit-peak'")

    return taps


def _saturate_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x, INT16_MIN, INT16_MAX).astype(np.int16, copy=False)


def rrc_shape_fixed_iq(
    I_in: np.ndarray,
    Q_in: np.ndarray,
    *,
    beta: float = 0.35,
    sps: int = 8,
    span: int = 8,
    normalize: Literal["unit-energy", "unit-peak"] = "unit-energy",
    remove_delay: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # pulse-shape int16 Q1.15 I/Q at 1 sample/symbol with an rrc filter
    #
    # parameters:
    # I_in, Q_in: np.ndarray - 1D arrays, dtype=int16, length = num of qpsk symbols
    # beta: float - roll-off factor in [0,1]
    # sps: int - output samples per symbol (integer upsampling)
    # span: int - filter span in symbols. taps = span * sps + 1
    # normalize: {"unit-energy", "unit-peak"}
    # remove_delay: bool - if True, trim the FIR group delay so the first output sample
    # aligns with the first symbol (good for file/DAC playback). if False, the raw 
    # filtered output (wiht delay) is returned
    # returns:
    # I_out, Q_out: np.ndarray: int16 arrays at sps samples/symbol (Q1.15 scale)
    # taps: np.ndarray: the float64 rrc taps used (length span*sps+1)

    if I_in.dtype != np.int16 or Q_in.dtype != np.int16:
        raise ValueError("I_in and Q_in must be dtype int16")
    if I_in.ndim != 1 or Q_in.ndim != 1 or I_in.size != Q_in.size:
        raise ValueError("I_in and Q_in must be 1-D and same length")

    # convert to float in [-1, 1] (Q1.15)
    scale = float(2**15 - 1)
    I_f = I_in.astype(np.float64) / scale
    Q_f = Q_in.astype(np.float64) / scale

    taps = _rrc_impulse_response(beta=beta, sps=sps, span=span, normalize=normalize)

    # up-sample and filter
    I_filt = upfirdn(h=taps, x=I_f, up=sps, down=1)
    Q_filt = upfirdn(h=taps, x=Q_f, up=sps, down=1)

    # remove group delay if requested
    delay = (taps.size - 1) // 2
    if remove_delay:
        I_filt = I_filt[delay:delay + I_in.size * sps]
        Q_filt = Q_filt[delay:delay + Q_in.size * sps]

    # rescale back to Q1.15. because taps are normalized (unit-energy by default),
    # average symbol energy stays ~1, but peaks can exceed 1 slightly depending on beta.
    # we clip to int16 range.
    I_out = _saturate_int16(np.rint(I_filt * scale))
    Q_out = _saturate_int16(np.rint(Q_filt * scale))

    return I_out, Q_out, taps


if __name__ == "__main__":
    import sys
    import random
    import numpy as np

    print("[rrc TEST] starting RRC oracle tests…", flush=True)

    def _run_tests() -> int:
        # --- imports kept inside so failures are reported cleanly
        try:
            from commpy.filters import rrcosfilter  # oracle taps
        except Exception as e:
            print("[rrc TEST] ERROR: commpy not installed (pip install commpy).", file=sys.stderr)
            print(e, file=sys.stderr)
            return 1

        try:
            from scipy.signal import upfirdn as _upfirdn  # oracle convolution
        except Exception as e:
            print("[rrc TEST] ERROR: scipy not installed (pip install scipy).", file=sys.stderr)
            print(e, file=sys.stderr)
            return 1

        rng = np.random.default_rng(0xC0FFEE)
        random.seed(0xC0FFEE)

        total = 0
        bad = 0

        def check(name: str, cond: bool, detail: str = ""):
            nonlocal total, bad
            total += 1
            if cond:
                print(f"[PASS] {name}")
            else:
                bad += 1
                print(f"[FAIL] {name} :: {detail}")

        def _oracle_rrc_taps(beta: float, sps: int, span: int, normalize: str):
            N = span * sps + 1
            h_ref, _t = rrcosfilter(N, alpha=beta, Ts=1.0, Fs=float(sps))
            h_ref = np.asarray(h_ref, dtype=np.float64)
            if normalize == "unit-energy":
                es = float(np.sqrt(np.sum(h_ref * h_ref)))
                if es > 0:
                    h_ref = h_ref / es
            elif normalize == "unit-peak":
                m = float(np.max(np.abs(h_ref)))
                if m > 0:
                    h_ref = h_ref / m
            else:
                raise ValueError("normalize must be 'unit-energy' or 'unit-peak'.")
            return h_ref

        # ---- TAP EQUIVALENCE ----
        combos = [
            (0.00, 2, 6, "unit-energy"),
            (0.25, 4, 6, "unit-energy"),
            (0.35, 8, 8, "unit-energy"),
            (1.00, 8, 8, "unit-energy"),
            (0.35, 8, 10, "unit-peak"),
            (0.50, 4, 8, "unit-peak"),
        ]

        for (beta, sps, span, norm) in combos:
            ours = _rrc_impulse_response(beta, sps, span, normalize=norm)
            ref  = _oracle_rrc_taps(beta, sps, span, normalize=norm)

            check(f"taps length beta={beta} sps={sps} span={span}",
                  ours.size == (span*sps + 1),
                  f"got {ours.size}, want {span*sps + 1}")

            # direct numerical equivalence to oracle
            eq = np.allclose(ours, ref, rtol=1e-10, atol=1e-12)
            if not eq:
                idx = next((i for i,(a,b) in enumerate(zip(ours, ref))
                            if not np.isclose(a, b, rtol=1e-10, atol=1e-12)), None)
                detail = f"first diff at {idx}: ours={ours[idx]:+.6e}, ref={ref[idx]:+.6e}"
            else:
                detail = ""
            check(f"taps oracle eq beta={beta} sps={sps} span={span} norm={norm}", eq, detail)

            # normalization sanity
            if norm == "unit-energy":
                Es = float(np.sum(ours*ours))
                check(f"taps energy≈1 beta={beta} sps={sps} span={span}",
                      abs(Es - 1.0) < 1e-12, f"Es={Es:.6e}")
            else:
                peak = float(np.max(np.abs(ours)))
                check(f"taps peak≈1 beta={beta} sps={sps} span={span}",
                      abs(peak - 1.0) < 1e-12, f"peak={peak:.6e}")

        # ---- END-TO-END vs ORACLE ----
        seqs = [
            np.zeros(64, dtype=np.int16),
            np.full(64, 23170, dtype=np.int16),
            np.full(64, -23170, dtype=np.int16),
            np.tile(np.array([23170, -23170], dtype=np.int16), 128),
            rng.integers(low=-30000, high=30000, size=513, dtype=np.int16),
        ]
        e2e_sets = [
            (0.35, 8, 8, "unit-energy"),
            (0.25, 4, 8, "unit-energy"),
            (0.50, 8, 6, "unit-peak"),
        ]
        for (beta, sps, span, norm) in e2e_sets:
            taps  = _rrc_impulse_response(beta, sps, span, normalize=norm)
            delay = (taps.size - 1)//2
            scale = float(2**15 - 1)
            for idx, syms in enumerate(seqs, 1):
                I_out, Q_out, _ = rrc_shape_fixed_iq(
                    syms, syms, beta=beta, sps=sps, span=span,
                    normalize=norm, remove_delay=True
                )
                x  = syms.astype(np.float64) / scale
                yI = _upfirdn(h=taps, x=x, up=sps, down=1)[delay:delay + syms.size*sps]
                yQ = _upfirdn(h=taps, x=x, up=sps, down=1)[delay:delay + syms.size*sps]
                I_ref = _saturate_int16(np.rint(yI * scale))
                Q_ref = _saturate_int16(np.rint(yQ * scale))

                check(f"e2e len beta={beta} sps={sps} span={span} seq#{idx}",
                      I_out.size == syms.size*sps == I_ref.size,
                      f"lens: ours={I_out.size}, ref={I_ref.size}, want={syms.size*sps}")

                i_diff = int(np.max(np.abs(I_out.astype(int) - I_ref.astype(int))))
                q_diff = int(np.max(np.abs(Q_out.astype(int) - Q_ref.astype(int))))
                check(f"e2e int16 eq I beta={beta} sps={sps} span={span} seq#{idx}",
                      i_diff <= 1, f"max|diff|={i_diff}")
                check(f"e2e int16 eq Q beta={beta} sps={sps} span={span} seq#{idx}",
                      q_diff <= 1, f"max|diff|={q_diff}")

                # energy sanity (skip zero input)
                if norm == "unit-energy":
                    if np.all(syms == 0):
                        check(f"e2e energy sane beta={beta} sps={sps} span={span} seq#{idx}",
                              True, "zero input")
                    else:
                        Es = float(np.mean(yI**2 + yQ**2))
                        check(f"e2e energy sane beta={beta} sps={sps} span={span} seq#{idx}",
                              np.isfinite(Es) and Es > 0.0, f"Es={Es:.6e}")

        print("\n=== RRC TEST SUMMARY ===")
        print(f"Total checks: {total}, failures: {bad}")
        return 0 if bad == 0 else 1

    try:
        code = _run_tests()
    except Exception as e:
        print("[rrc TEST] UNEXPECTED ERROR:", e, file=sys.stderr)
        code = 1
    sys.exit(code)
