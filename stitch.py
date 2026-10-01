"""Crossover stitching for the content/extension seam.
"""
import numpy as np
from scipy.signal import firwin, lfilter, welch

LP_CUTOFF_HZ = 4700.0      
HP_MIN_HZ = 2600.0         
HP_MAX_HZ = 4100.0         
XOVER_MARGIN_DB = 3.0     
BAND_HZ = 200.0            
N_TAPS = 1025           
KAISER_BETA = 7.9


def _band_db(x, sr, lo_hz, hi_hz):
    freqs, psd = welch(x.mean(axis=1) if x.ndim == 2 else x, sr, nperseg=8192)
    band = (freqs >= lo_hz) & (freqs < hi_hz)
    return 10 * np.log10(psd[band].mean() + 1e-20)


def pick_crossover_hz(content, extension, sr, hp_min=HP_MIN_HZ, hp_max=HP_MAX_HZ,
                       band_hz=BAND_HZ, margin_db=XOVER_MARGIN_DB):
    centers = np.arange(hp_min, hp_max + 1, 100.0)
    ok = [
        _band_db(extension, sr, c - band_hz / 2, c + band_hz / 2)
        >= _band_db(content, sr, c - band_hz / 2, c + band_hz / 2) - margin_db
        for c in centers
    ]
    for i in range(len(centers) - 1):
        if ok[i] and ok[i + 1]:
            return float(centers[i])
    return hp_max


def stitch_restored(vocals, music, extension, sr, lp_hz=LP_CUTOFF_HZ,
                     hp_min=HP_MIN_HZ, hp_max=HP_MAX_HZ):
    content = vocals + music
    crossover_hz = pick_crossover_hz(content, extension, sr, hp_min, hp_max)
    lp = firwin(N_TAPS, lp_hz, window=("kaiser", KAISER_BETA), fs=sr)
    hp = firwin(N_TAPS, crossover_hz, window=("kaiser", KAISER_BETA), pass_zero="highpass", fs=sr)
    k = min(len(content), len(extension))
    restored = lfilter(lp, [1.0], content[:k], axis=0) + lfilter(hp, [1.0], extension[:k], axis=0)
    return restored, crossover_hz
