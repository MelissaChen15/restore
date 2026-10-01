"""Pseudo-QMF (PQMF) multi-band analysis/synthesis filterbank."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal.windows import kaiser
from torch import nn

from .hdemucs import HDecLayer


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Kaiser-windowed FIR lowpass prototype filter."""
    assert taps % 2 == 0, "taps must be even"
    assert 0.0 < cutoff_ratio < 1.0

    omega_c = np.pi * cutoff_ratio
    n = np.arange(taps + 1) - 0.5 * taps
    with np.errstate(invalid="ignore", divide="ignore"):
        h_ideal = np.sin(omega_c * n) / (np.pi * n)
    h_ideal[taps // 2] = cutoff_ratio
    window = kaiser(taps + 1, beta)
    return h_ideal * window


class PQMF(nn.Module):
    """K-band cosine-modulated PQMF. `analysis`: (B, 1, T) -> (B, K, T/K).
    `synthesis`: (B, K, T/K) -> (B, 1, T)."""

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        super().__init__()
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        n = np.arange(taps + 1) - taps / 2

        h_analysis = np.zeros((subbands, taps + 1))
        h_synthesis = np.zeros((subbands, taps + 1))
        for k in range(subbands):
            phase = (-1) ** k * np.pi / 4
            arg = (2 * k + 1) * (np.pi / (2 * subbands)) * n
            h_analysis[k] = 2 * h_proto * np.cos(arg + phase)
            h_synthesis[k] = 2 * h_proto * np.cos(arg - phase)

        self.subbands = subbands
        self.taps = taps
        self.register_buffer("analysis_filter", torch.from_numpy(h_analysis).float().unsqueeze(1))
        self.register_buffer("synthesis_filter", torch.from_numpy(h_synthesis).float().unsqueeze(0))

        updown = torch.zeros((subbands, subbands, subbands))
        for k in range(subbands):
            updown[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown)

        self.pad_fn = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        return F.conv1d(self.pad_fn(x), self.analysis_filter, stride=self.subbands)

    def synthesis(self, x):
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)


def match_length(z, length):
    """Center-crop or zero-pad the last dim of `z` to exactly `length`."""
    cur = z.shape[-1]
    if cur == length:
        return z
    if cur > length:
        start = (cur - length) // 2
        return z[..., start:start + length]
    pad = length - cur
    left = pad // 2
    return F.pad(z, (left, pad - left))


class PQMFDecLayer(HDecLayer):
    """Drop-in replacement for HDecLayer at the time-branch's final decoder
    layer: PQMF multi-band synthesis instead of a learned ConvTranspose1d."""

    def __init__(self, chin, chout, *args, pqmf_k=4, pqmf_taps=62,
                 pqmf_cutoff_ratio=0.142, pqmf_beta=9.0, **kwargs):
        super().__init__(chin, chout, *args, **kwargs)
        assert not self.freq, "PQMFDecLayer replaces the time-branch final layer only"
        assert not self.empty, "PQMF path requires the real (non-empty) decode branch"
        self.chout = chout
        self.pqmf_k = pqmf_k
        self.subband_conv = nn.Conv1d(self.chin, chout * pqmf_k, kernel_size=1)
        self.pqmf = PQMF(subbands=pqmf_k, taps=pqmf_taps, cutoff_ratio=pqmf_cutoff_ratio, beta=pqmf_beta)
        del self.conv_tr, self.norm2

    def _project(self, y, length):
        B, _, T_in = y.shape
        sub = self.subband_conv(y)
        sub = sub.view(B * self.chout, self.pqmf_k, T_in)
        full = self.pqmf.synthesis(sub)
        full = full.view(B, self.chout, -1)
        z = match_length(full, length)
        if not self.last:
            z = F.gelu(z)
        return z, y
