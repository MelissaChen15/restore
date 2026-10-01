import numpy as np
import soundfile as sf
import soxr


def to_stereo(x):
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 2:
        return x
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    mono = x.mean(axis=1)
    return np.stack([mono, mono], axis=1)


def load_audio(path, target_sr=None):
    x, sr = sf.read(str(path), always_2d=False, dtype="float64")
    x = to_stereo(x)
    if target_sr is not None and sr != target_sr:
        x = soxr.resample(x, sr, target_sr, quality="HQ")
        sr = target_sr
    return x, sr


def _tpdf_dither(shape, lsb):
    u1 = np.random.uniform(-lsb / 2, lsb / 2, size=shape)
    u2 = np.random.uniform(-lsb / 2, lsb / 2, size=shape)
    return u1 + u2


def save_wav_pcm16(path, x, sr, dither=True):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0 - 1.0 / 32768.0)
    if dither:
        lsb = 1.0 / 32768.0
        x = np.clip(x + _tpdf_dither(x.shape, lsb), -1.0, 1.0 - 1.0 / 32768.0)
    sf.write(str(path), x, sr, subtype="PCM_16")


def save_wav_pcm24(path, x, sr):
    x = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0 - 1.0 / (1 << 23))
    sf.write(str(path), x, sr, subtype="PCM_24")


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def normalize_dbfs(x, target_dbfs=-23.0, eps=1e-12):
    cur = rms(x)
    target_rms = 10 ** (target_dbfs / 20.0)
    return x * (target_rms / (cur + eps))
