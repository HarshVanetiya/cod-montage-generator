"""Stage 2a - audio intensity per second (loudness + density of gunfire/explosion hits).

The WAV is read in one-minute blocks, so memory stays flat no matter how long the
recording is. Only small per-frame arrays (~43 values per second) are kept.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import log

HOP = 512
N_FFT = 1024
BLOCK_SECONDS = 60


def audio_features(wav: Path, sr: int) -> dict[str, np.ndarray]:
    """Per-STFT-frame RMS (dB) and onset strength for the whole recording."""
    import librosa
    import soundfile as sf

    rms_parts, onset_parts = [], []
    block = sr * BLOCK_SECONDS
    block -= block % HOP  # whole hops per block so frame times don't drift
    for y in sf.blocks(str(wav), blocksize=block, dtype="float32", fill_value=None):
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < N_FFT:
            y = np.pad(y, (0, N_FFT - len(y)))
        n = len(y) // HOP
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP, center=True)[0][:n]
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, n_fft=N_FFT, center=True)[:n]
        rms_parts.append(rms)
        onset_parts.append(onset)

    rms = np.concatenate(rms_parts) if rms_parts else np.zeros(0, np.float32)
    onset = np.concatenate(onset_parts) if onset_parts else np.zeros(0, np.float32)
    return {
        "rms_db": (20 * np.log10(np.maximum(rms, 1e-5))).astype(np.float32),
        "onset": onset.astype(np.float32),
        "frame_rate": np.float32(sr / HOP),
    }


def per_second(feats: dict[str, np.ndarray], n_sec: int) -> dict[str, np.ndarray]:
    """Turn frame features into per-second intensity in 0..1 (normalised per recording)."""
    zeros = np.zeros(n_sec, np.float32)
    rms_db, onset = feats.get("rms_db"), feats.get("onset")
    if rms_db is None or len(rms_db) == 0:
        return {"loudness": zeros, "density": zeros.copy(), "intensity": zeros.copy()}

    fr = float(feats["frame_rate"])
    idx = np.minimum((np.arange(len(rms_db)) / fr).astype(int), n_sec - 1)

    # Loudness: the loudest moment of each second, smoothed over ~0.1 s so a
    # single click doesn't count.
    smooth = np.convolve(rms_db, np.ones(5) / 5, mode="same")
    loud = np.full(n_sec, -np.inf, np.float32)
    np.maximum.at(loud, idx, smooth)
    loud[~np.isfinite(loud)] = np.median(smooth)

    # Gunfire / explosion density: count sharp transients per second.
    # Threshold is relative to this recording so quiet and loud captures both work.
    med = np.median(onset)
    mad = np.median(np.abs(onset - med)) * 1.4826 + 1e-6
    thr = med + 4 * mad
    is_peak = np.zeros(len(onset), bool)
    if len(onset) > 2:
        is_peak[1:-1] = (onset[1:-1] > thr) & (onset[1:-1] >= onset[:-2]) & (onset[1:-1] > onset[2:])
    density = np.bincount(idx[is_peak], minlength=n_sec)[:n_sec].astype(np.float32)

    def norm(x: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
        lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    loud_n = norm(loud, 50, 99)
    dens_n = norm(density, 0, 99)
    intensity = (0.55 * loud_n + 0.45 * dens_n).astype(np.float32)
    log.debug("audio: loud p50=%.1fdB p99=%.1fdB, onset p99=%.1f/s",
              np.percentile(loud, 50), np.percentile(loud, 99), np.percentile(density, 99))
    return {"loudness": loud_n, "density": dens_n, "intensity": intensity}
