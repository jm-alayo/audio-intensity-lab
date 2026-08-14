import sys
from pathlib import Path

import librosa
import librosa.effects
import numpy as np
import pyloudnorm as pyln

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.settings import PLAYLIST_BASE, TRACKS_XLSX
from shared.utils import load_tracks
from features.config import SR, TRIM_DB, TARGET_LUFS, MIN_LUFS, SEGMENT_MIN_DURATION

PLAYLIST = "rock_english"
AUDIO_DIR = PLAYLIST_BASE / PLAYLIST


def list_dataset_songs() -> list[str]:
    tracks = load_tracks(TRACKS_XLSX)
    folder_stems = {p.stem for p in AUDIO_DIR.glob("*.mp3")}

    return sorted(s for s in tracks["music_name"].tolist() if s in folder_stems)


def load_segments(mp3_path: Path) -> tuple[list[np.ndarray], int]:

    y, sr = librosa.load(str(mp3_path), sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

    min_seg_duration = 25
    if len(y) < sr * min_seg_duration:
        raise ValueError(f"Audio demasiado corto tras recortar silencio (<{min_seg_duration} s)")

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(y)

    if lufs > MIN_LUFS:
        y = pyln.normalize.loudness(y, lufs, TARGET_LUFS)
        y = np.clip(y, -1.0, 1.0)

    duration_s = len(y) / sr
    n_segments = max(1, int(duration_s // SEGMENT_MIN_DURATION))
    seg_len = len(y) // n_segments

    if seg_len < int(sr * 1):
        raise ValueError(f"Segmentos demasiado cortos (<1000 ms) con {n_segments} segmentos")

    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = len(y) if i == n_segments - 1 else (i + 1) * seg_len
        segments.append(y[start:end])

    return segments, sr


def perc_ratio_actual(seg: np.ndarray) -> float:

    yh, yp = librosa.effects.hpss(seg)
    e_h, e_p = float(np.sum(yh ** 2)), float(np.sum(yp ** 2))

    return e_p / (e_h + e_p + 1e-9)


def perc_ratio_fast(seg: np.ndarray, kernel: int) -> float:

    S = np.abs(librosa.stft(seg, n_fft=2048))
    H, P = librosa.decompose.hpss(S, kernel_size=kernel)
    e_h, e_p = float(np.sum(H ** 2)), float(np.sum(P ** 2))

    return e_p / (e_h + e_p + 1e-9)
