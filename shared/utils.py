import re
from pathlib import Path

import numpy as np
import pandas as pd

def filename_key(filename: str) -> str:
    return filename.removesuffix(".mp3").lower()

def find_latest_summary(out_dir: Path, playlist: str, id: int = None) -> Path | None:

    if id is not None:
        candidatos = sorted(out_dir.glob(f"_summary_{id}_{playlist}_segmin*.csv"))

        return candidatos[0] if candidatos else None

    pattern = re.compile(rf"^_summary_(\d+)_{re.escape(playlist)}_segmin\d+\.csv$")
    candidatos = [(int(m.group(1)), f) for f in out_dir.glob(f"_summary_*_{playlist}_segmin*.csv") if (m := pattern.match(f.name))]

    return max(candidatos, key=lambda t: t[0])[1] if candidatos else None

def find_current_id(out_dir: Path, playlist: str, segmin: int) -> int:

    pattern = re.compile(rf"^_(?:summary|features)_(\d+)_{re.escape(playlist)}_segmin{segmin}\.csv$")
    ids = [
        int(m.group(1)) for f in out_dir.glob(f"_*_{playlist}_segmin{segmin}.csv") 
            if (m := pattern.match(f.name))
    ]

    return max(ids, default=1)

def next_free_id(out_dir: Path, playlist: str, segmin: int) -> int:

    pattern = re.compile(rf"^_(?:summary|features)_(\d+)_{re.escape(playlist)}_segmin{segmin}\.csv$")
    ids = [
        int(m.group(1)) for f in out_dir.glob(f"_*_{playlist}_segmin{segmin}.csv") 
            if (m := pattern.match(f.name))
    ]

    return max(ids, default=0) + 1

def slope(values: list[float]) -> float:

    if len(values) <= 1:
        return 0.0

    x = np.arange(len(values))
    m, _ = np.polyfit(x, values, 1)

    return float(m)

def delta_extremos(values: list[float], frac: float = 0.25) -> float:
    
    n = len(values)
    k = max(1, int(round(n * frac)))

    return float(np.mean(values[-k:]) - np.mean(values[:k]))


def agg_media(values: list[float]) -> float:
    return round(float(np.mean(values)), 4)


def agg_mediana(values: list[float]) -> float:
    return round(float(np.median(values)), 4)


def normalize(v: float, lo: float, hi: float, kind: str = "linear") -> float:

    if kind == "linear":
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    elif kind == "sqrt":
        return float(np.clip((np.sqrt(v) - np.sqrt(lo)) / (np.sqrt(hi) - np.sqrt(lo)), 0.0, 1.0))

    raise ValueError(f"kind desconocido: {kind!r}")


def percentile(values: list[float], p: float) -> float:

    vals = sorted(values)

    if not vals:
        return float("nan")

    k = (len(vals) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(vals) - 1)

    if f == c:
        return vals[f]

    return vals[f] + (vals[c] - vals[f]) * (k - f)
