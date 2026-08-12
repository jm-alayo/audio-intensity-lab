import re
from pathlib import Path

import numpy as np
import pandas as pd


def filename_key(filename: str) -> str:
    return filename.removesuffix(".mp3").lower()


def load_tracks(xlsx_path: Path) -> pd.DataFrame:

    df = pd.read_excel(xlsx_path, sheet_name="data", dtype=str)

    return df[
        (df["categorico"].str.strip() != "sin_categoria")
        & (df["album"].str.strip() == "album-rock")
    ]


def find_latest_summary(out_dir: Path, playlist: str, id: int = None) -> Path | None:

    if id is not None:
        candidates = sorted(out_dir.glob(f"_summary_{id}_{playlist}_segmin*.csv"))

        return candidates[0] if candidates else None

    pattern = re.compile(rf"^_summary_(\d+)_{re.escape(playlist)}_segmin\d+\.csv$")
    candidates = [(int(m.group(1)), f) for f in out_dir.glob(f"_summary_*_{playlist}_segmin*.csv") if (m := pattern.match(f.name))]

    return max(candidates, key=lambda t: t[0])[1] if candidates else None


def find_current_id(out_dir: Path, playlist: str, segmin: int = 25) -> int:

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


def extremes_delta(values: list[float], frac: float = 0.25) -> float:

    n = len(values)
    k = max(1, int(round(n * frac)))

    return float(np.mean(values[-k:]) - np.mean(values[:k]))


def aggregate_mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4)


def aggregate_median(values: list[float]) -> float:
    return round(float(np.median(values)), 4)


def normalize(v: float, lo: float, hi: float, kind: str = "linear") -> float:

    if kind == "linear":
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    elif kind == "sqrt":
        return float(np.clip((np.sqrt(v) - np.sqrt(lo)) / (np.sqrt(hi) - np.sqrt(lo)), 0.0, 1.0))

    raise ValueError(f"unknown kind: {kind!r}")


def percentile(values: list[float], p: float) -> float:

    vals = sorted(values)

    if not vals:
        return float("nan")

    k = (len(vals) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(vals) - 1)

    if f == c:
        return vals[f]

    return vals[f] + (vals[c] - vals[f]) * (k - f)


def prepare_summary(summary_file: Path, xlsx_path: Path) -> pd.DataFrame:

    df = pd.read_csv(summary_file, sep=";")
    labels = load_tracks(xlsx_path)

    labels["categorico"] = labels["categorico"].replace(
        {"incrementable": "incrementable-decreciente", "decreciente": "incrementable-decreciente"})

    df = df.merge(labels[["music_name", "categorico"]], on="music_name")

    return df[df["categorico"].str.strip() != "sin_categoria"]


def derive_threshold_bounds(df: pd.DataFrame, p_lo: float = 5, p_hi: float = 95) -> list[tuple]:

    def bound(values: pd.Series) -> tuple:
        vals = values.tolist()

        return percentile(vals, p_lo), percentile(vals, p_hi)

    energy_change = (df["pendiente_energia"] * (df["n_segments"] - 1)).abs()
    energy_delta = df["delta_energia"].abs()

    return [
        bound(df["score_energia_mediana"]),
        bound(df["score_ritmo_mediana"]),
        bound(df["score_noise_mediana"]),
        bound(energy_change),
        bound(energy_delta),
    ]
