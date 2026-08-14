import argparse
import logging
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import librosa
import pandas as pd
import pyloudnorm as pyln
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.settings import FEATURES_DIR
import features.extract as extract_mod
from features.extract import extract_segment
from research.hpss_replication_96.common import AUDIO_DIR, load_segments, perc_ratio_actual

WORKER_CONFIGS = [1, 3, 16]
RANDOM_SELECTION = 1
RANDOM_SEED = 42
RAW_OUT_CSV = Path(__file__).parent / "evidence" / "benchmark_worker_times_96.csv"
SUMMARY_OUT_CSV = Path(__file__).parent / "evidence" / "worker_summary_96.csv"
SAMPLE_OUT_CSV = Path(__file__).parent / "evidence" / "worker_sample_selection_96.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def count_segments_per_song() -> pd.Series:

    features_csv = FEATURES_DIR / "_features_2_rock_english_segmin25.csv"
    df = pd.read_csv(features_csv, sep=";", encoding="utf-8-sig")

    seg_cols = [c for c in df.columns if c.startswith("seg_")]
    ref = df[df["feature"] == "onset_density"].set_index("music_name")

    return ref[seg_cols].notna().sum(axis=1)


def select_sample(random_selection: int = RANDOM_SELECTION, seed: int = RANDOM_SEED) -> tuple[list[str], list[int]]:

    counts = count_segments_per_song()
    rng = random.Random(seed)

    picked_names, picked_counts = [], []

    for n_seg, group in counts.groupby(counts):
        available = group.index.tolist()
        chosen = rng.sample(available, min(random_selection, len(available)))

        for name in chosen:
            picked_names.append(name)
            picked_counts.append(int(n_seg))

    order = sorted(range(len(picked_counts)), key=lambda i: picked_counts[i])

    return [picked_names[i] for i in order], [picked_counts[i] for i in order]


def warm_up_cache(sample: list[str]) -> None:

    for music_name in sample:
        mp3_path = AUDIO_DIR / f"{music_name}.mp3"
        librosa.load(str(mp3_path), sr=None, mono=True)


WITH_HPSS = True


def _init_worker(with_hpss: bool = True):
    global WITH_HPSS
    WITH_HPSS = with_hpss

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    logger.setLevel(logging.WARNING)
    extract_mod.logger.setLevel(logging.WARNING)


def _process_song(music_name: str) -> dict:

    mp3_path = AUDIO_DIR / f"{music_name}.mp3"
    segments, sr = load_segments(mp3_path)
    meter = pyln.Meter(sr)

    total_duration = 0.0
    total_hpss = 0.0

    for b_idx, seg in enumerate(segments):
        t0 = time.time()
        extract_segment(seg, sr, b_idx=b_idx, meter=meter)
        t1 = time.time()
        if WITH_HPSS:
            perc_ratio_actual(seg)
        t2 = time.time()

        total_duration += (t2 - t0)
        total_hpss += (t2 - t1)

    return {
        "name_file": music_name,
        "worker_id": os.getpid(),
        "execution_time_s": round(total_duration, 4),
        "hpss_time_s": round(total_hpss, 4),
        "margin_s": round(total_duration - total_hpss, 4),
    }


def run_stage(sample: list[str], n_workers: int, with_hpss: bool = True) -> tuple[list[dict], float]:

    t0 = time.time()

    with Pool(processes=n_workers, initializer=_init_worker, initargs=(with_hpss,)) as pool:
        rows = list(tqdm(
            pool.imap_unordered(_process_song, sample),
            total=len(sample), desc=f"workers_{n_workers}", unit="cancion"
        ))

    dt = time.time() - t0

    for row in rows:
        row["stage"] = f"workers_{n_workers}"
        row["total_execution_s"] = round(dt, 2)

    return rows, dt


def summarize(df: pd.DataFrame) -> pd.DataFrame:

    summary = (
        df.groupby("stage", as_index=False)
        .agg(
            total_execution_time_s=("execution_time_s", "sum"),
            avg_hpss_time_s=("hpss_time_s", "mean"),
            avg_margin_s=("margin_s", "mean"),
            total_execution_s=("total_execution_s", "first"),
            n_songs=("name_file", "nunique"),
        )
    )

    summary["ratio_overhead"] = summary["avg_margin_s"] / summary["avg_hpss_time_s"] * 100
    summary["parallelism_factor"] = summary["total_execution_time_s"] / summary["total_execution_s"]
    summary["real_time_per_song_s"] = summary["total_execution_s"] / summary["n_songs"]

    summary = summary.sort_values(by="real_time_per_song_s", ascending=True).round(2)

    return summary[[
        "stage", "total_execution_time_s", "avg_hpss_time_s", "avg_margin_s",
        "ratio_overhead", "total_execution_s", "n_songs", "parallelism_factor", "real_time_per_song_s",
    ]]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reverse-order", action="store_true",
                         help="corre WORKER_CONFIGS al reves (16,3,1) para auditar sesgo de orden/cache")
    parser.add_argument("--no-hpss", action="store_true",
                         help="corre solo extract_segment, sin HPSS, misma muestra/metodologia")
    args = parser.parse_args()

    with_hpss = not args.no_hpss

    stage_order = list(reversed(WORKER_CONFIGS)) if args.reverse_order else WORKER_CONFIGS
    suffix = ("_reversed" if args.reverse_order else "") + ("_no_hpss" if args.no_hpss else "")

    sample, n_segments = select_sample(random_selection=2)
    logger.info(f"Muestra: {len(sample)} canciones (RANDOM_SELECTION=2), n_segments {n_segments[0]}..{n_segments[-1]}")
    for name, n in zip(sample, n_segments):
        logger.info(f"  {name} -> {n} segmentos")

    pd.DataFrame({"music_name": sample, "n_segments": n_segments}).to_csv(
        SAMPLE_OUT_CSV, sep=";", index=False, encoding="utf-8")

    logger.info("Warm-up: cargando los mp3 de la muestra una vez antes de medir, para que las tres etapas arranquen con el mismo estado de cache de disco.")
    warm_up_cache(sample)

    logger.info(f"Orden de etapas: {stage_order}")

    all_rows = []
    for n_workers in stage_order:
        rows, dt = run_stage(sample, n_workers, with_hpss=with_hpss)
        logger.info(f"workers_{n_workers}: wall clock {dt:.2f}s")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    raw_out = RAW_OUT_CSV.parent / f"{RAW_OUT_CSV.stem}{suffix}.csv"
    summary_out = SUMMARY_OUT_CSV.parent / f"{SUMMARY_OUT_CSV.stem}{suffix}.csv"

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_out, sep=";", index=False, encoding="utf-8")
    logger.info(f"Evidencia cruda -> {raw_out}")

    summary = summarize(df)
    summary.to_csv(summary_out, sep=";", index=False, encoding="utf-8")
    logger.info("\n" + summary.to_string(index=False))
    logger.info(f"Resumen -> {summary_out}")


if __name__ == "__main__":
    main()
