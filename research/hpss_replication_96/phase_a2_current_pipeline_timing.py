import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pyloudnorm as pyln

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import features.extract as extract_mod
from features.extract import extract_segment
from features.config import N_WORKERS
from research.hpss_replication_96.common import AUDIO_DIR, load_segments
from research.hpss_replication_96.phase_a_worker_benchmark import select_sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_worker():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    logger.setLevel(logging.WARNING)
    extract_mod.logger.setLevel(logging.WARNING)


def _process_song(music_name: str) -> float:

    mp3_path = AUDIO_DIR / f"{music_name}.mp3"
    segments, sr = load_segments(mp3_path)
    meter = pyln.Meter(sr)

    t0 = time.time()
    for b_idx, seg in enumerate(segments):
        extract_segment(seg, sr, b_idx=b_idx, meter=meter)

    return time.time() - t0


def main():

    sample, n_segments = select_sample(random_selection=2)
    logger.info(f"N_WORKERS actual de produccion: {N_WORKERS}")
    logger.info(f"Muestra: {len(sample)} canciones")

    t0 = time.time()
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        durations = list(pool.imap_unordered(_process_song, sample))
    wall = time.time() - t0

    logger.info(f"Wall clock ({len(sample)} canciones, {N_WORKERS} workers, sin HPSS): {wall:.2f}s")
    logger.info(f"Extrapolado a 96 canciones: {wall / len(sample) * 96:.1f}s ({wall / len(sample) * 96 / 60:.2f} min)")


if __name__ == "__main__":
    main()
