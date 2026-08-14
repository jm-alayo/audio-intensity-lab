import logging
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from research.hpss_replication_96.common import AUDIO_DIR, list_dataset_songs, load_segments, perc_ratio_actual, perc_ratio_fast

KERNELS = [9, 11, 17]
OUT_CSV = Path(__file__).parent / "evidence" / "perc_ratio_per_segment.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_worker():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _process_song(music_name: str) -> list[dict]:

    mp3_path = AUDIO_DIR / f"{music_name}.mp3"

    try:
        segments, sr = load_segments(mp3_path)

    except Exception as exc:
        logger.warning(f"[SKIP] {music_name}: {exc}")
        return []

    rows = []
    for idx, seg in enumerate(segments, start=1):
        t0 = time.time()
        k31 = perc_ratio_actual(seg)
        t_k31 = time.time() - t0

        row = {"music_name": music_name, "segmento": idx, "perc_ratio_k31": k31, "hpss_k31_time_s": round(t_k31, 4)}

        for k in KERNELS:
            row[f"perc_ratio_k{k}"] = perc_ratio_fast(seg, k)

        rows.append(row)

    return rows


def main():
    songs = list_dataset_songs()
    logger.info(f"Procesando {len(songs)} canciones etiquetadas de {AUDIO_DIR.name}...")

    t0 = time.time()
    all_rows = []

    with Pool(processes=3, initializer=_init_worker) as pool:
        for rows in tqdm(pool.imap_unordered(_process_song, songs), total=len(songs), desc="perc_ratio", unit="cancion"):
            all_rows.extend(rows)

    dt = time.time() - t0

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, sep=";", index=False, encoding="utf-8")

    logger.info(f"OK: {len(df)} segmentos de {df['music_name'].nunique()} canciones -> {OUT_CSV}")
    logger.info(f"Tiempo total: {dt:.1f}s")


if __name__ == "__main__":
    main()
