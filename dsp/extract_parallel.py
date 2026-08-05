import os
import re
import time
import logging
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent

N_WORKERS = 3
N_WORKERS = os.cpu_count() if N_WORKERS == -1 else N_WORKERS

from extract_features import (
    process_file, 
    load_done, 
    find_current_id, 
    dump_config, 
    SUMMARY_COLS, PLAYLIST_BASE, FEATURES_DIR, SEGMENT_MIN_DURATION, CANCIONES_XLSX,
    load_canciones, 
    load_albums
)

def _next_free_id(out_dir: Path, playlist: str, segmin: int) -> int:
    pattern = re.compile(rf"^_(?:summary|features)_(\d+)_{re.escape(playlist)}_segmin{segmin}\.csv$")
    ids = [int(m.group(1)) for f in out_dir.glob(f"_*_{playlist}_segmin{segmin}.csv") if (m := pattern.match(f.name))]

    return max(ids, default=0) + 1

def _init_worker():

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    logging.getLogger("extract_features").setLevel(logging.WARNING)

def _worker(args: tuple[str, str]):

    mp3_path_str, album = args
    mp3_path = Path(mp3_path_str)

    try:

        features, summary = process_file(mp3_path, album)

        return features, summary

    except Exception as exc:

        summary = {h: "" for h in SUMMARY_COLS}
        summary.update({
            "filename": mp3_path.name,
            "album":    album,
            "error":    str(exc)[:300],
        })

        return None, summary


def run_parallel(pending: list[Path], albums: dict[str, str], out_csv: Path, out_features_csv: Path):

    total_ok = total_fail = 0

    file_exists = out_csv.exists()
    features_file_exists = out_features_csv.exists()
    
    with open(out_csv, "a", newline="", encoding="utf-8") as f, \
         open(out_features_csv, "a", newline="", encoding="utf-8") as f_feat, \
         Pool(processes=N_WORKERS, initializer=_init_worker, maxtasksperchild=20) as pool:

        header_written = file_exists
        features_header_written = features_file_exists

        job_args = [
            (str(p), albums.get(p.name, "")) for p in pending
        ]
        jobs = pool.imap_unordered(_worker, job_args, chunksize=1)

        for features, summary in tqdm(jobs, total=len(pending), desc="Extrayendo", unit="mp3"):
            if summary.get("error"):
                total_fail += 1
                tqdm.write(f"  [FAIL] {summary['filename']}: {summary['error']}")
            else:
                total_ok += 1

            pd.DataFrame([summary], columns=SUMMARY_COLS).to_csv(
                f, sep=";", index=False, header=not header_written)
            f.flush()
            header_written = True

            if features is not None:
                features.to_csv(f_feat, sep=";", index=True,
                                header=not features_header_written)
                f_feat.flush()
                features_header_written = True

    return total_ok, total_fail


def main():

    playlist = "rock_english"
    folder = PLAYLIST_BASE / playlist
    resume = False

    musics_names = load_canciones(CANCIONES_XLSX)
    albums = load_albums(CANCIONES_XLSX)

    mp3_files = [
        p for p in sorted(folder.glob("*.mp3")) 
            if p.stem in musics_names
    ]

    mp3_files_doesnt_exist_in_csv = [n for n in musics_names if n not in (p.stem for p in mp3_files)]
    logger.info(f"MP3s que no existen en disco: {len(mp3_files_doesnt_exist_in_csv)}")

    BASE_SCENARIO_CSV = SCRIPT_DIR / "out" / "scenarios" / "BASE_scenario" / "_classified_sum3_rock_english_esc_medianas_rms_clf.csv"
    SOLO_PROBLEMATICAS = False

    hardcoded = None

    logger.info(f"Procesando con {N_WORKERS} workers")

    if SOLO_PROBLEMATICAS:
        if not BASE_SCENARIO_CSV.exists():
            print(f"[AVISO] {BASE_SCENARIO_CSV} no existe -- se ignora el filtro, se procesan las {len(mp3_files)} canciones completas.")

        else:
            filtered_musics = pd.read_csv(BASE_SCENARIO_CSV, sep=";", dtype=str, keep_default_na=False)
            if "validez" not in filtered_musics.columns:
                print(f"[AVISO] {BASE_SCENARIO_CSV} no tiene columna 'validez' -- se ignora el filtro.")
            else:
                revisar = filtered_musics["revisar"].str.upper()
                validez = filtered_musics["validez"].str.upper()

                hardcoded = set(filtered_musics.loc[(revisar == "TRUE") & (validez == "FALSE"), "filename"])
                print(f"Modo solo-problematicas: {len(hardcoded)} cancion(es) (revisar=TRUE, validez=FALSE)")

    if hardcoded:
        mp3_files = [p for p in mp3_files if p.stem in hardcoded]

    if not mp3_files:
        print(f"No se encontraron MP3s en: {folder}")
        return

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if resume:
        id = find_current_id(FEATURES_DIR, playlist, SEGMENT_MIN_DURATION)
    else:
        id = _next_free_id(FEATURES_DIR, playlist, SEGMENT_MIN_DURATION)

    out_csv          = FEATURES_DIR / f"_summary_{id}_{playlist}_segmin{SEGMENT_MIN_DURATION}.csv"
    out_features_csv = FEATURES_DIR / f"_features_{id}_{playlist}_segmin{SEGMENT_MIN_DURATION}.csv"
    out_config_csv   = dump_config(FEATURES_DIR, playlist, SEGMENT_MIN_DURATION, id)

    if resume:
        done    = load_done(out_csv)
        pending = [p for p in mp3_files if p.name not in done]
    else:
        pending = mp3_files

    if not pending:
        print("Todo ya esta procesado.")
        return

    print(f"Procesando {len(pending)} canciones con {N_WORKERS} workers...")
    t0 = time.time()
    ok, fail = run_parallel(pending, albums, out_csv, out_features_csv)
    dt = time.time() - t0

    print("=" * 60)
    print(f"  OK      : {ok}")
    print(f"  Errores : {fail}")
    print(f"  Tiempo  : {dt:.1f}s ({dt/max(1, ok+fail):.1f}s/cancion efectivos)")
    print(f"  CSV     : {out_csv}")
    print("=" * 60)


if __name__ == "__main__":

    main()
