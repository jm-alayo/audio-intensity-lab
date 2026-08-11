import sys
import os
from pathlib import Path
import time
import logging
from multiprocessing import Pool

import librosa
import librosa.beat
import librosa.effects
import librosa.feature
import librosa.onset
import numpy as np
import pandas as pd
import pyloudnorm as pyln
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.settings import PLAYLIST_BASE, CANCIONES_XLSX, FEATURES_DIR, TIME_LOG_CSV
from shared.utils import find_current_id, slope, delta_extremos, agg_media, agg_mediana, normalize, load_music_xlsx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_WORKERS = 3
N_WORKERS = os.cpu_count() if N_WORKERS == -1 else N_WORKERS

SR = 22050
TRIM_DB = 45
TARGET_LUFS = -23.0
MIN_LUFS = -70.0
SEGMENT_MIN_DURATION = 25

LRA_LO = (1.0, 0.9372)
LRA_HI = (8.0, 13.0)
LRA_BLOCK_S = 3.0
LRA_HOP_S = 1.0
LRA_ABS_GATE_LUFS = -70.0
LRA_REL_GATE_LU = 20.0

ONSET_LO = (0.5, 2.3124)
ONSET_HI = (4.0, 5.6605)
TEMPO_LO = (70.0, 71.78)
TEMPO_HI = (140.0, 136.0)
CENT_LO = (800.0, 1437.63)
CENT_HI = (3500.0, 2976.01)
OSTR_LO = (0.5, 1.0283)
OSTR_HI = (3.5, 1.8669)
FLAT_LO = (0.001, 0.0034)
FLAT_HI = (0.05, 0.0437)
ZCR_LO = (0.03, 0.0541)
ZCR_HI = (0.15, 0.1604)
DYN_LO = (0.01, 0.0305)
DYN_HI = (0.06, 0.0961)

ONSET_GATE_PCT = 60
DEFAULT_BATCH = 20

BASE_COLS = ["music_name", "duration_s", "n_segments"]

COMPONENT_COLS = [
    f"{comp}_{agg}"
    for comp in ["onset_n", "tempo_n", "cent_n", "ostr_n", "flat_n", "zcr_n", "dyn_n"]
    for agg in ("promedio", "mediana")
]

SUMMARY_COLS = BASE_COLS + [
    "score_energia_promedio", "score_energia_mediana",
    "score_energia_lra_promedio", "score_energia_lra_mediana",
    "score_ritmo_promedio", "score_ritmo_mediana",
    "pendiente_energia", "pendiente_energia_lra", "pendiente_ritmo",
    "delta_energia", "delta_energia_lra", "delta_ritmo",
    "score_noise_promedio", "score_noise_mediana",
] + COMPONENT_COLS + ["error"]

FEATURES = ["onset_n_mediana", "tempo_n_mediana", "cent_n_mediana", "ostr_n_mediana", "flat_n_mediana", "zcr_n_mediana", "dyn_n_mediana", "pendiente_energia", "delta_energia", "pendiente_ritmo", "delta_ritmo","n_segments"]

CONFIG_SECCIONES = {
    "variables generales": [
        "SR", "TRIM_DB", "TARGET_LUFS", "MIN_LUFS", "ONSET_GATE_PCT", "DEFAULT_BATCH",
    ],
    "variables de calibracion": [
        "ONSET_LO", "ONSET_HI", "TEMPO_LO", "TEMPO_HI", "CENT_LO", "CENT_HI",
        "OSTR_LO", "OSTR_HI", "FLAT_LO", "FLAT_HI", "ZCR_LO", "ZCR_HI",
        "DYN_LO", "DYN_HI",
    ],
    "variables LRA": [
        "LRA_BLOCK_S", "LRA_HOP_S", "LRA_ABS_GATE_LUFS", "LRA_REL_GATE_LU",
    ],
}

def log_audit_time_processing(music_name: str, start_time: float, end_time: float, hpss_duration: float) -> None:

    fila = pd.DataFrame([{
        "music_name":    music_name,
        "worker_id":     os.getpid(),
        "total_duration": round(end_time - start_time, 4),
        "hpss_duration": round(hpss_duration, 4),
        "margin": round((end_time - start_time) - hpss_duration, 4)
    }])

    TIME_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    existe = TIME_LOG_CSV.exists() and TIME_LOG_CSV.stat().st_size > 0

    fila.to_csv(TIME_LOG_CSV, sep=";", mode="a", index=False, header=not existe, encoding="utf-8")

def dump_config(out_dir: Path, playlist: str, segmin: int, id: int) -> Path:

    rows = [
        {
            "seccion": seccion,
            "nom_variable": nombre,
            "valor": globals()[nombre]
        } for seccion, nombres in CONFIG_SECCIONES.items()
            for nombre in nombres
    ]

    variables_dir = out_dir / "variables"
    variables_dir.mkdir(parents=True, exist_ok=True)

    out_csv = variables_dir / f"_config_{id}_{playlist}_segmin{segmin}.csv"
    pd.DataFrame(rows, columns=["seccion", "nom_variable", "valor"]).to_csv(
        out_csv, sep=";", index=False, encoding="utf-8"
    )
    return out_csv

def load_done(csv_path: Path) -> set[str]:

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False)

    mask = df["error"] == ""

    return set(df.loc[mask, "music_name"])

def fold_tempo(bpm: float, lo: float = TEMPO_LO[-1], hi: float = TEMPO_HI[-1]) -> float:

    if bpm <= 0:
        return 100.0

    while bpm >= hi:
        bpm /= 2.0

    while bpm < lo:
        bpm *= 2.0

    return bpm

def compute_lra(seg: np.ndarray, sr: int, meter: "pyln.Meter") -> float:

    block_len = int(LRA_BLOCK_S * sr)
    hop_len   = int(LRA_HOP_S * sr)

    if len(seg) < block_len:
        return 0.0

    block_loudness = []
    for start in range(0, len(seg) - block_len + 1, hop_len):
        block = seg[start:start + block_len]

        try:
            l = meter.integrated_loudness(block)

        except Exception:
            continue

        if np.isfinite(l):
            block_loudness.append(l)

    block_loudness = np.array(block_loudness)

    if len(block_loudness) < 3:
        return 0.0

    abs_gated = block_loudness[block_loudness >= LRA_ABS_GATE_LUFS]
    if len(abs_gated) < 3:
        return 0.0

    rel_threshold = float(np.mean(abs_gated)) - LRA_REL_GATE_LU
    rel_gated = abs_gated[abs_gated >= rel_threshold]
    
    if len(rel_gated) < 3:
        rel_gated = abs_gated

    p10, p95 = np.percentile(rel_gated, [10, 95])

    return float(p95 - p10)

def extract_segment(seg: np.ndarray, sr: int, b_idx: int, meter: "pyln.Meter") -> dict:

    duration_s = len(seg) / sr

    onset_strength = librosa.onset.onset_strength(y=seg, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=sr)
    onset_strength_mean = float(onset_strength.mean())

    if len(onset_frames) > 0:
        gate           = np.percentile(onset_strength, ONSET_GATE_PCT)
        strong_onsets  = onset_frames[onset_strength[onset_frames] >= gate]
        onset_density  = len(strong_onsets) / duration_s if duration_s > 0 else 0.0

    else:
        onset_density = 0.0

    logger.info(f" ## SEGMENTO {b_idx + 1} ONSET: onset_density: {round(onset_density, 2)}, onset_strength_mean: {round(onset_strength_mean, 2)}")

    raw_tempo = librosa.beat.beat_track(onset_envelope=onset_strength, sr=sr)[0]

    bpm = float(raw_tempo[0]) if hasattr(raw_tempo, "__len__") else float(raw_tempo)
    bpm_folded = fold_tempo(bpm if bpm > 0 else 120.0)

    logger.info(f" ## SEGMENTO {b_idx + 1} TEMPO: bpm_folded: {round(bpm_folded)}")

    centroid = float(librosa.feature.spectral_centroid(y=seg, sr=sr).mean())

    flatness = float(librosa.feature.spectral_flatness(y=seg).mean())

    zcr = float(librosa.feature.zero_crossing_rate(y=seg).mean())

    logger.info(f" ## SEGMENTO {b_idx + 1} TIMBRE: centroid: {round(centroid, 2)} Hz, flatness: {round(flatness*100, 2)}%, zcr: {round(zcr*100, 2)}%")

    rms_frames = librosa.feature.rms(y=seg)[0]
    p90, p10   = np.percentile(rms_frames, [90, 10])

    dyn_spread = float(p90 - p10)

    lra   = compute_lra(seg, sr, meter)

    onset_n = normalize(onset_density, ONSET_LO[-1], ONSET_HI[-1], kind="linear")
    tempo_n = normalize(bpm_folded, TEMPO_LO[-1], TEMPO_HI[-1], kind="linear")
    cent_n  = normalize(centroid, CENT_LO[-1], CENT_HI[-1], kind="linear")
    ostr_n  = normalize(onset_strength_mean, OSTR_LO[-1], OSTR_HI[-1], kind="linear")
    flat_n  = normalize(flatness, FLAT_LO[-1], FLAT_HI[-1], kind="linear")
    zcr_n   = normalize(zcr, ZCR_LO[-1], ZCR_HI[-1], kind="linear")

    dyn_n   = normalize(dyn_spread, DYN_LO[-1], DYN_HI[-1], kind="sqrt")
    dyn_n_lra = normalize(lra, LRA_LO[-1], LRA_HI[-1], kind="linear")

    noise_n = 0.5 * flat_n + 0.5 * zcr_n

    score_energia = 0.35 * noise_n + 0.35 * ostr_n + 0.20 * cent_n + 0.10 * (1.0 - dyn_n)
    score_energia_lra = 0.35 * noise_n + 0.35 * ostr_n + 0.20 * cent_n + 0.10 * (1.0 - dyn_n_lra)
    score_ritmo   = 0.45 * onset_n + 0.20 * tempo_n

    logger.info(f" ## SEGMENTO {b_idx + 1} SCORES: score_energia: {round(score_energia, 4)}, score_ritmo: {round(score_ritmo, 4)}")
    logger.info(f" ## (ADICIONAL) SEGMENTO {b_idx + 1} SCORES: dyn_n: {round(dyn_spread, 2)}, lra: {round(lra, 2)}, dyn_n_normalized: {round(dyn_n, 2)}, dyn_n_lra: {round(dyn_n_lra, 2)}")

    logger.info(f" # SEGMENTO {b_idx + 1} SCORES: score_energia: {round(score_energia, 4)}, score_energia_lra: {round(score_energia_lra, 4)}, score_ritmo: {round(score_ritmo, 4)}")

    return {
        "onset_density":       round(onset_density, 4),
        "onset_strength_mean": round(onset_strength_mean, 4),
        "bpm":                 round(bpm, 2),
        "bpm_folded":          round(bpm_folded, 2),
        "centroid_hz":         round(centroid, 1),
        "flatness":            round(flatness, 6),
        "zcr":                 round(zcr, 5),
        "dyn_spread":          round(dyn_spread, 5),
        "lra":                 round(lra, 4),
        "score_energia":       round(score_energia, 4),
        "score_energia_lra":   round(score_energia_lra, 4),
        "score_ritmo":         round(score_ritmo, 4),
        "noise_n":             round(noise_n, 4),
        "onset_n":             round(onset_n, 4), 
        "tempo_n":             round(tempo_n, 4),
        "cent_n":              round(cent_n, 4),   
        "ostr_n":              round(ostr_n, 4),
        "flat_n":              round(flat_n, 4),   
        "zcr_n":               round(zcr_n, 4),
        "dyn_n":               round(dyn_n, 4)
    }

def process_file(mp3_path: Path) -> dict:

    full_time_start = time.time()

    logger.info(f" ## PROCESANDO CANCIÓN: {mp3_path.name}")

    start_time = time.time()
    y, sr = librosa.load(str(mp3_path), sr=SR, mono=True)

    y_original = y
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

    min_seg_duration = 25
    if len(y) < sr * min_seg_duration:
        raise ValueError(f"Audio demasiado corto tras recortar silencio (<{min_seg_duration} s)")

    end_time = time.time()
    logger.info(f" ## COMPARACIÓN DE DURACION: duration_original: {round(len(y_original) / sr, 2)} s, duration_recortada: {round(len(y) / sr, 2)} s")

    start_time = time.time()
    meter = pyln.Meter(sr)
    lufs  = meter.integrated_loudness(y)

    if lufs > MIN_LUFS:
        y = pyln.normalize.loudness(y, lufs, TARGET_LUFS)

        y = np.clip(y, -1.0, 1.0)

    duration_s = len(y) / sr
    n_segments = max(1, int(duration_s // SEGMENT_MIN_DURATION))
    seg_len    = len(y) // n_segments

    if seg_len < int(sr * 1):
        raise ValueError(f"Segmentos demasiado cortos (<1000 ms) con {n_segments} segmentos")

    end_time = time.time()
    logger.info(f"Tiempo de normalización LUFS: {end_time - start_time} segundos")

    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = len(y) if i == n_segments - 1 else (i + 1) * seg_len
        segments.append(y[start:end])

    meter = pyln.Meter(sr)
    feats = [extract_segment(s, sr, b_idx=b_idx, meter=meter) for b_idx, s in enumerate(segments)]

    energia_scores = [f["score_energia"] for f in feats]
    energia_lra_scores = [f["score_energia_lra"] for f in feats]
    ritmo_scores   = [f["score_ritmo"]   for f in feats]
    noise_scores   = [f["noise_n"]      for f in feats]

    summary = {
        "music_name":   mp3_path.stem,
        "duration_s":   round(duration_s, 2),
        "n_segments":   n_segments,
    }

    for comp in ["onset_n","tempo_n","cent_n","ostr_n","flat_n","zcr_n","dyn_n"]:
        vals = [f[comp] for f in feats]
        summary[f"{comp}_promedio"] = agg_media(vals)
        summary[f"{comp}_mediana"]  = agg_mediana(vals)

    summary["score_energia_promedio"] = agg_media(energia_scores)
    summary["score_energia_mediana"]  = agg_mediana(energia_scores)
    summary["score_ritmo_promedio"]   = agg_media(ritmo_scores)
    summary["score_ritmo_mediana"]    = agg_mediana(ritmo_scores)
    summary["score_energia_lra_promedio"] = agg_media(energia_lra_scores)
    summary["score_energia_lra_mediana"]  = agg_mediana(energia_lra_scores)
    summary["score_noise_promedio"] = agg_media(noise_scores)
    summary["score_noise_mediana"]  = agg_mediana(noise_scores)
    summary["pendiente_energia"] = round(slope(energia_scores), 4)
    summary["pendiente_energia_lra"] = round(slope(energia_lra_scores), 4)
    summary["pendiente_ritmo"]   = round(slope(ritmo_scores), 4)
    summary["delta_energia"] = round(delta_extremos(energia_scores), 4)
    summary["delta_energia_lra"] = round(delta_extremos(energia_lra_scores), 4)
    summary["delta_ritmo"]   = round(delta_extremos(ritmo_scores), 4)
    summary["error"] = ""

    features = pd.DataFrame({
        f"seg_{i}": {
            "onset_density":       feat["onset_density"],
            "onset_strength_mean": feat["onset_strength_mean"],
            "bpm":                 feat["bpm"],
            "bpm_folded":          feat["bpm_folded"],
            "centroid_hz":         feat["centroid_hz"],
            "flatness":            feat["flatness"],
            "zcr":                 feat["zcr"],
            "dyn_spread":          feat["dyn_spread"],
            "lra":                 feat["lra"],
            "score_energia":       feat["score_energia"],
            "score_energia_lra":   feat["score_energia_lra"],
            "score_ritmo":         feat["score_ritmo"],
            "noise_n":             feat["noise_n"]
        }
        for i, feat in enumerate(feats, 1)
    })

    features.index.name = "feature"

    features.insert(0, "music_name", mp3_path.stem)

    end_fulltime = time.time()
    logger.info(f"Tiempo de procesamiento de la canción: {end_fulltime - full_time_start} segundos")

    log_audit_time_processing(mp3_path.name, full_time_start, end_fulltime, 0.0)

    return features, summary

def _init_worker():

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    logger.setLevel(logging.WARNING)

def _worker(mp3_path_str: str):

    mp3_path = Path(mp3_path_str)

    try:

        features, summary = process_file(mp3_path)

        return features, summary

    except Exception as exc:

        summary = {
            h: "" for h in SUMMARY_COLS
        }

        summary.update({
            "filename": mp3_path.stem,
            "error":    str(exc)[:300],
        })

        return None, summary

def run_parallel(pending: list[Path], out_csv: Path, out_features_csv: Path):

    total_ok = total_fail = 0

    file_exists = out_csv.exists()
    features_file_exists = out_features_csv.exists()

    with open(out_csv, "a", newline="", encoding="utf-8") as f, \
         open(out_features_csv, "a", newline="", encoding="utf-8") as f_feat, \
         Pool(processes=N_WORKERS, initializer=_init_worker, maxtasksperchild=20) as pool:

        header_written = file_exists
        features_header_written = features_file_exists

        job_args = [str(p) for p in pending]
        jobs = pool.imap_unordered(_worker, job_args, chunksize=1)

        for features, summary in tqdm(jobs, total=len(pending), desc="Extrayendo", unit="mp3"):
            if summary.get("error"):
                total_fail += 1
                tqdm.write(f"  [FAIL] {summary['music_name']}: {summary['error']}")

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

    if not folder.exists():
        sys.exit(f"[ERROR] Carpeta no encontrada: {folder}")

    musics_names = load_music_xlsx(CANCIONES_XLSX)
    mp3_files = [
        p.stem for p in sorted(folder.glob("*.mp3"))
            if p.stem in musics_names["music_name"].tolist()
    ]

    logger.info(f"Se encontraron {len(mp3_files)} canciones: playlist {playlist}")

    if not mp3_files:
        sys.exit(f"[ERROR] No se encontraron MP3s en: {folder}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    id = find_current_id(FEATURES_DIR, playlist, SEGMENT_MIN_DURATION)

    id += 1 if id > 0 else 0

    summary_csv = FEATURES_DIR / f"_summary_{id}_{playlist}_segmin{SEGMENT_MIN_DURATION}.csv"
    features_csv = FEATURES_DIR / f"_features_{id}_{playlist}_segmin{SEGMENT_MIN_DURATION}.csv"

    done = load_done(summary_csv)
    pending = [
        folder / f"{f}.mp3" for f in mp3_files
            if f not in done
    ]

    logger.debug("=" * 60)
    logger.debug(f"  Playlist   : {folder.name}")
    logger.debug(f"  Total MP3s : {len(mp3_files)}")
    logger.debug(f"  Ya en CSV  : {len(done)}")
    logger.debug(f"  A procesar : {len(pending)}")
    logger.debug("=" * 60)

    if not pending:
        logger.info("\nTodo ya está procesado. Ejecuta classify.py para clasificar.")

        return

    logger.info(f"Procesando {len(pending)} canciones con {N_WORKERS} workers...")

    t0 = time.time()
    total_ok, total_fail = run_parallel(pending, summary_csv, features_csv)
    dt = time.time() - t0

    logger.info("=" * 60)
    logger.info(f"  Resueltos : {total_ok}")
    logger.info(f"  Errores   : {total_fail}")
    logger.info(f"  Tiempo    : {dt:.1f}s ({dt/max(1, total_ok+total_fail):.1f}s/cancion efectivos)")
    logger.info(f"  ID del registro: {id}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
