import sys
import os
import re
from pathlib import Path
import time
import librosa
import librosa.beat
import librosa.effects
import librosa.feature
import librosa.onset
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import logging
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))   # music-tagger-benchmark/ -- para poder importar shared/
from shared.settings import PLAYLIST_BASE, CANCIONES_XLSX, FEATURES_DIR, TIME_LOG_CSV
from shared.utils import find_current_id, slope, delta_extremos, agg_media, agg_mediana, normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANCIONES_SHEET = "data"
SIN_CATEGORIA   = "sin_categoria"

def log_tiempo(music_name: str, start_time: float, end_time: float, hpss_duration: float) -> None:

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

SR = 22050 # tasa de muestreo (Hz) por segundo
TRIM_DB = 45       # top_db para recortar silencio
TARGET_LUFS = -23.0    # normalización EBU R128
MIN_LUFS = -70.0    # límite inferior para aplicar normalización
SEGMENT_MIN_DURATION = 25 # segundos

# ── Rangos de normalización por feature (clip a [0, 1]) ────────────────────────
# Calibrados con percentiles p5/p95 del dataset real; recalcular en calibrate.py
# cuando se procesen más playlists.
#
# NOTA: el RMS medio deja de usarse como señal de intensidad — tras normalizar
# a -23 LUFS, converge en todas las canciones (86.7% de los segmentos saturaban
# norm_rms=1.0, el "modulador maestro" era una constante). Se reemplaza por
# features que sí sobreviven a la normalización LUFS: fuerza de ataque, brillo
# espectral, ratio percusivo (HPSS) y dinámica intra-segmento.

ONSET_LO, ONSET_HI = 2.3124, 5.6605  # onsets FUERTES/s (baja tras filtrar por gate)
TEMPO_LO, TEMPO_HI = 71.78, 136.0    # BPM tras plegado de octava
CENT_LO,  CENT_HI  = 1437.63, 2976.01 # Hz, centroide espectral
OSTR_LO,  OSTR_HI  = 1.0283, 1.8669  # fuerza media de onset (ONSET_STRENGTH_MEAN)
FLAT_LO,  FLAT_HI  = 0.0034, 0.0437  # spectral flatness — distorsión/ruido vs. tonal
ZCR_LO,   ZCR_HI   = 0.0541, 0.1604  # zero-crossing rate — complementa a flatness
PERC_LO,  PERC_HI  = 0.0589, 0.4079  # ratio de energía percusiva
DYN_LO,   DYN_HI   = 0.0305, 0.0961  # spread p90-p10 del RMS por frame

ONSET_GATE_PCT = 60   # percentil de la envolvente de onset para contar un ataque como "fuerte"
DEFAULT_BATCH = 20

BASE_COLS    = ["filename", "album", "duration_s", "n_segments"]
COMPONENT_COLS = [
    f"{comp}_{agg}"
    for comp in ["onset_n", "tempo_n", "cent_n", "ostr_n", "flat_n", "zcr_n", "perc_n", "dyn_n"]
    for agg in ("promedio", "mediana")
]
SUMMARY_COLS = BASE_COLS + [
    "score_energia_promedio", "score_energia_mediana",
    "score_energia_lra_promedio", "score_energia_lra_mediana",
    "score_ritmo_promedio", "score_ritmo_mediana",
    "pendiente_energia", "pendiente_energia_lra", "pendiente_ritmo",
    "delta_energia", "delta_energia_lra", "delta_ritmo",
    "score_noise_promedio", "score_noise_mediana",
] + COMPONENT_COLS + [
    "error"
]

FEATURES = ["onset_n_mediana", "tempo_n_mediana", "cent_n_mediana", "ostr_n_mediana", "flat_n_mediana", "zcr_n_mediana", "dyn_n_mediana", "pendiente_energia", "delta_energia", "pendiente_ritmo", "delta_ritmo","n_segments"]

# Snapshot de las variables con las que se generó un _summary_{id}/_features_{id}
# — quedan calibrándose seguido, así que sin esto un CSV viejo no es
# interpretable (¿con qué LO/HI se normalizó esa corrida?).
CONFIG_SECCIONES = {
    "variables generales": [
        "SR", "TRIM_DB", "TARGET_LUFS", "MIN_LUFS", "ONSET_GATE_PCT", "DEFAULT_BATCH",
    ],
    "variables de calibracion": [
        "ONSET_LO", "ONSET_HI", "TEMPO_LO", "TEMPO_HI", "CENT_LO", "CENT_HI",
        "OSTR_LO", "OSTR_HI", "FLAT_LO", "FLAT_HI", "ZCR_LO", "ZCR_HI",
        "PERC_LO", "PERC_HI", "DYN_LO", "DYN_HI",
    ],
    "variables LRA": [
        "LRA_BLOCK_S", "LRA_HOP_S", "LRA_ABS_GATE_LUFS", "LRA_REL_GATE_LU",
    ],
}

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

def load_music(xlsx_path: Path, sheet: str = CANCIONES_SHEET) -> list[str]:

    df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)

    df = df[
        (df["categorico"].str.strip() != SIN_CATEGORIA) &
        (df["album"].str.strip() != "sin_categoria")
    ]

    return df

def load_albums(xlsx_path: Path, sheet: str = CANCIONES_SHEET) -> dict[str, str]:

    df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
    df = df[df["categorico"].str.strip() != SIN_CATEGORIA]

    return {f"{n}.mp3": (a or "") for n, a in zip(df["music_name"], df["album"])}

def load_categorized(xlsx_path: Path, sheet: str = CANCIONES_SHEET) -> pd.DataFrame:

    df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)

    return df[df["categorico"].str.strip() != SIN_CATEGORIA]

def load_done(csv_path: Path) -> set[str]:

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False)

    mask = df["error"] == ""

    return set(df.loc[mask, "filename"])

def fold_tempo(bpm: float, lo: float = TEMPO_LO, hi: float = TEMPO_HI) -> float:

    if bpm <= 0:
        return 100.0

    while bpm >= hi:
        bpm /= 2.0

    while bpm < lo:
        bpm *= 2.0

    return bpm

LRA_LO, LRA_HI = 0.9372, 13.0     # en LU (unidades de loudness), no en amplitud — p5/p95 real, n=1015

LRA_BLOCK_S       = 3.0       # ventana "short-term" del estándar EBU R128
LRA_HOP_S         = 1.0       # solapamiento entre ventanas
LRA_ABS_GATE_LUFS = -70.0     # gate absoluto: descarta silencio/casi-silencio
LRA_REL_GATE_LU   = 20.0      # gate relativo: descarta bloques muy por debajo del

def compute_lra(seg: np.ndarray, sr: int, meter: "pyln.Meter") -> float:

    block_len = int(LRA_BLOCK_S * sr)
    hop_len   = int(LRA_HOP_S * sr)

    if len(seg) < block_len:
        return 0.0  # segmento muy corto para medir dinámica de forma confiable

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
        return 0.0  # insuficientes bloques para un percentil confiable

    # Gate absoluto: fuera los bloques por debajo de -70 LUFS (silencio real)
    abs_gated = block_loudness[block_loudness >= LRA_ABS_GATE_LUFS]
    if len(abs_gated) < 3:
        return 0.0

    # Gate relativo: fuera los bloques 20 LU por debajo del promedio ya gateado
    rel_threshold = float(np.mean(abs_gated)) - LRA_REL_GATE_LU
    rel_gated = abs_gated[abs_gated >= rel_threshold]
    if len(rel_gated) < 3:
        rel_gated = abs_gated  # fallback si el gate relativo deja muy poco

    p10, p95 = np.percentile(rel_gated, [10, 95])

    return float(p95 - p10)

def extract_segment(seg: np.ndarray, sr: int, b_idx: int, meter: "pyln.Meter") -> dict:

    ### EVALUAR TIEMPOS | FASE DE EXTRACCION DE FEATURES (UTILIZACION DE SEGMENTOS)
    duration_s = len(seg) / sr

    ## SUBFASE 1: ONSET — densidad FILTRADA por fuerza (solo ataques que superan
    # el gate) + fuerza media. Un onset_density sin filtrar cuenta por igual un
    # fingerpicking suave (Dust in the Wind) que un golpe de batería.
    # la envolvente (oenv) se reutiliza en tempo, evita recalcularla dos veces

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

    ## SUBFASE 2: TEMPO — con plegado de octava (86 vs 172 son la misma canción)
    
    raw_tempo = librosa.beat.beat_track(onset_envelope=onset_strength, sr=sr)[0]
    
    # beat_track puede devolver array en librosa>=0.10
    bpm = float(raw_tempo[0]) if hasattr(raw_tempo, "__len__") else float(raw_tempo)
    bpm_folded = fold_tempo(bpm if bpm > 0 else 120.0)

    logger.info(f" ## SEGMENTO {b_idx + 1} TEMPO: bpm_folded: {round(bpm_folded)}")

    ## SUBFASE 3: TIMBRE — brillo (centroide) + ruidosidad (flatness, ZCR)

    # retorna frecuencia en Hz (x < 1200 hz -> grave, x > 1200 hz && x < 3000 hz -> medio, x > 3000 hz -> agudo)
    centroid = float(librosa.feature.spectral_centroid(y=seg, sr=sr).mean()) # promedio resumen de la energia (frecuencias) en el espectro

    # retorna un valor entre 0.0 y 1.0 (x < 0.01 = sonido armonico, x > 0.01 && x < 0.10 = mezcla musical estandar, x > 0.10 = ruido estático)
    flatness = float(librosa.feature.spectral_flatness(y=seg).mean()) # promedio que representa la cantidad de ruido en el espectro

    # retorna un valor entre 0.0 y 1.0 (x < 0.05 = sonido limpio, x > 0.05 && x < 0.15 = voces de canto y/o musica dinámica, x > 0.10 = sonido áspero y agresivo)
    zcr = float(librosa.feature.zero_crossing_rate(y=seg).mean()) # promedio que representa la rugosidad del sonido

    logger.info(f" ## SEGMENTO {b_idx + 1} TIMBRE: centroid: {round(centroid, 2)} Hz, flatness: {round(flatness*100, 2)}%, zcr: {round(zcr*100, 2)}%")

    ## SUBFASE 5.1 (PERCENTILES MANUALES): DINAMICA INTRA-SEGMENTO — reemplaza al RMS promedio (muerto tras LUFS)

    rms_frames = librosa.feature.rms(y=seg)[0] # Devuelve un array de valores de presion sonora en cada frame (más constante, más pesado)
    p90, p10   = np.percentile(rms_frames, [90, 10])

    # retorna un valor entre 0.0 y 1.0 (x < 0.03 = audio con intensidad constante invariable, x > 0.03 && x < 0.12 = dinámica estandar comercial, x > 0.12 = seccion montañosa con muchas subidas y bajadas)
    dyn_spread = float(p90 - p10)

    ## SUBFASE 5.2 (PERCENTILES MANUALES): DINAMICA INTRA-SEGMENTO — reemplaza al RMS promedio (muerto tras LUFS)
    
    lra   = compute_lra(seg, sr, meter) 

    ## SUBFASE 6: NORMALIZACION Y SCORES DE DOS EJES (energía / ritmo)

    onset_n = normalize(onset_density, ONSET_LO, ONSET_HI, kind="linear")
    tempo_n = normalize(bpm_folded, TEMPO_LO, TEMPO_HI, kind="linear")
    cent_n  = normalize(centroid, CENT_LO, CENT_HI, kind="linear")
    ostr_n  = normalize(onset_strength_mean, OSTR_LO, OSTR_HI, kind="linear")
    flat_n  = normalize(flatness, FLAT_LO, FLAT_HI, kind="linear")
    zcr_n   = normalize(zcr, ZCR_LO, ZCR_HI, kind="linear")

    dyn_n   = normalize(dyn_spread, DYN_LO, DYN_HI, kind="sqrt")
    dyn_n_lra = normalize(lra, LRA_LO, LRA_HI, kind="linear")
    

    # Ruidosidad combinada: flatness y zcr miden lo mismo desde dos ángulos,
    # promediarlas la hace más robusta que cualquiera sola.
    noise_n = 0.5 * flat_n + 0.5 * zcr_n

    # Eje ENERGÍA (suave <-> fuerte/agresiva): ruidosidad/distorsión + fuerza de
    # ataque + brillo. (1-dyn) baja a 0.10: con peso alto castigaba canciones
    # agresivas con estructura verso-quieto/coro-fuerte (Alien Blues).
    score_energia = 0.35 * noise_n + 0.35 * ostr_n + 0.20 * cent_n + 0.10 * (1.0 - dyn_n)
    score_energia_lra = 0.35 * noise_n + 0.35 * ostr_n + 0.20 * cent_n + 0.10 * (1.0 - dyn_n_lra)
    # Eje RITMO (contemplativa <-> rítmica): onsets fuertes + percusividad + tempo
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
        "onset_n":             round(onset_n, 4), "tempo_n": round(tempo_n, 4),
        "cent_n":              round(cent_n, 4),   "ostr_n": round(ostr_n, 4),
        "flat_n":              round(flat_n, 4),   "zcr_n": round(zcr_n, 4),
        "dyn_n": round(dyn_n, 4),
    }

def process_file(mp3_path: Path, album: str = "") -> dict:

    ### EVALUAR TIEMPOS | FASE INCIAL DE PROCESAMIENTO
    # cargar el archivo de audio, devuelve la señal y la tasa de muestreo
    full_time_start = time.time()

    logger.info(f" ## PROCESANDO CANCIÓN: {mp3_path.name}")

    start_time = time.time()
    y, sr = librosa.load(str(mp3_path), sr=SR, mono=True)

    # recortar silencio, devuelve la señal y el tiempo de silencio
    y_original = y
    y, _ = librosa.effects.trim(y, top_db=TRIM_DB)

    min_seg_duration = 25
    if len(y) < sr * min_seg_duration:
        raise ValueError(f"Audio demasiado corto tras recortar silencio (<{min_seg_duration} s)")

    end_time = time.time()
    logger.info(f" ## COMPARACIÓN DE DURACION: duration_original: {round(len(y_original) / sr, 2)} s, duration_recortada: {round(len(y) / sr, 2)} s")

    ### EVALUAR TIEMPOS | FASE DE NORMALIZACION
    # Normalización LUFS — skip si la señal es demasiado silenciosa
    start_time = time.time()
    meter = pyln.Meter(sr)
    lufs  = meter.integrated_loudness(y)

    if lufs > MIN_LUFS:
        y = pyln.normalize.loudness(y, lufs, TARGET_LUFS)

        # Clip por seguridad tras normalización (distorciona el sonido pero no afecta a la clasificacion)
        y = np.clip(y, -1.0, 1.0)

    duration_s = len(y) / sr
    n_segments = max(1, int(duration_s // SEGMENT_MIN_DURATION))
    seg_len    = len(y) // n_segments

    # El SR equivale a Hertz (Hz) y representa la cantidad de muestras por segundo.
    if seg_len < int(sr * 1):
        raise ValueError(f"Segmentos demasiado cortos (<1000 ms) con {n_segments} segmentos")

    end_time = time.time()
    logger.info(f"Tiempo de normalización LUFS: {end_time - start_time} segundos")

    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = len(y) if i == n_segments - 1 else (i + 1) * seg_len
        segments.append(y[start:end])

    ### EVALUAR TIEMPOS | FASE DE EXTRACCION DE FEATURES (UTILIZACION DE SEGMENTOS)
    meter = pyln.Meter(sr)
    feats = [extract_segment(s, sr, b_idx=b_idx, meter=meter) for b_idx, s in enumerate(segments)]

    energia_scores = [f["score_energia"] for f in feats]
    energia_lra_scores = [f["score_energia_lra"] for f in feats]
    ritmo_scores   = [f["score_ritmo"]   for f in feats]
    noise_scores   = [f["noise_n"]      for f in feats]

    summary = {
        "filename":     mp3_path.name,
        "album":        album,
        "duration_s":   round(duration_s, 2),
        "n_segments":   n_segments,
    }

    for comp in ["onset_n","tempo_n","cent_n","ostr_n","flat_n","zcr_n","perc_n","dyn_n"]:
        vals = [f[comp] for f in feats]
        summary[f"{comp}_promedio"] = agg_media(vals)
        summary[f"{comp}_mediana"]  = agg_mediana(vals)

    # reemplazamos por agg_media
    summary["score_energia_promedio"] = agg_media(energia_scores)
    summary["score_energia_mediana"]  = agg_mediana(energia_scores)
    summary["score_ritmo_promedio"]   = agg_media(ritmo_scores)
    summary["score_ritmo_mediana"]    = agg_mediana(ritmo_scores)
    # LRA
    summary["score_energia_lra_promedio"] = agg_media(energia_lra_scores)
    summary["score_energia_lra_mediana"]  = agg_mediana(energia_lra_scores)
    # Noise: Ruidosidad
    summary["score_noise_promedio"] = agg_media(noise_scores)
    summary["score_noise_mediana"]  = agg_mediana(noise_scores)
    # pendiente: regresión lineal simple (tendencia sostenida), un eje por separado
    summary["pendiente_energia"] = round(slope(energia_scores), 4)
    summary["pendiente_energia_lra"] = round(slope(energia_lra_scores), 4)
    summary["pendiente_ritmo"]   = round(slope(ritmo_scores), 4)
    # delta: mean(último 25%) - mean(primer 25%), detecta un clímax final que
    # la pendiente diluye cuando hay muchos segmentos (ej. Exit Music)
    summary["delta_energia"] = round(delta_extremos(energia_scores), 4)
    summary["delta_energia_lra"] = round(delta_extremos(energia_lra_scores), 4)
    summary["delta_ritmo"]   = round(delta_extremos(ritmo_scores), 4)
    summary["error"] = ""

    # filas: features crudas + scores de ambos ejes — columnas: filename + seg_1, seg_2, ...
    features = pd.DataFrame({
        f"seg_{i}": {
            "onset_density":       feat["onset_density"],
            "onset_strength_mean": feat["onset_strength_mean"],
            "bpm":                 feat["bpm"],
            "bpm_folded":          feat["bpm_folded"],
            "centroid_hz":         feat["centroid_hz"],
            "flatness":            feat["flatness"],
            "zcr":                 feat["zcr"],
            "perc_ratio":          feat["perc_ratio"],
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
    
    features.insert(0, "filename", mp3_path.name)

    end_fulltime = time.time()
    logger.info(f"Tiempo de procesamiento de la canción: {end_fulltime - full_time_start} segundos")

    hpss_total = sum(f["hpss_duration"] for f in feats)
    log_tiempo(mp3_path.name, full_time_start, end_fulltime, hpss_total)

    return features, summary

def main():

    playlist = "rock_english"
    batch = 20

    folder = PLAYLIST_BASE / playlist

    if not folder.exists():
        sys.exit(f"[ERROR] Carpeta no encontrada: {folder}")

    musics_names = load_music(CANCIONES_XLSX)
    mp3_files = [
        p.stem for p in sorted(folder.glob("*.mp3"))
            if p.stem in musics_names["music_name"].tolist()
    ]
    albums = load_albums(CANCIONES_XLSX)

    logger.info(f"Se encontraron {len(mp3_files)} canciones: playlist {playlist}")

    if not mp3_files:
        sys.exit(f"[ERROR] No se encontraron MP3s en: {folder}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    id = find_current_id(FEATURES_DIR, playlist, SEGMENT_MIN_DURATION)
    summary_csv = FEATURES_DIR / f"_summary_{id+1}_{playlist}_segmin{SEGMENT_MIN_DURATION}.csv"
    features_csv = FEATURES_DIR / f"_features_{id+1}_{playlist}_segmin{SEGMENT_MIN_DURATION}.csv"
    # out_config_csv = dump_config(FEATURES_DIR, playlist, SEGMENT_MIN_DURATION, id)

    headers = SUMMARY_COLS
    done = load_done(summary_csv)
    pending = [
        f for f in mp3_files 
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

    file_exists = summary_csv.exists()
    features_file_exists = features_csv.exists()

    with open(summary_csv, "a", newline="", encoding="utf-8") as f, \
         open(features_csv, "a", newline="", encoding="utf-8") as f_feat:
        header_written = file_exists
        features_header_written = features_file_exists
        total_ok = total_fail = 0

        for b_idx, b_start in enumerate(range(0, len(pending), batch), 1):
            batch = pending[b_start : b_start + batch]

            logger.info(f"\n── Batch {b_idx} ({len(batch)} canciones) ──")

            for mp3 in tqdm(batch, desc="Extrayendo", unit="mp3", leave=True):
                features = None
                album = albums.get(mp3, "")
                try:
                    
                    features, summary = process_file(PLAYLIST_BASE / playlist / f"{mp3}.mp3", album)
                    total_ok += 1

                except Exception as exc:
                    summary = {
                        h: "" for h in headers
                    }
                    summary.update({
                        "filename": mp3,
                        "album": album,
                        "error": str(exc)[:300],
                    })
                    total_fail += 1
                    tqdm.write(f"  [FAIL] {mp3.name}: {exc}")

                pd.DataFrame([summary], columns=headers).to_csv(
                    f, sep=";", index=False, header=not header_written
                )
                f.flush()
                header_written = True

                if features is not None:
                    features.to_csv(
                        f_feat, sep=";", index=True, header=not features_header_written
                    )
                    f_feat.flush()
                    features_header_written = True

    print("\n" + "=" * 60)
    print(f"  OK      : {total_ok}")
    print(f"  Errores : {total_fail}")
    print(f"  CSV     : {summary_csv}")
    print("=" * 60)

    if total_ok:
        print("\nSiguiente paso: python calibrate.py " + playlist)


if __name__ == "__main__":
    main()
