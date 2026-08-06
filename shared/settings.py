from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent          # music-tagger-benchmark/shared/
BENCHMARK_DIR = SCRIPT_DIR.parent              # music-tagger-benchmark/
PROJECT_ROOT  = BENCHMARK_DIR.parent
DSP_DIR       = BENCHMARK_DIR / "dsp"

PLAYLIST_BASE = PROJECT_ROOT / "async-music-fetch-engine" / "music-extract" / "data" / "music-downloaded"
FEATURES_DIR = BENCHMARK_DIR / "data" / "features"
# mp3-categoricos-revisados.csv quedo obsoleto -- la etiqueta humana ahora sale
# de la columna "categorico" del xlsx via extract_features.load_music().
CANCIONES_XLSX  = BENCHMARK_DIR / "data" / "audio" / "_canciones_rock_english.xlsx"

# Estos dos viven junto a los scripts de dsp/ (no junto a shared/) -- ahi es
# donde train_tree.py/extract_features.py ya vienen leyendo/escribiendo.
OUT_MODELS_DIR = DSP_DIR / "models-tree"
OUT_FILES_DIR = DSP_DIR / "out"

TIME_LOG_CSV = FEATURES_DIR / "time" / "registro_tiempo.csv"