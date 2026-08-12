from pathlib import Path

SCRIPT_DIR    = Path(__file__).parent
BENCHMARK_DIR = SCRIPT_DIR.parent
PROJECT_ROOT  = BENCHMARK_DIR.parent

PLAYLIST_BASE = PROJECT_ROOT / "async-music-fetch-engine" / "music-extract" / "data" / "music-downloaded"
FEATURES_DIR  = BENCHMARK_DIR / "data" / "features"
TRACKS_XLSX   = BENCHMARK_DIR / "data" / "audio" / "_canciones_rock_english.xlsx"

ML_ARTIFACTS_DIR = BENCHMARK_DIR / "ml" / "artifacts"
ML_RUNS_DIR       = BENCHMARK_DIR / "ml" / "runs"

TIME_LOG_CSV = FEATURES_DIR / "time" / "registro_tiempo.csv"
