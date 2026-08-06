import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # music-tagger-benchmark/ -- para poder importar shared/
from shared.utils import percentile

FEATURES_CRUDAS = [
    "onset_density", "onset_strength_mean", "bpm_folded",
    "centroid_hz", "flatness", "zcr", "perc_ratio",
    "dyn_spread", "lra",
]


def cargar_valores_por_feature(csv_path: Path) -> dict[str, list[float]]:
    valores = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # descarta cabecera — no confiar en su ancho
        for row in reader:
            if len(row) < 3:
                continue
            feature_name = row[0]
            if feature_name not in FEATURES_CRUDAS:
                continue
            for v in row[2:]:               # row[1] es filename, se ignora
                v = v.strip()
                if v == "":
                    continue
                try:
                    valores[feature_name].append(float(v))
                except ValueError:
                    pass
    return valores


def calibrar(csv_path: Path, p_lo: float = 5, p_hi: float = 95) -> dict[str, tuple]:
    datos = cargar_valores_por_feature(csv_path)
    rangos = {}
    for feat in FEATURES_CRUDAS:
        vals = datos.get(feat, [])
        if len(vals) < 10:
            print(f"  [AVISO] {feat}: solo {len(vals)} valores — muy pocos para "
                  f"un percentil confiable, esperar a correr sobre las 119 completas.")
        lo, hi = percentile(vals, p_lo), percentile(vals, p_hi)
        rangos[feat] = (round(lo, 6), round(hi, 6), len(vals))
    return rangos


# Constantes ACTUALES en extract_features.py, para comparar lado a lado.
# Actualiza este diccionario si tus constantes reales son distintas.
CONSTANTES_ACTUALES = {
    "onset_density":       (0.5, 4.0),    # ONSET_LO, ONSET_HI
    "onset_strength_mean": (0.5, 3.5),    # OSTR_LO, OSTR_HI
    "bpm_folded":          (70.0, 140.0), # TEMPO_LO, TEMPO_HI
    "centroid_hz":         (800.0, 3500.0),  # CENT_LO, CENT_HI
    "flatness":            (0.001, 0.05),    # FLAT_LO, FLAT_HI
    "zcr":                 (0.03, 0.15),     # ZCR_LO, ZCR_HI
    "perc_ratio":          (0.10, 0.60),     # PERC_LO, PERC_HI
    "dyn_spread":          (0.01, 0.06),     # DYN_LO, DYN_HI
    "lra":                 (1.0, 8.0),       # LRA_LO, LRA_HI
}


if __name__ == "__main__":

    SCRIPT_DIR   = Path(__file__).parent
    features_csv = SCRIPT_DIR / "out" / "_features_7_rock_english_segmin25.csv"

    if features_csv is None:
        raise SystemExit(f"[ERROR] No se encontró ningún _features_*_rock_english_segmin*.csv en {SCRIPT_DIR / 'out'}")

    rangos = calibrar(features_csv)

    print(f"\n{'Feature':22s} {'LO actual':>10s} {'HI actual':>10s} | {'LO real (p5)':>13s} {'HI real (p95)':>14s}  n_segmentos")
    for feat, (lo_real, hi_real, n) in rangos.items():
        lo_act, hi_act = CONSTANTES_ACTUALES.get(feat, (float("nan"), float("nan")))
        alerta = "  <-- REVISAR, difiere bastante" if (
            abs(lo_real - lo_act) / max(abs(lo_act), 1e-9) > 0.3 or
            abs(hi_real - hi_act) / max(abs(hi_act), 1e-9) > 0.3
        ) else ""
        print(f"{feat:22s} {lo_act:10.4f} {hi_act:10.4f} | {lo_real:13.4f} {hi_real:14.4f}  n={n}{alerta}")