# classify_arbol.py — el equivalente a classify.py, pero para los modelos de
# arbol/Random Forest entrenados en train_tree.py. NO modifica classify.py:
# solo IMPORTA su funcion validate() (y algunas utilidades) para que la
# comparacion lineal-vs-arbol use exactamente la misma metrica y el mismo
# calculo de matriz de confusion que ya usaste para comparar rms vs lra.

import sys
from pathlib import Path

import joblib
import pandas as pd

from classify import (
    OUT_DIR, MP3_CATEGORICOS_REVISADOS,
    load_categoricos, find_latest_summary, validate,
)


SCRIPT_DIR = Path(__file__).parent
OUT_MODELS_DIR = SCRIPT_DIR / "models-tree"

MODELOS = {
    "arbol_decision": OUT_MODELS_DIR / "modelo_arbol_root_A.pkl",
    "random_forest":  OUT_MODELS_DIR / "modelo_rf.pkl",
}

OUT_HEADERS = [
    "filename", "duration_s", "spotify_url",
    "cambio_energia", "margen",
    "categoria_sugerida", "categoria_humano", "revisar",
]

THR_MARGEN_ARBOL = 0.15   # analogo al THR_MARGEN de classify.py, pero para
                          # la diferencia de probabilidad top1-top2 del arbol


def classify_tree_row(row: pd.Series, modelo, features_cols: list[str], categoricos: dict) -> dict | None:
    try:
        x = [float(row[c]) for c in features_cols]
    except (KeyError, ValueError):
        return None   # fila con columnas faltantes o no numericas -- se omite, igual que en classify.py

    pred  = modelo.predict([x])[0]
    proba = modelo.predict_proba([x])[0]

    orden = sorted(zip(modelo.classes_, proba), key=lambda t: -t[1])
    top1_p = orden[0][1]
    top2_p = orden[1][1] if len(orden) > 1 else 0.0
    margen = top1_p - top2_p   # que tan clara fue la decision del arbol -- analogo al "margen" lineal

    nombre     = row["filename"].removesuffix(".mp3").lower()
    categorico = categoricos.get(nombre, {})

    return {
        "filename":           row["filename"],
        "spotify_url":        categorico.get("spotify_url", ""),
        "duration_s":         row.get("duration_s", ""),
        "cambio_energia":     "",   # no aplica al arbol -- se deja vacio por compatibilidad de columnas
        "margen":             round(margen, 3),
        "categoria_sugerida": pred,
        "categoria_humano":   categorico.get("categoria", ""),
        "revisar":            bool(margen < THR_MARGEN_ARBOL),
    }


def main():
    arg_playlist = "rock_english"

    in_csv = find_latest_summary(OUT_DIR, arg_playlist)
    if in_csv is None:
        sys.exit(f"[ERROR] No se encontro summary para {arg_playlist}. Corre extract_features.py primero.")

    df = pd.read_csv(in_csv, sep=";", encoding="utf-8", dtype=str).fillna("")
    df = df[df["error"].str.strip() == ""]
    categoricos = load_categoricos(MP3_CATEGORICOS_REVISADOS)

    resumen_pct = {}
    for nombre_modelo, path_pkl in MODELOS.items():
        if not path_pkl.exists():
            print(f"[AVISO] No existe {path_pkl} -- corre train_tree.py primero.")
            continue

        paquete = joblib.load(path_pkl)
        modelo, features_cols = paquete["modelo"], paquete["features"]

        records = [
            rec for _, row in df.iterrows()
            if (rec := classify_tree_row(row, modelo, features_cols, categoricos)) is not None
        ]
        if not records:
            print(f"[AVISO] Modelo '{nombre_modelo}': no se pudo clasificar ninguna cancion "
                  f"(revisa que las columnas {features_cols} existan en el summary).")
            continue

        out_csv = OUT_DIR / f"_classified_{nombre_modelo}_{arg_playlist}.csv"
        pd.DataFrame(records, columns=OUT_HEADERS).to_csv(out_csv, sep=";", index=False, encoding="utf-8")

        print(f"\n{'=' * 60}\n  Modelo: {nombre_modelo}\n{'=' * 60}")
        pct = validate(out_csv)   # MISMA funcion que compara rms/lra -- comparacion justa, sin duplicar logica
        resumen_pct[nombre_modelo] = pct

    print("\n" + "=" * 60)
    print("  Comparacion final: arbol/RF vs lo que ya tenias con classify.py")
    print("=" * 60)
    for k, v in resumen_pct.items():
        print(f"  {k:20s}: {v:.1f}%" if v is not None else f"  {k:20s}: sin datos")
    print("  (compara esto contra el % de promedios_rms / medianas_rms / etc. de classify.py)")


if __name__ == "__main__":
    main()
