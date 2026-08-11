import sys
from pathlib import Path

import pandas as pd

from classify import OUT_DIR, CATEGORIAS, load_categoricos
from shared.utils import find_latest_summary, filename_key
from shared.ml_func import MLFunc

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

    rank = MLFunc.rank_probabilities(modelo, x)   # predict_proba + top1/top2/margen

    nombre     = filename_key(row["filename"])
    categorico = categoricos.get(nombre, {})

    return {
        "filename": row["filename"],
        "spotify_url": categorico.get("spotify_url", ""),
        "duration_s": row.get("duration_s", ""),
        "cambio_energia": "",
        "margen": rank["margen"],
        "categoria_sugerida": rank["top1_categoria"],
        "categoria_humano": categorico.get("categoria", ""),
        "revisar": bool(rank["margen"] < THR_MARGEN_ARBOL),
    }


def main():
    arg_playlist = "rock_english"

    in_csv = find_latest_summary(OUT_DIR, arg_playlist)
    if in_csv is None:
        sys.exit(f"[ERROR] No se encontro summary para {arg_playlist}. Corre extract_features.py primero.")

    df = pd.read_csv(in_csv, sep=";", encoding="utf-8", dtype=str).fillna("")
    df = df[df["error"].str.strip() == ""]
    categoricos = load_categoricos()

    resumen_pct = {}
    for nombre_modelo, path_pkl in MODELOS.items():
        if not path_pkl.exists():
            print(f"[AVISO] No existe {path_pkl} -- corre train_tree.py primero.")
            continue

        modelo, features_cols = MLFunc.load_model(path_pkl)

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
        pct = MLFunc.validate_predictions(out_csv, CATEGORIAS)   # MISMA funcion que compara rms/lra -- comparacion justa, sin duplicar logica
        resumen_pct[nombre_modelo] = pct

    print("\n" + "=" * 60)
    print("  Comparacion final: arbol/RF vs lo que ya tenias con classify.py")
    print("=" * 60)
    for k, v in resumen_pct.items():
        print(f"  {k:20s}: {v:.1f}%" if v is not None else f"  {k:20s}: sin datos")
    print("  (compara esto contra el % de promedios_rms / medianas_rms / etc. de classify.py)")


if __name__ == "__main__":
    main()
