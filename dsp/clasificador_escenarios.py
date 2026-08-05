# escenario_clasificacion.py — en vez de un solo numero de "margen", muestra
# el ESCENARIO completo por cancion: si la clasificacion es confiable, si
# podria pasar como cualquiera de DOS categorias (empate real, no error),
# o si ninguna categoria domina lo suficiente como para confiar en algo
# (candidata a "tachar"/revisar a oido).
#
# Usa predict_proba del modelo ya entrenado (train_tree.py) -- no requiere
# volver a entrenar nada.

import sys
import importlib.util
from pathlib import Path

import joblib
import pandas as pd

from classify import OUT_DIR, MP3_CATEGORICOS_REVISADOS, load_categoricos, find_latest_summary
from extract_features import PLAYLIST_BASE

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models-tree"   # train_tree.py guarda los .pkl aca, no en OUT_DIR

# clasify-laion.py vive en llm-clap/ (nombre con guion, no se puede "import"
# directo) -- lo cargamos con importlib solo para reusar MODEL_ID/SR_CLAP/
# CANDIDATOS/pipeline/torch/librosa, sin duplicar esa configuracion aca.
CLAP_SCRIPT = SCRIPT_DIR.parent / "llm-clap" / "clasify-laion.py"


def _cargar_modulo_clap():
    spec = importlib.util.spec_from_file_location("clasify_laion", CLAP_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def resolver_ambiguas_con_clap(df: pd.DataFrame, playlist: str) -> pd.DataFrame:
    df = df.copy()
    df["clap_top1"] = ""
    df["veredicto_clap"] = ""
    df["categoria_final"] = df["top1_categoria"]

    ambiguas = df[df["escenario"] == "ambigua"]
    if ambiguas.empty:
        return df

    print(f"\nResolviendo {len(ambiguas)} canciones 'ambigua' con CLAP (Route B)...")

    clap = _cargar_modulo_clap()
    clasificador = clap.pipeline(
        task="zero-shot-audio-classification", model=clap.MODEL_ID,
        device=0 if clap.torch.cuda.is_available() else -1,
    )
    etiquetas_texto = list(clap.CANDIDATOS.values())
    cat_por_texto = {v: k for k, v in clap.CANDIDATOS.items()}
    carpeta = PLAYLIST_BASE / playlist

    for idx, row in ambiguas.iterrows():
        mp3 = carpeta / f"{row['filename']}.mp3"
        if not mp3.exists():
            print(f"  [AVISO] no encontrado: {row['filename']}")
            continue

        y, _ = clap.librosa.load(str(mp3), sr=clap.SR_CLAP, mono=True)
        salida = clasificador(y, candidate_labels=etiquetas_texto)
        clap_top1 = cat_por_texto[salida[0]["label"]]

        rf_top1, rf_top2 = row["top1_categoria"], row["top2_categoria"]

        if clap_top1 == rf_top1:
            veredicto, categoria_final = "confirma_top1", rf_top1
        elif clap_top1 == rf_top2:
            veredicto, categoria_final = "voto_por_top2", rf_top2
        else:
            veredicto, categoria_final = "triple_desacuerdo", ""

        df.loc[idx, "clap_top1"] = clap_top1
        df.loc[idx, "veredicto_clap"] = veredicto
        df.loc[idx, "categoria_final"] = categoria_final

    return df

# ── Umbrales del escenario -- ajustables, empieza con estos y calibra a ojo ──
UMBRAL_CONFIABLE = 0.45   # el top1 debe superar esto para llamarse "confiable"
UMBRAL_AMBIGUA   = 0.12   # si top1-top2 es menor a esto (y top1 no es bajisimo), es "ambigua entre dos"
# Todo lo que no caiga en ninguna de las dos reglas de arriba -> "revisar"


def clasificar_escenario(proba_ordenada: list[tuple[str, float]]) -> dict:
    top1_cat, top1_p = proba_ordenada[0]
    top2_cat, top2_p = proba_ordenada[1] if len(proba_ordenada) > 1 else (None, 0.0)
    margen = top1_p - top2_p

    if margen < UMBRAL_AMBIGUA:
        return {
            "escenario": "ambigua",
            "descripcion": f"{top1_cat} o {top2_cat} (empate real, {top1_p:.0%} vs {top2_p:.0%})",
            "categoria_principal": top1_cat,
            "categoria_alternativa": top2_cat,
        }
    if top1_p >= UMBRAL_CONFIABLE:
        return {
            "escenario": "confiable",
            "descripcion": f"{top1_cat} ({top1_p:.0%})",
            "categoria_principal": top1_cat,
            "categoria_alternativa": "",
        }
    return {
        "escenario": "revisar",
        "descripcion": f"ninguna categoria domina (mejor: {top1_cat} con solo {top1_p:.0%})",
        "categoria_principal": top1_cat,
        "categoria_alternativa": "",
    }


def procesar(modelo_pkl: Path, df: pd.DataFrame, categoricos: dict) -> pd.DataFrame:
    paquete = joblib.load(modelo_pkl)
    modelo, features_cols = paquete["modelo"], paquete["features"]

    filas = []
    for _, row in df.iterrows():
        try:
            x = [float(row[c]) for c in features_cols]
        except (KeyError, ValueError):
            continue

        proba = modelo.predict_proba([x])[0]
        orden = sorted(zip(modelo.classes_, proba), key=lambda t: -t[1])

        esc = clasificar_escenario(orden)
        nombre = row["filename"].removesuffix(".mp3").lower()
        humano = categoricos.get(nombre, {}).get("categoria", "")

        filas.append({
            "filename": row["filename"],
            "spotify_url": row.get("spotify_url", ""),
            "categoria_humano": humano,
            "escenario": esc["escenario"],
            "descripcion": esc["descripcion"],
            "top1_categoria": orden[0][0], "top1_prob": round(orden[0][1], 3),
            "top2_categoria": orden[1][0] if len(orden) > 1 else "",
            "top2_prob": round(orden[1][1], 3) if len(orden) > 1 else 0.0,
            "coincide_con_humano": humano == esc["categoria_principal"] if humano else None,
        })

    return pd.DataFrame(filas)


def main():
    modelo_nombre = "modelo_rf_root_B"
    modelo_pkl = MODELS_DIR / f"{modelo_nombre}.pkl"
    if not modelo_pkl.exists():
        sys.exit(f"[ERROR] No existe {modelo_pkl}. Corre train_tree.py primero.")

    playlist = "rock_english"
    in_csv = find_latest_summary(OUT_DIR, playlist)
    df = pd.read_csv(in_csv, sep=";", encoding="utf-8", dtype=str).fillna("")
    df = df[df["error"].str.strip() == ""]
    categoricos = load_categoricos(MP3_CATEGORICOS_REVISADOS)

    resultado = procesar(modelo_pkl, df, categoricos)
    resultado = resolver_ambiguas_con_clap(resultado, playlist)

    out_csv = OUT_DIR / "scenarios" / "with-models" / f"_escenarios_{modelo_nombre}_{playlist}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(out_csv, sep=";", index=False, encoding="utf-8")

    print("=" * 70)
    print(f"  Distribucion de escenarios ({modelo_nombre}, n={len(resultado)})")
    print("=" * 70)
    conteo = resultado["escenario"].value_counts()
    for esc in ["confiable", "ambigua", "revisar"]:
        n = conteo.get(esc, 0)
        print(f"  {esc:12s}: {n:3d}  ({100*n/len(resultado):.1f}%)")

    # Dentro de "confiable", que tan seguido coincide con tu etiqueta humana --
    # valida si la confianza del modelo realmente correlaciona con acierto
    con_humano = resultado[resultado["categoria_humano"] != ""]
    print(f"\nDe las etiquetadas a mano (n={len(con_humano)}), % de acuerdo por escenario:")
    for esc in ["confiable", "ambigua", "revisar"]:
        sub = con_humano[con_humano["escenario"] == esc]
        if len(sub) == 0:
            continue
        pct = 100 * sub["coincide_con_humano"].mean()
        print(f"  {esc:12s}: {pct:.1f}% de acuerdo con tu etiqueta (n={len(sub)})")

    print(f"\nCSV completo: {out_csv}")
    print("\nEjemplos de 'ambigua' (para que veas el formato):")
    print(resultado[resultado["escenario"] == "ambigua"][["filename", "descripcion", "categoria_humano"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()