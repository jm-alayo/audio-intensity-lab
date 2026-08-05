import re
import sys
import time
from pathlib import Path
from collections import Counter
import pandas as pd

# La consola de Windows suele quedar en cp1252, que no puede imprimir "█"/"⚠"
# de los reportes de abajo y revienta a mitad de la corrida.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_DIR   = Path(__file__).parent

OUT_DIR = SCRIPT_DIR / "out"
MP3_CATEGORICOS_REVISADOS = SCRIPT_DIR / "files" / "mp3-categoricos-revisados.csv"

# Umbrales calibrados sobre los HUECOS de la distribución del run rock_english
# (v2, 10 canciones), no sobre la mediana (que fuerza un split 50/50 arbitrario
# — así fue como Love Hurts terminó mal clasificada la vez pasada).
# RECALIBRAR tras correr v3 y tras ampliar el dataset: son válidos para esta
# escala comprimida, no absolutos.
THR_E      = 0.335   # hueco energía: 0.313 → 0.361
THR_R      = 0.390   # hueco ritmo:   0.332 → 0.451
THR_CAMBIO = 0.10    # cambio total de energía inicio→fin (pendiente * (n_segments-1))
THR_DELTA  = 0.12    # delta_extremos — atrapa clímax concentrados (v3, ej. Exit Music)
THR_MARGEN = 0.04    # distancia al umbral bajo la cual se pide revisión manual
THR_NOISE = 0.5

# CATEGORIAS = ["alta-agresiva", "alta-ritmica", "baja-contemplativa", "baja-ritmica", "incrementable", "decreciente"]
CATEGORIAS = ["alta-agresiva", "alta-ritmica", "baja-contemplativa", "baja-ritmica", "incrementable-decreciente"]

def classify_2d(
        energia: float, noise: float,
        ritmo: float, 
        pendiente_energia: float,
        pendiente_ritmo: float, delta_ritmo: float,
        delta_energia: float, 
        n_segments: int,
        thr_e: float = THR_E, thr_r: float = THR_R, thr_noise: float = THR_NOISE,
        thr_cambio: float = THR_CAMBIO, thr_delta: float = THR_DELTA,
        thr_margen: float = THR_MARGEN
    ) -> dict:

    # se multiplica porque la pendiente por si sola es engañosa, tenemos que agregar a la ecuacion los segmentos totales
    cambio_e = pendiente_energia * (n_segments - 1)

    # Paso 1: tendencia como gate
    if cambio_e >= thr_cambio or delta_energia >= thr_delta:

        margen = max(cambio_e - thr_cambio, delta_energia - thr_delta)
        
        return {
                # "categoria": "incrementable", 
                "categoria": "incrementable-decreciente",
                "cambio_energia": round(cambio_e, 3),
                "margen": round(margen, 3),
                "revisar": bool(margen < thr_margen)
            }

    if cambio_e <= -thr_cambio or delta_energia <= -thr_delta:

        margen = max(-thr_cambio - cambio_e, -thr_delta - delta_energia)

        return {
            # "categoria": "decreciente", 
            "categoria": "incrementable-decreciente",
            "cambio_energia": round(cambio_e, 3),
            "margen": round(margen, 3), 
            "revisar": bool(margen < thr_margen)
        }

    # Paso 2: grilla 2D
    e_alta = energia >= thr_e
    r_alto = ritmo >= thr_r
    if e_alta and r_alto:
        cat = "alta-ritmica"
    elif e_alta:
        cat = "alta-agresiva"
    elif r_alto:
        cat = "baja-ritmica"
    else:
        cat = "baja-contemplativa"

    # Paso 3: margen de confianza (también contra el gate de tendencia: una
    # canción con cambio 0.09 casi fue incrementable → revisar)
    margen = min(
        abs(energia - thr_e), abs(ritmo - thr_r), abs(noise - thr_noise),
        abs(abs(pendiente_energia*(n_segments-1)) - thr_cambio),
        abs(abs(pendiente_ritmo*(n_segments-1)) - thr_cambio)
    )

    return {
        "categoria": cat, 
        "cambio_energia": round(cambio_e, 3),
        "margen": round(margen, 3), 
        "revisar": bool(margen < thr_margen)
    }


def load_categoricos(csv_path: Path) -> dict[str, dict]:

    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False)

    return {
        row["filename"].lower(): {"categoria": row["categoria"], "spotify_url": row["spotify_url"]}
        for _, row in df.iterrows()
    }

# extract_features.py genera _summary_{id}_{playlist}_segmin{N}.csv (sin
# timestamp); siempre se trabaja con el ID más alto disponible.
def find_latest_summary(out_dir: Path, playlist: str, id: int = None) -> Path | None:

    if id is not None:
        candidatos = sorted(out_dir.glob(f"_summary_{id}_{playlist}_segmin*.csv"))
        return candidatos[0] if candidatos else None

    pattern = re.compile(rf"^_summary_(\d+)_{re.escape(playlist)}_segmin\d+\.csv$")
    candidatos = [(int(m.group(1)), f) for f in out_dir.glob(f"_summary_*_{playlist}_segmin*.csv") if (m := pattern.match(f.name))]

    return max(candidatos, key=lambda t: t[0])[1] if candidatos else None

# Cada escenario evalúa la misma clasificación con distintas columnas de
# entrada: (col_energia, col_ritmo, col_pendiente_energia, col_delta_energia)
ESCENARIOS = {
    "promedios_rms": ("score_energia_promedio",     "score_ritmo_promedio", "score_noise_promedio", "pendiente_energia",     "delta_energia",     "pendiente_ritmo", "delta_ritmo"),
    "medianas_rms":  ("score_energia_mediana",      "score_ritmo_mediana",  "score_noise_mediana",  "pendiente_energia",     "delta_energia",     "pendiente_ritmo", "delta_ritmo"),
    "promedios_lra": ("score_energia_lra_promedio", "score_ritmo_promedio", "score_noise_promedio", "pendiente_energia_lra", "delta_energia_lra", "pendiente_ritmo", "delta_ritmo"),
    "medianas_lra":  ("score_energia_lra_mediana",  "score_ritmo_mediana",  "score_noise_mediana",  "pendiente_energia_lra", "delta_energia_lra", "pendiente_ritmo", "delta_ritmo"),
}

def _classify_row_scenario(
        row: pd.Series,
        col_energia: str, col_ritmo: str, col_noise: str,
        col_slope_energia: str, col_delta_energia: str,
        col_slope_ritmo: str, col_delta_ritmo: str,
        categoricos: dict[str, dict]
    ) -> dict | None:

    try:
        energia    = float(row[col_energia])
        ritmo      = float(row[col_ritmo])
        noise      = float(row[col_noise])
        slope_e    = float(row[col_slope_energia])
        delta_e    = float(row[col_delta_energia])
        slope_r    = float(row[col_slope_ritmo])
        delta_r    = float(row[col_delta_ritmo])
        n_segments = int(row["n_segments"])

    except (KeyError, ValueError):
        return None

    res = classify_2d(energia, noise, ritmo, slope_e, slope_r, delta_r, delta_e, n_segments)

    nombre     = row["filename"].removesuffix(".mp3").lower()
    categorico = categoricos.get(nombre, {})

    return {
        "filename":           row["filename"],
        "spotify_url":        categorico.get("spotify_url", ""),
        "duration_s":         row.get("duration_s", ""),
        "cambio_energia":     res["cambio_energia"],
        "margen":             res["margen"],
        "categoria_sugerida": res["categoria"],
        "categoria_humano":   categorico.get("categoria", ""),
        "revisar":            res["revisar"],
    }

def validate(classified_csv: Path) -> float | None:

    df = pd.read_csv(classified_csv, sep=";", encoding="utf-8", dtype=str).fillna("")

    df["categoria_humano"]   = df["categoria_humano"].str.strip()
    df["categoria_sugerida"] = df["categoria_sugerida"].str.strip()

    df = df[(df["categoria_humano"] != "") & (df["categoria_sugerida"] != "")]

    total = len(df)
    if total == 0:
        print("[AVISO] No hay filas con categoria_humano rellena.")

        return None

    acuerdos  = int((df["categoria_humano"] == df["categoria_sugerida"]).sum())
    confusion = Counter(zip(df["categoria_sugerida"], df["categoria_humano"]))

    pct = 100 * acuerdos / total

    df["validez"] = df["categoria_humano"] == df["categoria_sugerida"]

    matriz = pd.DataFrame(
        [[confusion.get((s_cat, h_cat), 0) for h_cat in CATEGORIAS] for s_cat in CATEGORIAS],
        index=CATEGORIAS, columns=CATEGORIAS
    )
    matriz.index.name   = "sugerida \\ humano"

    print(f"\nValidación sobre {total} canciones etiquetadas manualmente:")
    print(f"  Acuerdo sistema vs humano: {acuerdos}/{total} = {pct:.1f}%")
    print("\n  Matriz de confusión (sistema → humano):")
    print(matriz.to_string())

    return pct

def main():

    arg_playlist = "rock_english"

    in_csv = find_latest_summary(OUT_DIR, arg_playlist)

    # imprimir el nombre del archivo
    print(f"Nombre del archivo: {in_csv.name}")

    if in_csv is None:
        sys.exit(
            f"[ERROR] No se encontró ningún _summary_*_{arg_playlist}_segmin*.csv en {OUT_DIR}\n"
            "Ejecuta extract_features.py primero."
        )

    id_summary = re.match(r"^_summary_(\d+)_", in_csv.name).group(1)

    df = pd.read_csv(in_csv, sep=";", encoding="utf-8", dtype=str).fillna("")
    df = df[df["error"].str.strip() == ""]

    if df.empty:
        sys.exit("[ERROR] No hay filas válidas en el CSV de entrada.")

    categoricos = load_categoricos(MP3_CATEGORICOS_REVISADOS)

    print("=" * 60)
    print(f"  Playlist      : {arg_playlist}")
    print(f"  Summary ID    : {id_summary}")
    print(f"  Canciones     : {len(df)}")
    print(f"  URLs Spotify  : {len(categoricos)}/{len(df)} encontradas")
    print("=" * 60)

    out_headers = [
        "filename", "duration_s", "spotify_url",
        "cambio_energia", "margen",
        "categoria_sugerida", "categoria_humano", "revisar"
    ]

    scenarios_dir = OUT_DIR / "scenarios" / f"scenario_{time.strftime('%Y%m%d_%H%M%S')}"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    resumen_pct: dict[str, float | None] = {}
    resumen_rev: dict[str, int] = {}

    for nombre_escenario, (col_e, col_r, col_noise, col_se, col_de, col_sr, col_dr) in ESCENARIOS.items():
        records = [
            rec for _, row in df.iterrows()
            if (rec := _classify_row_scenario(row, col_e, col_r, col_noise, col_se, col_de, col_sr, col_dr, categoricos)) is not None
        ]

        if not records:
            print(f"\n[AVISO] Escenario '{nombre_escenario}': no se clasificó ninguna canción.")
            continue

        counts = Counter(r["categoria_sugerida"] for r in records)
        total  = len(records)

        out_csv = scenarios_dir / f"_classified_sum{id_summary}_{arg_playlist}_esc_{nombre_escenario}_clf.csv"
        pd.DataFrame(records, columns=out_headers).to_csv(out_csv, sep=";", index=False, encoding="utf-8")

        print(f"\n── Escenario: {nombre_escenario} ──")
        for cat in CATEGORIAS:
            n = counts.get(cat, 0)
            bar = "█" * int(40 * n / total)
            print(f"  {cat:<20} {n:4d}  ({100*n/total:.1f}%)  {bar}")

        n_rev = sum(1 for r in records if r["revisar"])
        if n_rev:
            print(f"  ⚠ {n_rev} canción(es) marcadas para revisión manual (margen < {THR_MARGEN})")

        print(f"  CSV: {out_csv}")

        resumen_pct[nombre_escenario] = validate(out_csv)
        resumen_rev[nombre_escenario] = n_rev

    resumen = pd.DataFrame({
        "escenario":  list(resumen_pct.keys()),
        "% acierto":  [
            f"{p:.1f}% ({resumen_rev[esc]} a revisar)" if p is not None else "sin datos"
            for esc, p in resumen_pct.items()
        ],
    }).set_index("escenario")

    print("\n" + "=" * 60)
    print("  Resumen de escenarios — % de acierto vs categoria_humano")
    print("=" * 60)
    print(resumen.to_string())


if __name__ == "__main__":
    main()
