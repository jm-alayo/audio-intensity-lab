# classify_clap.py — Route B: clasificacion zero-shot sobre AUDIO CRUDO,
# sin pasar por extract_features.py en absoluto. Archivo totalmente separado
# de Route A (extract_features.py / classify.py / train_tree.py) -- no los
# toca, no depende de ellos, solo reusa validate()/load_categoricos() para
# que la comparacion final sea con la misma metrica de siempre.
#
# Requiere: pip install transformers torch --break-system-packages
# (librosa ya lo tenes instalado para Route A)

import sys
from pathlib import Path
import librosa
import pandas as pd
import torch
from transformers import pipeline

SCRIPT_DIR   = Path(__file__).parent
ANALYZE_DIR  = SCRIPT_DIR.parent / "analyze-mp3"
sys.path.insert(0, str(ANALYZE_DIR))   # classify.py / extract_features.py viven en analyze-mp3/

from classify import MP3_CATEGORICOS_REVISADOS, load_categoricos, validate
from extract_features import PLAYLIST_BASE, strip_mp3   # reusa la misma ruta de mp3s que ya tenes: spotdl-poc\out-playlist

RESULTS_DIR = SCRIPT_DIR / "results"
# "laion/larger_clap_music" tiene logit_scale ~1.03 (debería ser ~50-100):
# da probabilidades casi uniformes (~20% por categoria) sin importar el audio
# -- confirmado con silencio/ruido/musica real dando el mismo embedding.
# clap-htsat-unfused es el checkpoint que HuggingFace usa en su propia
# documentacion del pipeline y sí discrimina correctamente.
MODEL_ID = "laion/clap-htsat-unfused"
SR_CLAP  = 48000   # CLAP espera 48kHz -- resample aparte de lo que uses en Route A

# Las descripciones en texto son EL prompt de este enfoque -- iterar aca es
# el equivalente a lo que haces con Gemini, pero mas barato de probar en loop.
CANDIDATOS = {
    "alta-agresiva": (
        "a heavy rock song with highly distorted electric guitars, loud and intense drums, "
        "raw and saturated garage rock, grunge or metal sound, harsh wall of noise, "
        "screamed or gritty vocals, high sonic friction and violent energy"
    ),
    
    "alta-ritmica": (
        "a song with a fast and constant danceable pulse, upbeat tempo, dominant rhythmic "
        "section with an elastic bassline and steady drum beat, pop or disco percussion, "
        "linear instrumentation or a high-intensity monolithic sonic block that invites body movement"
    ),
    
    "baja-contemplativa": (
        "a minimalist and intimate musical piece, solitary acoustic guitar or soft piano, "
        "floating, nostalgic and reflective atmosphere, empty soundscape without heavy drums, "
        "delicate campfire or basement dynamics, whispers and pure low-friction acoustic emotion"
    ),
    
    "baja-ritmica": (
        "a track with an elastic swaying motion and subtle groove, light boogie-woogie or walking bassline, "
        "strummed acoustic guitar or clean telecaster notes with short slapback delay, discreet drums "
        "with brushes or dry snare hits, upbeat tempo but with low volume and zero distortion"
    ),
    
    "incrementable-decreciente": (
        "a song structured as an ascending spiral over a six-eighths time signature rhythm, "
        "starting low in silence with a solitary arpeggio and gradually building tension and instruments "
        "millimeter by millimeter up to a volcanic eruption, or vice versa, fading out from chaos "
        "down to a sonic thread"
    )
}

# CLAP se entrenó con captions cortas y directas (LAION-Audio-630K), no con prosa
# metaforica -- version corta para comparar contra CANDIDATOS antes de comprometerse
# a una de las dos (probar primero con n_max chico).
CANDIDATOS_CORTO = {
    "alta-agresiva":             "aggressive distorted rock music with harsh electric guitars and loud drums",
    "alta-ritmica":               "upbeat danceable pop or disco music with a strong steady rhythm",
    "baja-contemplativa":        "calm acoustic ballad with soft piano or guitar and gentle vocals",
    "baja-ritmica":               "soft acoustic music with a light groove and clean guitar",
    "incrementable-decreciente":  "a song that gradually builds from quiet to loud, or fades from loud to quiet",
}


def cargar_clasificador():
    clasificador = pipeline(
        task="zero-shot-audio-classification", model=MODEL_ID,
        device=0 if torch.cuda.is_available() else -1,
    )
    # laion/larger_clap_music tiene logit_scale ~1.03 (degenerado, ver comentario
    # de MODEL_ID mas arriba) -- este chequeo evita repetir esa investigacion a
    # ciegas si en el futuro se cambia de checkpoint.
    logit_scale = clasificador.model.logit_scale_a.exp().item()
    if logit_scale < 10:
        raise RuntimeError(
            f"[ALERTA] logit_scale={logit_scale:.2f} -- checkpoint '{MODEL_ID}' probablemente "
            f"degenerado (probabilidades casi uniformes). Probar otro checkpoint antes de continuar."
        )
    print(f"logit_scale = {logit_scale:.2f} (ok, checkpoint parece sano)")
    return clasificador


def clasificar_playlist(playlist: str = "rock_english", n_max: int | None = None, candidatos: dict = CANDIDATOS):
    clasificador = cargar_clasificador()
    categoricos = load_categoricos(MP3_CATEGORICOS_REVISADOS)

    # mismo filtro que usa extract_features.py (Route A) contra el CSV de
    # revisión manual, para comparar peras con peras -- sin esto se cuelan acá
    # mp3s que Route A ya descartó (duplicados, pruebas, etc.)
    df_filter = pd.read_csv(ANALYZE_DIR / "files" / "mp3-categoricos-revisados.csv",
                            sep=";", dtype=str, keep_default_na=False)
    allowed_names = {strip_mp3(n).lower() for n in df_filter["filename"]}

    carpeta = PLAYLIST_BASE / playlist
    mp3s = [p for p in sorted(carpeta.glob("*.mp3")) if strip_mp3(p.name).lower() in allowed_names]
    if n_max:
        mp3s = mp3s[:n_max]

    etiquetas_texto = list(candidatos.values())
    cat_por_texto = {v: k for k, v in candidatos.items()}

    resultados = []
    for mp3 in mp3s:
        y, _ = librosa.load(str(mp3), sr=SR_CLAP, mono=True)
        salida = clasificador(y, candidate_labels=etiquetas_texto)
        top1 = salida[0]

        nombre = mp3.stem.lower()
        humano = categoricos.get(nombre, {}).get("categoria", "")

        resultados.append({
            "filename": mp3.name,
            "duration_s": len(y) / SR_CLAP,
            "spotify_url": categoricos.get(nombre, {}).get("spotify_url", ""),
            "cambio_energia": "",
            "margen": round(top1["score"] - salida[1]["score"], 3),
            "categoria_sugerida": cat_por_texto[top1["label"]],
            "categoria_humano": humano,
            "revisar": bool(top1["score"] < 0.35),
        })
        print(f"  {mp3.name[:45]:45s} -> {cat_por_texto[top1['label']]:22s} ({top1['score']:.2f})")

    return resultados


def main():
    resultados = clasificar_playlist("rock_english", candidatos=CANDIDATOS_CORTO)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "_classified_clap_rock_english.csv"
    pd.DataFrame(resultados).to_csv(out_csv, sep=";", index=False, encoding="utf-8")

    print(f"\nCSV: {out_csv}")
    validate(out_csv)   # misma funcion, misma metrica que Route A -- comparacion justa

if __name__ == "__main__":
    main()