import sys
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.optimize import differential_evolution

DSP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DSP_DIR))            # dsp/ -- para poder importar classify
sys.path.insert(0, str(DSP_DIR.parent))     # music-tagger-benchmark/ -- para poder importar shared/

from classify import classify_2d
from shared.settings import OUT_FILES_DIR, CANCIONES_XLSX
from shared.utils import find_latest_summary
from extract_features import load_categorized

# Grados de libertad reducidos: 12 parámetros sobre 119 canciones (con clases
# desbalanceadas, una con solo 10 ejemplos) es ambicioso — riesgo de memorizar
# ruido específico del dataset en vez de aprender señal real. Se fijan a mano
# las 3 dimensiones de menor peso relativo en el modelo v3 original
# (extract_features.py): centroide, dinámica (energía) y tempo (ritmo).
# Quedan libres: noise + onset_strength (energía), onset_density + perc_ratio
# (ritmo) y los 5 umbrales = 9 parámetros libres en vez de 12.
W_E_CENTROIDE_FIJO = 0.05   # bajo pero no-cero — el optimizador lo llevaba a ~0.02 libremente
W_E_DYN_FIJO       = 0.10   # peso original v3
W_R_TEMPO_FIJO     = 0.20   # peso original v3

def prepare_summary(summary_file: Path, agg: str = "mediana") -> pd.DataFrame:

    df = pd.read_csv(summary_file, sep=";")
    lab = load_categorized(CANCIONES_XLSX)

    lab["categorico"] = lab["categorico"].replace(
        {"incrementable": "incrementable-decreciente", "decreciente": "incrementable-decreciente"})

    df = df.merge(lab[["music_name","categorico"]], on="music_name")

    return df

def predecir(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta, agg="mediana"):
    w_e = np.array(w_e) / np.sum(w_e)   # normaliza para que sumen 1
    w_r = np.array(w_r) / np.sum(w_r)
    sufijo = "_mediana" if agg == "mediana" else "_promedio"

    energia = (w_e[0]*df[f"flat_n{sufijo}"]*0.5 + w_e[0]*df[f"zcr_n{sufijo}"]*0.5   # noise_n = mezcla flat/zcr
               + w_e[1]*df[f"ostr_n{sufijo}"] + w_e[2]*df[f"cent_n{sufijo}"]
               + w_e[3]*(1 - df[f"dyn_n{sufijo}"]))
    ritmo = w_r[0]*df[f"onset_n{sufijo}"] + w_r[1]*df[f"perc_n{sufijo}"] + w_r[2]*df[f"tempo_n{sufijo}"]
    noise = 0.5*df[f"flat_n{sufijo}"] + 0.5*df[f"zcr_n{sufijo}"]

    preds = [
        classify_2d(energia=e, noise=n, ritmo=r,
                    pendiente_energia=pe, pendiente_ritmo=pr,
                    delta_ritmo=dr, delta_energia=de, n_segments=ns,
                    thr_e=thr_e, thr_r=thr_r, thr_noise=thr_noise,
                    thr_cambio=thr_cambio, thr_delta=thr_delta)["categoria"]
        for e, r, n, pe, de, pr, dr, ns in zip(
            energia, ritmo, noise,
            df["pendiente_energia"], df["delta_energia"],
            df["pendiente_ritmo"], df["delta_ritmo"], df["n_segments"])
    ]
    return np.array(preds)

def evaluar(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta, agg="mediana"):
    preds = predecir(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta, agg)
    return (preds == df["categoria"].values).mean()

def objetivo(params, df):
    # 9 libres: w_e[noise, ostr] + w_r[onset, perc] + 5 umbrales
    w_e = [*params[:2], W_E_CENTROIDE_FIJO, W_E_DYN_FIJO]
    w_r = [*params[2:4], W_R_TEMPO_FIJO]
    thr_e, thr_r, thr_noise, thr_cambio, thr_delta = params[4:]

    preds = predecir(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta)
    categoria = df["categoria"].values

    # balanced accuracy: cada categoría pesa IGUAL sin importar cuántos
    # ejemplos tenga. Con accuracy global, sacrificar las 8 alta-agresiva
    # (8.4% del train) para afinar mejor las 29 alta-ritmica le convenía
    # matemáticamente al optimizador — este es el punto ciego que corrige.
    recalls = [(preds[categoria == cat] == cat).mean() for cat in np.unique(categoria)]
    return -np.mean(recalls)   # differential_evolution minimiza

def calibrar(df):
    # 2 pesos energia + 2 pesos ritmo + 5 umbrales = 9 parametros libres
    # (centroide/dinamica/tempo fijados a mano, ver W_E_CENTROIDE_FIJO etc.)
    bounds = [(0.01,1)]*4 + [(0.25,0.6),(0.35,0.8),(0.2,0.8),(0.05,0.3),(0.05,0.3)]
    result = differential_evolution(objetivo, bounds, args=(df,), seed=42,
                                     maxiter=200, popsize=25, tol=1e-4, workers=-1)
    w_e = [*result.x[:2], W_E_CENTROIDE_FIJO, W_E_DYN_FIJO]
    w_r = [*result.x[2:4], W_R_TEMPO_FIJO]
    thr_e, thr_r, thr_noise, thr_cambio, thr_delta = result.x[4:]
    # "accuracy" se sigue reportando cruda (igual que evaluar()) para que las
    # comparaciones train/test en cross_validate.py sigan siendo manzanas con
    # manzanas; "balanced_accuracy" es la métrica que objetivo() optimiza.
    acc = evaluar(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta)
    return {
        "w_energia": (np.array(w_e)/np.sum(w_e)).round(3).tolist(),
        "w_ritmo":   (np.array(w_r)/np.sum(w_r)).round(3).tolist(),
        "thr_e": round(thr_e,3), "thr_r": round(thr_r,3), "thr_noise": round(thr_noise,3),
        "thr_cambio": round(thr_cambio,3), "thr_delta": round(thr_delta,3),
        "accuracy": round(acc*100, 1),
        "balanced_accuracy": round(-result.fun*100, 1),
    }

if __name__ == "__main__":

    summary_csv = find_latest_summary(OUT_FILES_DIR, "rock_english")
    
    if summary_csv is None:
        raise SystemExit(f"[ERROR] No se encontró ningún _summary_*_rock_english_segmin*.csv en {OUT_FILES_DIR}")

    df = prepare_summary(summary_csv)

    print(calibrar(df))