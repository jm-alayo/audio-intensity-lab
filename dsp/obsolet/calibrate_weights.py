import sys
import logging
import numpy as np
from pathlib import Path
from scipy.optimize import differential_evolution

DSP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DSP_DIR))
sys.path.insert(0, str(DSP_DIR.parent))

from classify import classify_2d
from shared.settings import FEATURES_DIR, CANCIONES_XLSX
from shared.utils import find_latest_summary, prepare_summary, derive_threshold_bounds

logger = logging.getLogger(__name__)

W_E_CENTROID_FIXED = 0.05
W_E_DYN_FIXED       = 0.10
W_R_TEMPO_FIXED     = 0.20

def predict(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta, agg="mediana"):
    w_e = np.array(w_e) / np.sum(w_e)
    w_r = np.array(w_r) / np.sum(w_r)
    suffix = "_mediana" if agg == "mediana" else "_promedio"

    energia = (
        w_e[0]*df[f"flat_n{suffix}"]*0.5 + w_e[0]*df[f"zcr_n{suffix}"]*0.5 + 
        w_e[1]*df[f"ostr_n{suffix}"] + 
        w_e[2]*df[f"cent_n{suffix}"] + 
        w_e[3]*(1 - df[f"dyn_n{suffix}"])
    )

    ritmo = w_r[0]*df[f"onset_n{suffix}"] + w_r[1]*df[f"tempo_n{suffix}"]
    noise = 0.5*df[f"flat_n{suffix}"] + 0.5*df[f"zcr_n{suffix}"]

    preds = [
        classify_2d(
            energia=e, 
            noise=n, 
            ritmo=r,
            pendiente_energia=pe, 
            pendiente_ritmo=pr,
            delta_ritmo=dr, 
            delta_energia=de, 
            n_segments=ns,
            thr_e=thr_e, 
            thr_r=thr_r, 
            thr_noise=thr_noise,
            thr_cambio=thr_cambio, 
            thr_delta=thr_delta)["categoria"]
        for e, r, n, pe, de, pr, dr, ns in zip(
            energia, 
            ritmo, 
            noise,
            df["pendiente_energia"], 
            df["delta_energia"],
            df["pendiente_ritmo"], 
            df["delta_ritmo"], 
            df["n_segments"]
        )
    ]
    return np.array(preds)

def evaluate(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta, agg="mediana"):

    preds = predict(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta, agg)

    return (preds == df["categorico"].values).mean()

def objective(params, df):

    w_e = [*params[:2], W_E_CENTROID_FIXED, W_E_DYN_FIXED]
    w_r = [params[2], W_R_TEMPO_FIXED]
    thr_e, thr_r, thr_noise, thr_cambio, thr_delta = params[3:8]

    preds = predict(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta)

    categoricos = df["categorico"].values

    recalls = [(preds[categoricos == cat] == cat).mean() for cat in np.unique(categoricos)]

    return -np.mean(recalls)

def calibrate(df):

    bounds = [(0.01,1)]*3 + derive_threshold_bounds(df)

    result = differential_evolution(
        objective, 
        bounds, 
        args=(df,), 
        seed=42, 
        maxiter=200, 
        popsize=25,
        strategy="best1bin",
        tol=1e-4, 
        workers=1
    )

    w_e = [*result.x[:2], W_E_CENTROID_FIXED, W_E_DYN_FIXED]
    w_r = [result.x[2], W_R_TEMPO_FIXED]

    thr_e, thr_r, thr_noise, thr_cambio, thr_delta = result.x[3:8]

    acc = evaluate(df, w_e, w_r, thr_e, thr_r, thr_noise, thr_cambio, thr_delta)

    return {
        "w_energy":  (np.array(w_e)/np.sum(w_e)).round(3).tolist(),
        "w_rhythm":  (np.array(w_r)/np.sum(w_r)).round(3).tolist(),
        "thr_e": round(thr_e,3), "thr_r": round(thr_r,3), "thr_noise": round(thr_noise,3),
        "thr_cambio": round(thr_cambio,3), "thr_delta": round(thr_delta,3),
        "accuracy": round(acc*100, 1),
        "balanced_accuracy": round(-result.fun*100, 1),
    }

if __name__ == "__main__":

    playlist = "rock_english"
    summary_csv = find_latest_summary(FEATURES_DIR, playlist)

    if summary_csv is None:
        raise SystemExit(f"[ERROR] No _summary_*_{playlist}_segmin*.csv found in {FEATURES_DIR}")

    df = prepare_summary(summary_csv, CANCIONES_XLSX)

    logger.info(calibrate(df))
