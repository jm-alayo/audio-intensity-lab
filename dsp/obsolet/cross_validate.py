from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from calibrate_weights import cargar, calibrar, evaluar, predecir
from classify import find_latest_summary, CATEGORIAS

def accuracy_por_categoria(df, preds) -> pd.DataFrame:
    tmp = df.assign(pred=preds, ok=lambda d: d["pred"] == d["categoria"])
    resumen = tmp.groupby("categoria")["ok"].agg(n="size", accuracy="mean")
    resumen["accuracy"] = (resumen["accuracy"] * 100).round(1)
    return resumen.reindex(CATEGORIAS)

def cross_validar(df, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accs_train, accs_test = [], []

    for fold, (idx_train, idx_test) in enumerate(skf.split(df, df["categoria"]), 1):
        train, test = df.iloc[idx_train].copy(), df.iloc[idx_test].copy()

        params = calibrar(train)   # calibra SOLO con el fold de entrenamiento
        thr_args = (params["thr_e"], params["thr_r"], params["thr_noise"],
                    params["thr_cambio"], params["thr_delta"])

        preds_train = predecir(train, params["w_energia"], params["w_ritmo"], *thr_args)
        preds_test  = predecir(test,  params["w_energia"], params["w_ritmo"], *thr_args)
        acc_test    = (preds_test == test["categoria"].values).mean()

        tabla = accuracy_por_categoria(train, preds_train).add_suffix("_train").join(
                accuracy_por_categoria(test,  preds_test).add_suffix("_test"))

        print(f"\nFold {fold}: train={params['accuracy']}%  test={acc_test*100:.1f}%")
        print(tabla.to_string())

        accs_train.append(params["accuracy"])
        accs_test.append(acc_test*100)

    print(f"\nPromedio TRAIN: {np.mean(accs_train):.1f}%  (lo que reportarías sin CV — inflado)")
    print(f"Promedio TEST:  {np.mean(accs_test):.1f}% ± {np.std(accs_test):.1f}  (el número honesto)")

if __name__ == "__main__":
    SCRIPT_DIR   = Path(__file__).parent
    MP3_CATEGORICOS_REVISADOS = SCRIPT_DIR / "files" / "mp3-categoricos-revisados.csv"

    summary_csv = find_latest_summary(SCRIPT_DIR / "out", "rock_english")
    if summary_csv is None:
        raise SystemExit(f"[ERROR] No se encontró ningún _summary_*_rock_english_segmin*.csv en {SCRIPT_DIR / 'out'}")

    df = cargar(summary_csv, MP3_CATEGORICOS_REVISADOS)
    cross_validar(df)