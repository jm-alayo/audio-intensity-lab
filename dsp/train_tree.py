# train_tree.py — reemplaza calibrate_weights.py: en vez de buscar pesos para
# una formula lineal, entrena un arbol/Random Forest que aprende sus propios
# cortes (incluyendo interacciones no-lineales) directamente de las columnas
# ya calculadas por extract_features.py.
#
# NO requiere las features normalizadas (_n) especificamente calibradas con
# LO/HI -- los arboles son invariantes a reescalado monotono, asi que se le
# pueden pasar las columnas CRUDAS o las _n, da lo mismo para el resultado
# (aunque _n sigue siendo util para que vos entiendas las magnitudes).

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))   # calibrate_weights.py vive en obsolet/, no junto a este script

from obsolet.calibrate_weights import cargar   # reutiliza el mismo loader/merge de etiquetas
from classify import find_latest_summary

OUT_MODELS_DIR = SCRIPT_DIR / "models-tree"
OUT_FILES_DIR = SCRIPT_DIR / "out"

# Columnas de entrada: las mismas que ya tienes en el summary. Podes pasar las
# crudas o las _n -- para un arbol da igual (invariante a reescalado).
FEATURES = [
    "onset_n_mediana", "tempo_n_mediana", "cent_n_mediana", "ostr_n_mediana",
    "flat_n_mediana", "zcr_n_mediana", "perc_n_mediana", "dyn_n_mediana",
    "pendiente_energia", "delta_energia", "pendiente_ritmo", "delta_ritmo",
    "n_segments",
]

def preparar_xy(df: pd.DataFrame):
    X = df[FEATURES].fillna(0).values
    y = df["categoria"].values
    return X, y

def evaluar_cv(modelo, X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=skf, scoring="balanced_accuracy")
    return scores.mean(), scores.std()

def main():
    summary_csv = find_latest_summary(OUT_FILES_DIR, "rock_english")
    if summary_csv is None:
        sys.exit(f"[ERROR] No se encontró ningún _summary_*_rock_english_segmin*.csv en {OUT_FILES_DIR}")

    df = cargar(summary_csv, SCRIPT_DIR / "files" / "mp3-categoricos-revisados.csv")
    X, y = preparar_xy(df)

    print("=" * 60)
    print("  Arbol de decision simple (interpretable)")
    print("=" * 60)
    mejor = (0, None)
    for profundidad in [2, 3, 4, 5, 6]:
        arbol = DecisionTreeClassifier(
            max_depth=profundidad, min_samples_leaf=5,
            class_weight="balanced",   # equivalente automatico al objetivo balanceado que armamos a mano antes
            random_state=42,
        )
        media, std = evaluar_cv(arbol, X, y)
        print(f"  max_depth={profundidad}: balanced_accuracy = {media*100:.1f}% ± {std*100:.1f}")
        if media > mejor[0]:
            mejor = (media, profundidad)

    print(f"\nMejor profundidad: {mejor[1]} ({mejor[0]*100:.1f}%)")

    # Entrena el arbol final sobre TODOS los datos (igual que calibrate_final.py)
    # y muestra su estructura -- esto es lo que reemplaza a "los 7 pesos calibrados"
    arbol_final = DecisionTreeClassifier(
        max_depth=mejor[1], min_samples_leaf=5, class_weight="balanced", random_state=42
    ).fit(X, y)
    print("\nEstructura del arbol (que reglas aprendio):")
    print(export_text(arbol_final, feature_names=FEATURES))

    OUT_MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"modelo": arbol_final, "features": FEATURES}, OUT_MODELS_DIR / "modelo_arbol_root_A.pkl")
    print(f"\nModelo guardado en: {OUT_MODELS_DIR / 'modelo_arbol_root_A.pkl'}")

    print("=" * 60)
    print("  Random Forest (mas estable, menos interpretable en detalle)")
    print("=" * 60)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        class_weight="balanced", random_state=42,
    )
    media, std = evaluar_cv(rf, X, y)
    print(f"  balanced_accuracy = {media*100:.1f}% ± {std*100:.1f}")

    rf.fit(X, y)
    print("\nImportancia de cada feature (cuanto la usa el bosque para decidir):")
    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {feat:20s}: {imp:.3f}")

    joblib.dump({"modelo": rf, "features": FEATURES}, OUT_MODELS_DIR / "modelo_rf.pkl")
    print(f"\nModelo guardado en: {OUT_MODELS_DIR / 'modelo_rf.pkl'}")


if __name__ == "__main__":
    main()
