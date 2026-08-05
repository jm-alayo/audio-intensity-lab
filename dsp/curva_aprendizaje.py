import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))   # calibrate_weights.py vive en obsolet/, no junto a este script

from obsolet.calibrate_weights import cargar
from classify import find_latest_summary
from train_tree import FEATURES, preparar_xy   # reusa exactamente las mismas columnas

def main():
    summary_csv = find_latest_summary(SCRIPT_DIR / "out", "rock_english")
    if summary_csv is None:
        sys.exit(f"[ERROR] No se encontró ningún _summary_*_rock_english_segmin*.csv en {SCRIPT_DIR / 'out'}")

    df = cargar(summary_csv, SCRIPT_DIR / "files" / "mp3-categoricos-revisados.csv")
    X, y = preparar_xy(df)

    modelo = DecisionTreeClassifier(
        max_depth=4, min_samples_leaf=5, class_weight="balanced", random_state=42
    )

    tamaños = np.linspace(0.3, 1.0, 8)   # 30%, 40%, ..., 100% del dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        modelo, X, y, train_sizes=tamaños, cv=skf,
        scoring="balanced_accuracy", shuffle=True, random_state=42,
    )

    print(f"{'n_canciones':>12s} {'train (inflado)':>17s} {'test (honesto)':>16s}")
    for n, tr, te in zip(train_sizes, train_scores.mean(axis=1), test_scores.mean(axis=1)):
        print(f"{n:12.0f} {tr*100:16.1f}% {te*100:15.1f}%")

    # Pendiente del tramo final: si sigue subiendo con fuerza, mas datos ayudan mucho.
    ultimo_salto = (test_scores.mean(axis=1)[-1] - test_scores.mean(axis=1)[-3])
    print(f"\nCambio de accuracy test entre el 75% y el 100% del dataset actual: {ultimo_salto*100:+.1f} pts")
    if ultimo_salto > 0.03:
        print("-> Sigue subiendo con fuerza: MAS ETIQUETAS probablemente ayuden bastante.")
    elif ultimo_salto > 0.005:
        print("-> Sube, pero despacio: mas etiquetas ayudan, con retornos decrecientes.")
    else:
        print("-> Practicamente plano: el techo NO es de cantidad de datos -- revisar")
        print("   consistencia de etiquetado o definicion de categorias en su lugar.")


if __name__ == "__main__":
    main()