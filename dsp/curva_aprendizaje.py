import sys
from pathlib import Path
import logging

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))   # music-tagger-benchmark/ -- para poder importar shared/

from shared.settings import OUT_FILES_DIR
from shared.ml_func import MLFunc
from extract_features import FEATURES
from obsolet.calibrate_weights import prepare_summary

logger = logging.getLogger(__name__)

def main():

    summary_csv = OUT_FILES_DIR / "_summary_12_rock_english_segmin25.csv"

    if not summary_csv.exists():
        sys.exit(f"[ERROR] No se encontró ningún _summary_*_rock_english_segmin*.csv")

    df = prepare_summary(summary_csv)
    values, categories = MLFunc.prepare_xy(df, FEATURES, label_col="categorico")

    modelo = DecisionTreeClassifier(
        max_depth=5, 
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    tamaños = np.linspace(0.3, 1.0, 8)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        modelo, 
        values, 
        categories, 
        train_sizes=tamaños, 
        cv=skf,
        scoring="balanced_accuracy", 
        shuffle=True, 
        random_state=42
    )

    logger.info(f"{'n_canciones':>12s} {'train (inflado)':>17s} {'test (honesto)':>16s}")
    
    for n, tr, te in zip(
            train_sizes, 
            train_scores.mean(axis=1), 
            test_scores.mean(axis=1)
        ):

        logger.info(f"{n:12.0f} {tr*100:16.1f}% {te*100:15.1f}%")

    last_change = (
        test_scores.mean(axis=1)[-1] - test_scores.mean(axis=1)[-3]
    )

    logger.info(f"\nCambio de accuracy test entre el 75% y el 100% del dataset actual: {last_change*100:+.1f} pts")

    if last_change > 0.03:
        logger.info("-> Sigue subiendo con fuerza: MAS ETIQUETAS probablemente ayuden bastante.")

    elif last_change > 0.005:
        logger.info("-> Sube, pero despacio: mas etiquetas ayudan, con retornos decrecientes.")

    else:
        logger.info("-> Practicamente plano: el techo NO es de cantidad de datos")

if __name__ == "__main__":
    main()