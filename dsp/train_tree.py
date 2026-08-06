import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from shared.settings import OUT_FILES_DIR, OUT_MODELS_DIR
from shared.ml_func import MLFunc
from extract_features import FEATURES
from obsolet.calibrate_weights import prepare_summary

logger = logging.getLogger(__name__)

def main():

    summary_file = OUT_FILES_DIR / "_summary_12_rock_english_segmin25.csv"

    if summary_file is None:
        sys.exit(f"[ERROR] No se encontró ningún _summary_*_rock_english_segmin*.csv en {OUT_FILES_DIR}")

    df = prepare_summary(summary_file)

    values, categories = MLFunc.prepare_xy(df, FEATURES, label_col="categorico")

    k = MLFunc.safe_k_folds(categories)

    if k < 5:
        logger.warning(f"Pocos datos por clase. Usando k={k} folds en vez de 5 para la validacion cruzada")

    logger.info("  Arbol de decision simple (interpretable)")

    best = (0, None)

    for depth in np.arange(2, 7):
        tree = DecisionTreeClassifier(
            max_depth=depth, 
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )

        media, std = MLFunc.evaluate_cv(tree, values, categories, k=k)
        
        logger.info(f"  max_depth={depth}: balanced_accuracy = {media*100:.1f}% ± {std*100:.1f}")

        if media > best[0]:
            best = (media, depth)

    logger.info(f"\nMejor profundidad: {best[1]} ({best[0]*100:.1f}%)")

    tree = DecisionTreeClassifier(
        max_depth=best[1], 
        min_samples_leaf=5, 
        class_weight="balanced", 
        random_state=42
    ).fit(values, categories)
    
    logger.info("\nEstructura del arbol (que reglas aprendio):")

    MLFunc.save_model(OUT_MODELS_DIR / "modelo_test_tree.pkl", tree, FEATURES)


    logger.info("  Random Forest (mas estable, menos interpretable en detalle)")

    rf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=6, 
        min_samples_leaf=5,
        class_weight="balanced", 
        random_state=42,
    )

    media, std = MLFunc.evaluate_cv(rf, values, categories, k=k)
    logger.info(f"  balanced_accuracy = {media*100:.1f}% ± {std*100:.1f}")

    rf.fit(values, categories)

    logger.info("\nImportancia de cada feature (cuanto la usa el bosque para decidir):")

    for feat, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1]):
        logger.info(f"  {feat:20s}: {imp:.3f}")

    MLFunc.save_model(OUT_MODELS_DIR / "modelo_test_rf.pkl", rf, FEATURES)

if __name__ == "__main__":
    main()
