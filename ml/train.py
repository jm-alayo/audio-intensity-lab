import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.settings import FEATURES_DIR, TRACKS_XLSX, ML_ARTIFACTS_DIR, ML_RUNS_DIR
from shared.utils import find_latest_summary, prepare_summary
from features.extract import FEATURES
from ml.models import TreeClassifier, LinearRuleClassifier
from ml.calibration import ModelCalibrator
from ml.confidence import ConfidenceRouter
from ml.evaluation import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_dataset(playlist: str = "rock_english", summary_id: int = None) -> pd.DataFrame:

    summary_csv = find_latest_summary(FEATURES_DIR, playlist, id=summary_id)

    if summary_csv is None:
        sys.exit(f"[ERROR] no se encontro summary para {playlist} en {FEATURES_DIR}")

    return prepare_summary(summary_csv, TRACKS_XLSX)

def train_tree(df: pd.DataFrame, kind: str = "random_forest") -> TreeClassifier:
    
    X = df[FEATURES].fillna(0)
    y = df["categorico"].values

    model = TreeClassifier(kind=kind, features=FEATURES)
    cv = Evaluator.cross_validate(model, X, y)

    logger.info(f"{kind}: balanced_accuracy = {cv['mean']*100:.1f}% +/- {cv['std']*100:.1f} (k={cv['k']})")

    model.fit(X, y)

    if kind == "random_forest":
        for feat, imp in model.feature_importances().items():
            logger.info(f"  {feat:20s}: {imp:.3f}")

    return model

def train_linear(df: pd.DataFrame) -> LinearRuleClassifier:

    model = ModelCalibrator().calibrate(df)

    logger.info(f"linear_rule: balanced_accuracy (train) = {model.calibration_score*100:.1f}%")

    return model


def run(args: argparse.Namespace) -> None:

    df = load_dataset(summary_id=args.summary_id)
    logger.info(f"Dataset: {len(df)} canciones etiquetadas")

    models = {}

    if args.model in ("random_forest", "all"):
        models["random_forest"] = train_tree(df, kind="random_forest")

    if args.model in ("decision_tree", "all"):
        models["decision_tree"] = train_tree(df, kind="decision_tree")

    if args.model in ("linear_rule", "all"):
        models["linear_rule"] = train_linear(df)

    ML_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.save(ML_ARTIFACTS_DIR / f"{name}.joblib")

    if args.learning_curve:
        X = df[FEATURES].fillna(0)
        y = df["categorico"].values

        curve = Evaluator.learning_curve(TreeClassifier(kind="decision_tree", features=FEATURES), X, y)

        logger.info("\n" + curve.to_string(index=False))

    if "random_forest" in models:
        router = ConfidenceRouter(models["random_forest"]).fit_thresholds(df[FEATURES])
        routed = router.route(df[FEATURES], linear_model=models.get("linear_rule"))
        routed["music_name"] = df["music_name"].values
        routed["category_human"] = df["categorico"].values

        ML_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = ML_RUNS_DIR / f"confidence_routing_{args.summary_id or 'latest'}.csv"
        routed.to_csv(out_csv, sep=";", index=False, encoding="utf-8")

        pct_agreement, matrix = Evaluator.confusion_report(routed["final_category"], routed["category_human"])
        
        logger.info(f"Acuerdo final (con desempate) vs humano: {pct_agreement:.1f}%")
        logger.info("\n" + matrix.to_string())
        logger.info(f"CSV: {out_csv}")

        tiers = routed["tier"].value_counts()

        for tier in ["confiable", "ambigua", "revisar"]:
            n = tiers.get(tier, 0)
            
            logger.info(f"  {tier:12s}: {n:3d}  ({100*n/len(routed):.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["random_forest", "decision_tree", "linear_rule", "all"], default="all")
    parser.add_argument("--summary-id", type=int, default=None)
    parser.add_argument("--learning-curve", action="store_true")

    args = parser.parse_args()

    run(args)

if __name__ == "__main__":
    main()
