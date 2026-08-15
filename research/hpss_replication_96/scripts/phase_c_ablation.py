import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from shared.settings import FEATURES_DIR, TRACKS_XLSX
from shared.utils import find_latest_summary, prepare_summary, percentile, normalize, aggregate_median
from features.extract import FEATURES
from ml.models import TreeClassifier
from ml.evaluation import Evaluator

EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"
PERC_SEGMENT_CSV = EVIDENCE_DIR / "phase_b_kernel_validation" / "perc_ratio_per_segment.csv"
IMPORTANCE_OUT_CSV = EVIDENCE_DIR / "phase_c_ablation" / "feature_importance_rf_with_perc.csv"
CV_FOLDS_OUT_CSV = EVIDENCE_DIR / "phase_c_ablation" / "ablation_cv_folds.csv"
CV_SUMMARY_OUT_CSV = EVIDENCE_DIR / "phase_c_ablation" / "ablation_summary.csv"
PERC_BOUNDS_OUT_CSV = EVIDENCE_DIR / "phase_c_ablation" / "perc_ratio_bounds.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_perc_feature() -> pd.DataFrame:

    seg = pd.read_csv(PERC_SEGMENT_CSV, sep=";")

    raw_values = seg["perc_ratio_k31"].tolist()
    lo, hi = percentile(raw_values, 5), percentile(raw_values, 95)

    pd.DataFrame([{"lo_p5": lo, "hi_p95": hi, "n_segmentos": len(raw_values)}]).to_csv(
        PERC_BOUNDS_OUT_CSV, sep=";", index=False, encoding="utf-8"
    )
    logger.info(f"PERC_LO={lo:.4f} PERC_HI={hi:.4f} (n={len(raw_values)} segmentos)")

    per_song = seg.groupby("music_name")["perc_ratio_k31"].apply(lambda v: aggregate_median(v.tolist())).reset_index()
    per_song = per_song.rename(columns={"perc_ratio_k31": "perc_ratio_mediana"})
    per_song["perc_n_mediana"] = per_song["perc_ratio_mediana"].apply(lambda v: normalize(v, lo, hi, kind="linear"))

    return per_song


def cross_validate_folds(model, X, y, k: int = 5) -> list[float]:

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    return cross_val_score(model.model, X, y, cv=skf, scoring="balanced_accuracy").tolist()


def main():

    summary_csv = find_latest_summary(FEATURES_DIR, "rock_english")
    df = prepare_summary(summary_csv, TRACKS_XLSX)
    logger.info(f"Dataset: {len(df)} canciones etiquetadas")

    perc_df = build_perc_feature()
    df = df.merge(perc_df, on="music_name", how="inner")
    logger.info(f"Con perc_ratio recalculado: {len(df)} canciones")

    features_with = FEATURES + ["perc_n_mediana"]
    y = df["categorico"].values

    fold_rows = []
    summary_rows = []

    for kind in ["decision_tree", "random_forest"]:
        for label, feats in [("sin_perc_ratio", FEATURES), ("con_perc_ratio", features_with)]:
            X = df[feats].fillna(0)
            model = TreeClassifier(kind=kind, features=feats)
            scores = cross_validate_folds(model, X, y)

            for i, s in enumerate(scores):
                fold_rows.append({"modelo": kind, "features": label, "fold": i, "balanced_accuracy": s})

            mean, std = float(np.mean(scores)), float(np.std(scores))
            ci_half = 2 * std
            summary_rows.append({
                "modelo": kind, "features": label,
                "mean": round(mean, 4), "std": round(std, 4),
                "ci_lo": round(mean - ci_half, 4), "ci_hi": round(mean + ci_half, 4),
            })
            logger.info(f"{kind} | {label}: {mean*100:.1f}% +/- {std*100:.1f}")

    pd.DataFrame(fold_rows).to_csv(CV_FOLDS_OUT_CSV, sep=";", index=False, encoding="utf-8")
    pd.DataFrame(summary_rows).to_csv(CV_SUMMARY_OUT_CSV, sep=";", index=False, encoding="utf-8")

    rf_model = TreeClassifier(kind="random_forest", features=features_with)
    rf_model.fit(df[features_with].fillna(0), y)

    importances = rf_model.feature_importances()
    pd.DataFrame(
        [{"feature": f, "importance": imp} for f, imp in importances.items()]
    ).to_csv(IMPORTANCE_OUT_CSV, sep=";", index=False, encoding="utf-8")

    for f, imp in importances.items():
        logger.info(f"  {f:20s}: {imp:.3f}")

    logger.info(f"Evidencia -> {IMPORTANCE_OUT_CSV}, {CV_FOLDS_OUT_CSV}, {CV_SUMMARY_OUT_CSV}, {PERC_BOUNDS_OUT_CSV}")


if __name__ == "__main__":
    main()
