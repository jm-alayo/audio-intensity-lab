import pandas as pd
import re
import sys
import logging
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.settings import FEATURES_DIR
from shared.utils import percentile
from features.config import (
    ONSET_LO, ONSET_HI, TEMPO_LO, TEMPO_HI, CENT_LO, CENT_HI,
    OSTR_LO, OSTR_HI, FLAT_LO, FLAT_HI, ZCR_LO, ZCR_HI,
    DYN_LO, DYN_HI, LRA_LO, LRA_HI,
)

FEATURE_BOUNDS = {
    "onset_density":       (ONSET_LO, ONSET_HI),
    "onset_strength_mean": (OSTR_LO, OSTR_HI),
    "bpm_folded":          (TEMPO_LO, TEMPO_HI),
    "centroid_hz":         (CENT_LO, CENT_HI),
    "flatness":            (FLAT_LO, FLAT_HI),
    "zcr":                 (ZCR_LO, ZCR_HI),
    "dyn_spread":          (DYN_LO, DYN_HI),
    "lra":                 (LRA_LO, LRA_HI),
}

RAW_FEATURES = list(FEATURE_BOUNDS.keys())

DEVIATION_ALERT_RATIO = 0.3
MIN_RELIABLE_SAMPLES = 10

logger = logging.getLogger(__name__)

def load_values_by_feature(features_csv: Path, features_names: list[str] = None) -> dict[str, list[float]]:

    values = defaultdict(list)

    df = pd.read_csv(features_csv, sep=";", encoding="utf-8", engine="python", on_bad_lines="skip")

    if features_names is None:
        features_names = df.iloc[:, 0].unique().tolist()

    for feature_name in features_names:
        values[feature_name].extend(
            pd.to_numeric(
                df.loc[df.iloc[:, 0] == feature_name, df.columns[2:]].stack(),
                errors="coerce"
            ).dropna().tolist()
        )

    return values

def calibrate_ranges(features_csv: Path, p_lo: float = 5, p_hi: float = 95) -> dict[str, tuple]:

    feature_data = load_values_by_feature(features_csv, RAW_FEATURES)

    ranges = {}

    for feat in RAW_FEATURES:
        vals = feature_data.get(feat, [])

        if len(vals) < MIN_RELIABLE_SAMPLES:
            logger.warning(f"  [WARNING] {feat}: only {len(vals)} values -- too few for a reliable percentile")

        lo, hi = percentile(vals, p_lo), percentile(vals, p_hi)

        ranges[feat] = (round(lo, 6), round(hi, 6), len(vals))

    return ranges

def find_latest_features_csv(out_dir: Path, playlist: str) -> Path | None:

    pattern = re.compile(rf"^_features_(\d+)_{re.escape(playlist)}_segmin\d+\.csv$")
    candidates = [
        (int(m.group(1)), f) for f in out_dir.glob(f"_features_*_{playlist}_segmin*.csv")
            if (m := pattern.match(f.name))
    ]

    return max(candidates, key=lambda t: t[0])[1] if candidates else None

if __name__ == "__main__":

    playlist = "rock_english"
    features_csv = find_latest_features_csv(FEATURES_DIR, playlist)

    if features_csv is None:
        raise SystemExit(f"[ERROR] No _features_*_{playlist}_segmin*.csv found in {FEATURES_DIR}")

    ranges = calibrate_ranges(features_csv)

    report = pd.DataFrame([
        {
            "feature": feat,
            "version": version,
            "lo_current": lo_cur,
            "hi_current": hi_cur,
            "lo_real_p5": lo_real,
            "hi_real_p95": hi_real,
            "n_segments": n,
        }
        for feat, (lo_real, hi_real, n) in ranges.items()
        for version, (lo_cur, hi_cur) in enumerate(zip(*FEATURE_BOUNDS[feat]))
    ])

    deviation_lo = (report["lo_real_p5"] - report["lo_current"]).abs() / report["lo_current"].abs().clip(lower=1e-9)
    deviation_hi = (report["hi_real_p95"] - report["hi_current"]).abs() / report["hi_current"].abs().clip(lower=1e-9)
    report["alert"] = ((deviation_lo > DEVIATION_ALERT_RATIO) | (deviation_hi > DEVIATION_ALERT_RATIO)).map(
        {True: "REVIEW, significant difference", False: ""}
    )

    logger.info("\n" + report.to_string(index=False, formatters={
        "lo_current":   "{:.4f}".format,
        "hi_current":   "{:.4f}".format,
        "lo_real_p5":   "{:.4f}".format,
        "hi_real_p95":  "{:.4f}".format,
    }))
