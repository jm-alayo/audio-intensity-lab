import sys
import pandas as pd
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.settings import FEATURES_DIR

WORKER_BENCHMARK_CSV = FEATURES_DIR / "time" / "registro_tiempo_original.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_execution_times():
    df = pd.read_csv(
        WORKER_BENCHMARK_CSV,
        sep=";",
        encoding="cp1252",
        keep_default_na=False
    )

    summary = (
        df.groupby("stage", as_index=False)
        .agg(
            total_execution_time_s=("execution_time_s", "sum"),
            avg_hpss_time_s=("hpss_time_s", "mean"),
            avg_margin_s=("margin_s", "mean"),
            total_execution_s=("total_execution_s", "first"),
        )
    )

    summary["ratio_overhead"] = (
        summary["avg_margin_s"] / summary["avg_hpss_time_s"] * 100
    )

    summary["time_per_audio"] = (
        summary["total_execution_time_s"] / summary["total_execution_s"]
    )

    summary = summary.sort_values(
        by="time_per_audio",
        ascending=False
    ).round(2)

    summary = summary[["stage", "total_execution_time_s", "avg_hpss_time_s", "avg_margin_s", "ratio_overhead", "total_execution_s", "time_per_audio"]]

    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    analyze_execution_times()