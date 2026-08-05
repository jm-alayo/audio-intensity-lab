import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
TIME_LOG_CSV = SCRIPT_DIR / "files" / "registro_tiempo.csv"

def analyze_execution_times():
    df = pd.read_csv(
        TIME_LOG_CSV,
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