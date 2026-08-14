import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SPEARMAN_SEGURO = 0.98
DESVIO_MEDIO_OK = 2.0
DESVIO_MAX_OK = 10.0
KERNELS = [17, 11, 9]

SEGMENT_CSV = Path(__file__).parent / "evidence" / "perc_ratio_per_segment.csv"
OUT_CSV = Path(__file__).parent / "evidence" / "kernel_validation_summary.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_kernel(df: pd.DataFrame, kernel: int) -> dict:

    a = df["perc_ratio_k31"].values
    b = df[f"perc_ratio_k{kernel}"].values

    rho, _ = spearmanr(a, b)
    desvio = 100 * (b - a) / np.maximum(np.abs(a), 1e-9)

    desvio_medio_abs = float(np.mean(np.abs(desvio)))
    desvio_max_abs = float(np.max(np.abs(desvio)))
    sesgo = float(np.mean(desvio))

    n_rotos = int(np.sum(np.abs(desvio) > DESVIO_MAX_OK))
    total = len(desvio)

    return {
        "kernel": kernel,
        "spearman": round(float(rho), 3),
        "desvio_medio_pct": round(desvio_medio_abs, 1),
        "desvio_max_pct": round(desvio_max_abs, 1),
        "segmentos_desvio_gt10": f"{n_rotos}/{total} ({100*n_rotos/total:.0f}%)",
        "sesgo_pct": round(sesgo, 1),
        "pasa_umbral": bool(rho >= SPEARMAN_SEGURO and desvio_medio_abs <= DESVIO_MEDIO_OK and desvio_max_abs <= DESVIO_MAX_OK),
    }


def main():

    df = pd.read_csv(SEGMENT_CSV, sep=";")
    logger.info(f"{len(df)} segmentos de {df['music_name'].nunique()} canciones")

    rows = [evaluate_kernel(df, k) for k in KERNELS]
    result = pd.DataFrame(rows)

    result.to_csv(OUT_CSV, sep=";", index=False, encoding="utf-8")
    logger.info("\n" + result.to_string(index=False))
    logger.info(f"Evidencia -> {OUT_CSV}")


if __name__ == "__main__":
    main()
