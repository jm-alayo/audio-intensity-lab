import sys
from pathlib import Path
import logging
import librosa
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.settings import FEATURES_DIR

SPEARMAN_SEGURO = 0.98
SPEARMAN_LIMITE = 0.95
DESVIO_MEDIO_OK = 2.0
DESVIO_MAX_OK = 10.0

logger = logging.getLogger(__name__)

def current_perc_ratio(seg: np.ndarray) -> float:

    yh, yp = librosa.effects.hpss(seg)
    e_h, e_p = float(np.sum(yh ** 2)), float(np.sum(yp ** 2))

    return e_p / (e_h + e_p + 1e-9)

def fast_perc_ratio(seg: np.ndarray, kernel: int = 9) -> float:
    
    S = np.abs(librosa.stft(seg, n_fft=2048))
    H, P = librosa.decompose.hpss(S, kernel_size=kernel)
    e_h, e_p = float(np.sum(H ** 2)), float(np.sum(P ** 2))

    return e_p / (e_h + e_p + 1e-9)

def interpret(df: pd.DataFrame, kernel: int) -> None:
    
    a = df["perc_ratio_k31"].values
    b = df[f"perc_ratio_k{kernel}"].values

    rho, _ = spearmanr(a, b)
    r, _ = pearsonr(a, b)
    desvio = 100 * (b - a) / np.maximum(np.abs(a), 1e-9)

    desvio_medio_abs = float(np.mean(np.abs(desvio)))
    desvio_max_abs = float(np.max(np.abs(desvio)))
    sesgo = float(np.mean(desvio))
    mismo_signo = float(np.mean(np.sign(desvio) == np.sign(sesgo)))

    logger.info("\n" + "=" * 72)
    logger.info(f"  RESULTADO — kernel_size={kernel} vs actual (kernel 31)")
    logger.info("=" * 72)

    logger.info(f"\n1) CORRELACION DE SPEARMAN = {rho:.4f}")

    if rho >= SPEARMAN_SEGURO:
        logger.info(f"   -> OK: >= {SPEARMAN_SEGURO}. El orden se conserva.")

    elif rho >= SPEARMAN_LIMITE:
        logger.info(f"   -> LIMITE: entre {SPEARMAN_LIMITE} y {SPEARMAN_SEGURO}. Revisar kernel=17.")

    else:
        logger.info(f"   -> MAL: < {SPEARMAN_LIMITE}. El orden cambia, NO usar este kernel.")

    logger.info(f"\n2) DESVIO RELATIVO: medio {desvio_medio_abs:.2f}%  |  maximo {desvio_max_abs:.2f}%")

    mask_rotos = np.abs(desvio) > DESVIO_MAX_OK
    cant_rotos = int(np.sum(mask_rotos))
    total_segmentos = len(desvio)
    porc_rotos = 100 * cant_rotos / total_segmentos

    logger.info(f"   -> {cant_rotos} de {total_segmentos} segmentos (={porc_rotos:.1f}%) tienen desvio > {DESVIO_MAX_OK}%.")
    
    if desvio_medio_abs <= DESVIO_MEDIO_OK and desvio_max_abs <= DESVIO_MAX_OK:
        logger.info(f"   -> OK: medio <= {DESVIO_MEDIO_OK}% y maximo <= {DESVIO_MAX_OK}%.")

    else:
        logger.info(f"   -> ATENCION: supera el umbral (medio {DESVIO_MEDIO_OK}%, max {DESVIO_MAX_OK}%).")


    logger.info(f"\n3) SESGO SISTEMATICO = {sesgo:+.2f}%  ({100*mismo_signo:.0f}% de los segmentos van en esa direccion)")

    if mismo_signo >= 0.90:
        logger.info("   -> Sesgo sistematico y consistente: inocuo, absorbible recalibrando.")

    elif mismo_signo >= 0.70:
        logger.info("   -> Parcialmente sistematico.")

    else:
        logger.info("   -> ERRATICO: el desvio alterna de signo. Esto es ruido, no un offset.")

    df_tmp = df.copy()
    df_tmp["desvio_%"] = desvio
    peores = df_tmp.reindex(df_tmp["desvio_%"].abs().sort_values(ascending=False).index).head(5)

    logger.info("\n4) PEORES 5 SEGMENTOS (mirar si son casos raros: outros, silencios, fades)")
    logger.info(peores[["filename", "segmento", "perc_ratio_k31", f"perc_ratio_k{kernel}", "desvio_%"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ok_orden  = rho >= SPEARMAN_SEGURO
    ok_desvio = desvio_medio_abs <= DESVIO_MEDIO_OK and desvio_max_abs <= DESVIO_MAX_OK
    ok_sesgo  = mismo_signo >= 0.70

    logger.info("\n" + "-" * 72)

    if ok_orden and ok_desvio:
        logger.info("  VEREDICTO: APLICAR el cambio. No hace falta recalibrar PERC_LO/PERC_HI.")
        logger.info("  Siguiente paso: correr classify.py con ambas versiones y confirmar")
        logger.info("  que ninguna cancion cambia de categoria.")

    elif ok_orden and ok_sesgo:
        logger.info("  VEREDICTO: APLICABLE, pero SI hay que recalibrar PERC_LO/PERC_HI")
        logger.info("  (el orden se conserva pero la escala se movio de forma sistematica).")

    elif rho >= SPEARMAN_LIMITE:
        logger.info("  VEREDICTO: DUDOSO con este kernel. Probar kernel_size=17, que sigue")
        logger.info("  dando ~2x de speedup con menos distorsion.")

    else:
        logger.info("  VEREDICTO: NO aplicar. El orden entre segmentos no se conserva.")

    logger.info("-" * 72)

def _perc_ratio_long(df: pd.DataFrame, nombre_col: str) -> pd.DataFrame:

    df = df[df["feature"] == "perc_ratio"]
    
    seg_cols = [c for c in df.columns if c.startswith("seg_")]

    largo = df.melt(id_vars=["filename"], value_vars=seg_cols, var_name="segmento", value_name=nombre_col)

    return largo.dropna(subset=[nombre_col])

def main():

    KERNEL_SIZE = 9

    df_kbase = pd.read_csv(FEATURES_DIR / "_features_10_rock_english_segmin25.csv", encoding="utf-8-sig", sep=";")
    df5_k31 = pd.read_csv(FEATURES_DIR / "_features_5_rock_english_segmin25.csv", encoding="utf-8-sig", sep=";")

    largo_kbase = _perc_ratio_long(df_kbase, f"perc_ratio_k{KERNEL_SIZE}")
    largo_k31 = _perc_ratio_long(df5_k31, "perc_ratio_k31")

    df = largo_k31.merge(largo_kbase, on=["filename", "segmento"])

    interpret(df, KERNEL_SIZE)

if __name__ == "__main__":
    main()