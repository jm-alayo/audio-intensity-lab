from collections import Counter
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

class MLFunc:

    @staticmethod
    def prepare_xy(df: pd.DataFrame, features: list[str], label_col: str = "categoria") -> tuple[np.ndarray, np.ndarray]:

        df = df.dropna(subset=[label_col])

        X = df[features].fillna(0).values
        y = df[label_col].values

        return X, y

    @staticmethod
    def safe_k_folds(categories: np.ndarray, max_k: int = 5) -> int:
    
        _, counts = np.unique(categories, return_counts=True)

        if counts.min() < 2:
            raise ValueError(f"Clase con menos ejemplos: {counts.min()} -- se necesitan >= 2 por clase.")

        return min(max_k, int(counts.min()))

    @staticmethod
    def evaluate_cv(model, X: np.ndarray, y: np.ndarray, k: int = 5, scoring: str = "balanced_accuracy") -> tuple[float, float]:
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)

        return scores.mean(), scores.std()

    @staticmethod
    def accuracy_by_category(df: pd.DataFrame, preds, categories_order: list[str] = None, label_col: str = "categoria") -> pd.DataFrame:
        
        tmp = df.assign(pred=preds, ok=lambda d: d["pred"] == d[label_col])
        resumen = tmp.groupby(label_col)["ok"].agg(n="size", accuracy="mean")
        resumen["accuracy"] = (resumen["accuracy"] * 100).round(1)

        return resumen.reindex(categories_order) if categories_order else resumen

    @staticmethod
    def rank_probabilities(model, x: list[float]) -> dict:

        proba = model.predict_proba([x])[0]
        orden = sorted(zip(model.classes_, proba), key=lambda t: -t[1])

        top1_cat, top1_p = orden[0]
        top2_cat, top2_p = orden[1] if len(orden) > 1 else (None, 0.0)

        return {
            "orden": orden,
            "top1_categoria": top1_cat, "top1_prob": round(float(top1_p), 3),
            "top2_categoria": top2_cat, "top2_prob": round(float(top2_p), 3),
            "margen": round(float(top1_p - top2_p), 3),
        }

    @staticmethod
    def save_model(path: Path, model, features: list[str]) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({"modelo": model, "features": features}, path)

    @staticmethod
    def load_model(path: Path) -> tuple:

        paquete = joblib.load(path)

        return paquete["modelo"], paquete["features"]

    @staticmethod
    def validate_predictions(classified_csv: Path, categorias: list[str]) -> float | None:
        
        df = pd.read_csv(classified_csv, sep=";", encoding="utf-8", dtype=str).fillna("")

        df["categoria_humano"]   = df["categoria_humano"].str.strip()
        df["categoria_sugerida"] = df["categoria_sugerida"].str.strip()
        df = df[(df["categoria_humano"] != "") & (df["categoria_sugerida"] != "")]

        total = len(df)
        if total == 0:
            logger.warning("No hay filas con categoria_humano rellena.")
            return None

        acuerdos  = int((df["categoria_humano"] == df["categoria_sugerida"]).sum())
        confusion = Counter(zip(df["categoria_sugerida"], df["categoria_humano"]))
        pct = 100 * acuerdos / total

        matriz = pd.DataFrame(
            [[confusion.get((s_cat, h_cat), 0) for h_cat in categorias] for s_cat in categorias],
            index=categorias, columns=categorias
        )
        matriz.index.name = "sugerida \\ humano"

        logger.info(f"\nValidacion sobre {total} canciones etiquetadas manualmente:")
        logger.info(f"  Acuerdo sistema vs humano: {acuerdos}/{total} = {pct:.1f}%")
        logger.info("\n  Matriz de confusion (sistema -> humano):")
        logger.info(matriz.to_string())

        return pct
