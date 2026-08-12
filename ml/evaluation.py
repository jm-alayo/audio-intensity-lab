from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import learning_curve as sk_learning_curve

from ml.models import CATEGORIES

def _unwrap(model):
    return model.model if hasattr(model, "model") else model

class Evaluator:

    @staticmethod
    def safe_k_folds(labels: np.ndarray, max_k: int = 5) -> int:
        _, counts = np.unique(labels, return_counts=True)

        if counts.min() < 2:
            raise ValueError(f"class with fewer than 2 examples: {counts.min()}")

        return min(max_k, int(counts.min()))

    @staticmethod
    def cross_validate(model, X, y, k: int = None, scoring: str = "balanced_accuracy") -> dict:

        k = k or Evaluator.safe_k_folds(y)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(_unwrap(model), X, y, cv=skf, scoring=scoring)

        return {
            "mean": scores.mean(), 
            "std": scores.std(), 
            "k": k,
        }

    @staticmethod
    def learning_curve(model, X, y, train_sizes: np.ndarray = None, k: int = 5) -> pd.DataFrame:

        train_sizes = train_sizes if train_sizes is not None else np.linspace(0.3, 1.0, 8)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        sizes, train_scores, test_scores = sk_learning_curve(
            _unwrap(model), X, y,
            train_sizes=train_sizes, 
            cv=skf, 
            scoring="balanced_accuracy", 
            shuffle=True, 
            random_state=42
        )

        return pd.DataFrame({
            "n_songs": sizes,
            "train_balanced_accuracy": train_scores.mean(axis=1),
            "test_balanced_accuracy": test_scores.mean(axis=1),
        })

    @staticmethod
    def accuracy_by_category(df: pd.DataFrame, predictions, label_col: str = "categorico") -> pd.DataFrame:

        tmp = df.assign(prediction=predictions, correct=lambda d: d["prediction"] == d[label_col])
        summary = tmp.groupby(label_col)["correct"].agg(n="size", accuracy="mean")
        summary["accuracy"] = (summary["accuracy"] * 100).round(1)

        return summary.reindex(CATEGORIES)

    @staticmethod
    def confusion_report(predicted: pd.Series, human: pd.Series, categories: list[str] = CATEGORIES) -> tuple[float, pd.DataFrame]:

        mask = (human.astype(str).str.strip() != "") & (predicted.astype(str).str.strip() != "")
        predicted, human = predicted[mask], human[mask]
        total = len(predicted)

        if total == 0:
            return None, pd.DataFrame()

        agreements = int((predicted.values == human.values).sum())
        confusion = Counter(zip(predicted, human))

        matrix = pd.DataFrame(
            [
                [confusion.get((s, h), 0) 
                    for h in categories] 
                        for s in categories
            ],
            index=categories, 
            columns=categories
        )
        
        matrix.index.name = "predicted \\ human"

        return 100 * agreements / total, matrix
