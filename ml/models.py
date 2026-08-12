from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

CATEGORIES = ["alta-agresiva", "alta-ritmica", "baja-contemplativa", "baja-ritmica", "incrementable-decreciente"]

class BaseClassifier(ABC):

    @abstractmethod
    def fit(self, X, y) -> "BaseClassifier": ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "BaseClassifier":
        return joblib.load(path)


class TreeClassifier(BaseClassifier):

    def __init__(self, kind: str = "random_forest", features: list[str] = None, **sklearn_kwargs):
        self.kind = kind
        self.features = features

        params = dict(min_samples_leaf=5, class_weight="balanced", random_state=42)
        params.update(sklearn_kwargs)

        if kind == "decision_tree":
            self.model = DecisionTreeClassifier(**params)

        elif kind == "random_forest":
            params.setdefault("n_estimators", 200)
            params.setdefault("max_depth", 6)
            self.model = RandomForestClassifier(**params)

        else:
            raise ValueError(f"unknown kind: {kind!r}")

    def fit(self, X, y) -> "TreeClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self.model.classes_

    def feature_importances(self) -> dict[str, float]:
        return dict(sorted(zip(self.features, self.model.feature_importances_), key=lambda t: -t[1]))


class LinearRuleClassifier(BaseClassifier):

    ENERGY_COMPONENTS = ("flat_n", "zcr_n", "ostr_n", "cent_n", "dyn_n")
    RHYTHM_COMPONENTS = ("onset_n", "tempo_n")

    def __init__(self, agg: str = "mediana",
                 weights_energy: list[float] = None, weights_rhythm: list[float] = None,
                 thr_energy: float = 0.5, thr_rhythm: float = 0.5, thr_noise: float = 0.5,
                 thr_change: float = 0.1, thr_delta: float = 0.12, thr_margin: float = 0.04):
        self.agg = agg
        self.weights_energy = list(weights_energy) if weights_energy is not None else [0.2] * 5
        self.weights_rhythm = list(weights_rhythm) if weights_rhythm is not None else [0.5, 0.5]
        self.thr_energy = thr_energy
        self.thr_rhythm = thr_rhythm
        self.thr_noise = thr_noise
        self.thr_change = thr_change
        self.thr_delta = thr_delta
        self.thr_margin = thr_margin

    def _scores(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        suffix = f"_{self.agg}"
        w_e = np.array(self.weights_energy) / np.sum(self.weights_energy)
        w_r = np.array(self.weights_rhythm) / np.sum(self.weights_rhythm)

        energy = sum(w * df[f"{c}{suffix}"] for w, c in zip(w_e, self.ENERGY_COMPONENTS))
        rhythm = sum(w * df[f"{c}{suffix}"] for w, c in zip(w_r, self.RHYTHM_COMPONENTS))
        noise = 0.5 * df[f"flat_n{suffix}"] + 0.5 * df[f"zcr_n{suffix}"]

        return energy.values, rhythm.values, noise.values

    def _classify_row(self, energy, rhythm, noise, energy_slope, rhythm_slope, rhythm_delta, energy_delta, n_segments) -> dict:

        energy_change = energy_slope * (n_segments - 1)

        if abs(energy_change) >= self.thr_change or abs(energy_delta) >= self.thr_delta:
            margin = max(abs(energy_change) - self.thr_change, abs(energy_delta) - self.thr_delta)
            return {"category": "incrementable-decreciente", "margin": round(float(margin), 3)}

        high_energy = energy >= self.thr_energy
        high_rhythm = rhythm >= self.thr_rhythm

        if high_energy and high_rhythm:
            category = "alta-ritmica"

        elif high_energy:
            category = "alta-agresiva"

        elif high_rhythm:
            category = "baja-ritmica"

        else:
            category = "baja-contemplativa"

        margin = min(
            abs(energy - self.thr_energy), abs(rhythm - self.thr_rhythm), abs(noise - self.thr_noise),
            abs(abs(energy_slope * (n_segments - 1)) - self.thr_change),
            abs(abs(rhythm_slope * (n_segments - 1)) - self.thr_change),
        )
        return {"category": category, "margin": round(float(margin), 3)}

    def fit(self, X, y=None) -> "LinearRuleClassifier":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        energy, rhythm, noise = self._scores(X)

        return np.array([
            self._classify_row(e, r, n, se, sr_, dr, de, ns)["category"]
            for e, r, n, se, sr_, dr, de, ns in zip(
                energy, rhythm, noise,
                X["pendiente_energia"], 
                X["pendiente_ritmo"],
                X["delta_ritmo"], 
                X["delta_energia"], 
                X["n_segments"]
            )
        ])

    def predict_with_margin(self, X: pd.DataFrame) -> pd.DataFrame:

        energy, rhythm, noise = self._scores(X)

        rows = [
            self._classify_row(e, r, n, se, sr_, dr, de, ns)

            for e, r, n, se, sr_, dr, de, ns in zip(
                energy, rhythm, noise,
                X["pendiente_energia"], 
                X["pendiente_ritmo"],
                X["delta_ritmo"], 
                X["delta_energia"], 
                X["n_segments"]
            )
        ]
        
        result = pd.DataFrame(rows, index=X.index)
        result["review"] = result["margin"] < self.thr_margin

        return result
