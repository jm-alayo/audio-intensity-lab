import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils import derive_threshold_bounds
from ml.models import LinearRuleClassifier

class ModelCalibrator:

    def __init__(self, agg: str = "mediana", seed: int = 42, maxiter: int = 200, popsize: int = 25):
        self.agg = agg
        self.seed = seed
        self.maxiter = maxiter
        self.popsize = popsize

    def _build_model(self, params: np.ndarray) -> LinearRuleClassifier:
        return LinearRuleClassifier(
            agg=self.agg,
            weights_energy=params[:5],
            weights_rhythm=params[5:7],
            thr_energy=params[7], 
            thr_rhythm=params[8], 
            thr_noise=params[9],
            thr_change=params[10], 
            thr_delta=params[11],
        )

    def _objective(self, params: np.ndarray, df: pd.DataFrame, label_col: str) -> float:

        predictions = self._build_model(params).predict(df)
        labels = df[label_col].values

        recalls = [(predictions[labels == cat] == cat).mean() for cat in np.unique(labels)]

        return -float(np.mean(recalls))

    def calibrate(self, df: pd.DataFrame, label_col: str = "categorico") -> LinearRuleClassifier:

        weight_bounds = [(0.01, 1.0)] * 7
        threshold_bounds = derive_threshold_bounds(df)
        bounds = weight_bounds + threshold_bounds

        result = differential_evolution(
            self._objective, 
            bounds, 
            args=(df, label_col),
            seed=self.seed, 
            maxiter=self.maxiter, 
            popsize=self.popsize,
            strategy="best1bin", 
            tol=1e-4, 
            workers=1
        )

        model = self._build_model(result.x)

        model.calibration_score = -result.fun

        return model
