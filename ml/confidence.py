import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.utils import percentile
from ml.models import TreeClassifier, LinearRuleClassifier

class ConfidenceRouter:

    def __init__(self, tree_model: TreeClassifier, confident_pct: float = None, ambiguous_gap: float = None):
        self.tree_model = tree_model
        self.confident_pct = confident_pct
        self.ambiguous_gap = ambiguous_gap

    def fit_thresholds(self, X: pd.DataFrame, p_confident: float = 60, p_ambiguous: float = 25) -> "ConfidenceRouter":
        
        proba = self.tree_model.predict_proba(X)
        sorted_proba = np.sort(proba, axis=1)[:, ::-1]
        top1, top2 = sorted_proba[:, 0], sorted_proba[:, 1]
        margins = top1 - top2

        self.confident_pct = percentile(top1.tolist(), p_confident)
        self.ambiguous_gap = percentile(margins.tolist(), p_ambiguous)

        return self

    def route(self, X: pd.DataFrame, linear_model: LinearRuleClassifier = None) -> pd.DataFrame:

        if self.confident_pct is None or self.ambiguous_gap is None:
            raise RuntimeError("call fit_thresholds() before route()")

        proba = self.tree_model.predict_proba(X)
        classes = self.tree_model.classes_
        order = np.argsort(-proba, axis=1)

        linear_votes = linear_model.predict(X) if linear_model is not None else None

        rows = []
        for i in range(len(X)):
            top1_idx, top2_idx = order[i, 0], order[i, 1]
            top1_cat, top1_p = classes[top1_idx], float(proba[i, top1_idx])
            top2_cat, top2_p = classes[top2_idx], float(proba[i, top2_idx])
            margin = top1_p - top2_p

            if margin < self.ambiguous_gap:
                tier = "ambigua"

            elif top1_p >= self.confident_pct:
                tier = "confiable"

            else:
                tier = "revisar"

            final_category = top1_cat
            tie_break = ""

            if tier == "ambigua" and linear_votes is not None:
                vote = linear_votes[i]

                if vote == top1_cat:
                    tie_break, final_category = "confirma_top1", top1_cat

                elif vote == top2_cat:
                    tie_break, final_category = "vota_top2", top2_cat

                else:
                    tie_break, final_category = "sin_acuerdo", top1_cat

            rows.append({
                "tier": tier,
                "top1_category": top1_cat, "top1_prob": round(top1_p, 3),
                "top2_category": top2_cat, "top2_prob": round(top2_p, 3),
                "margin": round(margin, 3),
                "tie_break": tie_break, "final_category": final_category,
            })

        return pd.DataFrame(rows, index=X.index)
