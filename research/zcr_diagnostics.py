import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.settings import FEATURES_DIR
from shared.utils import find_latest_summary

MAX_SEGS = 25
cols = ["feature", "music_name"] + [f"seg_{i}" for i in range(1, MAX_SEGS + 1)]

features_csv = find_latest_summary(FEATURES_DIR, "rock_english").with_name(
    find_latest_summary(FEATURES_DIR, "rock_english").name.replace("_summary_", "_features_")
)

df = pd.read_csv(
    features_csv,
    sep=";",
    header=None,
    skiprows=1,
    names=cols,
    engine="python",
)

print(f"Archivo: {features_csv.name}")

values = defaultdict(list)
for row in df.itertuples(index=False):
    if row[0] not in ("zcr", "flatness"):
        continue
    for v in row[2:]:
        if pd.notna(v):
            values[row[0]].append(float(v))

for feat in ("zcr", "flatness"):
    vals = np.array(values[feat])
    p = np.percentile(vals, [1, 5, 25, 50, 75, 90, 95, 99])
    print(f"\n{feat} (n={len(vals)}):")
    print(f"  p1={p[0]:.4f} p5={p[1]:.4f} p25={p[2]:.4f} p50={p[3]:.4f} "
          f"p75={p[4]:.4f} p90={p[5]:.4f} p95={p[6]:.4f} p99={p[7]:.4f}")
    print(f"  mediana cerca del piso p5? {'SI -- distribucion sesgada' if (p[3]-p[1]) < (p[6]-p[3])*0.3 else 'no'}")
