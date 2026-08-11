import pandas as pd
from collections import defaultdict
from pathlib import Path

MAX_SEGS = 25
SCRIPT_DIR = Path(__file__).parent
id = 10
cols = ["feature", "filename"] + [f"seg_{i}" for i in range(1, MAX_SEGS + 1)]

df = pd.read_csv(
    SCRIPT_DIR / "out" / f"_features_{id}_rock_english_segmin25.csv",
    sep=";",
    header=None,
    skiprows=1,
    names=cols,
    engine="python",
)

print(f"Archivo: {f'_features_{id}_rock_english_segmin25.csv'}")

valores = defaultdict(list)
for row in df.itertuples(index=False):
    if row[0] not in ("zcr", "flatness"):
        continue
    for v in row[2:]:
        if pd.notna(v):
            valores[row[0]].append(float(v))

import numpy as np
for feat in ("zcr", "flatness"):
    vals = np.array(valores[feat])
    p = np.percentile(vals, [1, 5, 25, 50, 75, 90, 95, 99])
    print(f"\n{feat} (n={len(vals)}):")
    print(f"  p1={p[0]:.4f} p5={p[1]:.4f} p25={p[2]:.4f} p50={p[3]:.4f} "
          f"p75={p[4]:.4f} p90={p[5]:.4f} p95={p[6]:.4f} p99={p[7]:.4f}")
    print(f"  ¿mediana cerca del piso p5? {'SI -- distribucion sesgada' if (p[3]-p[1]) < (p[6]-p[3])*0.3 else 'no'}")
