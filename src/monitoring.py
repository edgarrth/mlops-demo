from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DriftMonitor:
    """Calcula PSI y KS para variables numéricas de referencia vs. producción."""

    def compare(self, reference: pd.DataFrame, current: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
        rows = []
        for col in numeric_columns:
            ref = pd.to_numeric(reference[col], errors="coerce").dropna()
            cur = pd.to_numeric(current[col], errors="coerce").dropna()
            ks_pvalue = float(ks_2samp(ref, cur).pvalue) if len(ref) and len(cur) else 1.0
            rows.append({"feature": col, "psi": psi(ref, cur), "ks_pvalue": ks_pvalue})
        return pd.DataFrame(rows).sort_values("psi", ascending=False)
