import pandas as pd
from src.monitoring import DriftMonitor


def test_drift_monitor():
    ref = pd.DataFrame({"x": range(100)})
    cur = pd.DataFrame({"x": range(10, 110)})
    out = DriftMonitor().compare(ref, cur, ["x"])
    assert list(out.columns) == ["feature", "psi", "ks_pvalue"]
    assert out.iloc[0]["psi"] >= 0
