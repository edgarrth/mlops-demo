import argparse
import json
from pathlib import Path
import pandas as pd

from src.config import load_config
from src.data import NUMERIC
from src.monitoring import DriftMonitor

parser = argparse.ArgumentParser()
parser.add_argument("--month", type=int, default=201509)
args = parser.parse_args()

config = load_config()
df = pd.read_csv(config["data"]["path"], sep=";")
reference = df[df["MES"] <= config["data"]["train_end"]]
current = df[df["MES"] == args.month]
report = DriftMonitor().compare(reference, current, NUMERIC)
Path("reports/monitoring").mkdir(parents=True, exist_ok=True)
report.to_csv("reports/monitoring/drift.csv", index=False)
summary = {
    "month": args.month,
    "features": int(len(report)),
    "psi_warning_or_more": int((report["psi"] >= config["monitoring"]["psi_warning"]).sum()),
    "psi_critical": int((report["psi"] >= config["monitoring"]["psi_critical"]).sum()),
}
Path("reports/monitoring/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

# Evidently es opcional: si está instalado, se genera un HTML adicional.
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    report_html = Report([DataDriftPreset()])
    result = report_html.run(reference_data=reference[NUMERIC], current_data=current[NUMERIC])
    result.save_html("reports/monitoring/evidently.html")
    print("Reporte Evidently generado")
except Exception as exc:
    print(f"Evidently no disponible o incompatible; PSI/KS sí fueron calculados: {exc}")
