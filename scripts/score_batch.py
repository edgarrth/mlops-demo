import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.data import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", default="artifacts/campaign_ranking.csv")
args = parser.parse_args()

bundle = joblib.load("artifacts/model.joblib")
df = pd.read_csv(args.input, sep=";")
scores = bundle["model"].predict_proba(df[FEATURES])[:, 1]
out = df[["MES", "CLIENTE"]].copy()
out["score_renovacion"] = scores
out = out.sort_values("score_renovacion", ascending=False)
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(args.output, index=False)
print(f"Ranking generado: {args.output} ({len(out)} clientes)")
