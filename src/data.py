from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = [
    "MES", "CLIENTE", "LINEA_RENOVADO", "PLAZO_RENOVADO", "FLAG_VENTA",
    "USO_LINEA_TOTAL_TC_T2", "USO_TRIM_LINEA_BBVA", "NR_ENTIDADES_TOTAL_T2",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12", "SDO_CONSUMO_T2",
    "RESENCIA_OFERTA_PLD_RENOVADO", "Ahorro_Sldo_Bco_T1", "PConsumo_Sldo_Bco_T1",
    "SDO_BCO_tot_sm_pasivo_Bco_6M", "EDAD", "SEXO", "EST_CIVIL",
    "ANTIGUEDAD_MES", "REGION", "FLAG_LIMA_PROVINCIA", "SUELDO_ESTIMADO",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
]

FEATURES = [c for c in EXPECTED_COLUMNS if c not in {"MES", "CLIENTE", "FLAG_VENTA"}]
CATEGORICAL = ["SEXO", "EST_CIVIL", "REGION"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]


@dataclass
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class LoanData:
    """Carga, valida y divide el dataset por tiempo."""

    def __init__(self, config: dict):
        self.config = config

    def load(self, path: str | Path | None = None) -> pd.DataFrame:
        path = Path(path or self.config["data"]["path"])
        return pd.read_csv(path, sep=";")

    def validate(self, df: pd.DataFrame) -> dict:
        missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
        if missing_columns:
            raise ValueError(f"Faltan columnas: {missing_columns}")
        if df.empty:
            raise ValueError("El dataset está vacío")
        target = self.config["data"]["target"]
        if not set(df[target].dropna().unique()).issubset({0, 1}):
            raise ValueError("FLAG_VENTA debe ser binaria (0/1)")
        if df[self.config["data"]["id_column"]].duplicated().any():
            raise ValueError("CLIENTE tiene duplicados")
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "positive_rate": float(df[target].mean()),
            "missing_cells": int(df.isna().sum().sum()),
        }

    def temporal_split(self, df: pd.DataFrame) -> DataSplit:
        time_col = self.config["data"]["time_column"]
        train_end = self.config["data"]["train_end"]
        val_months = self.config["data"]["validation_months"]
        test_month = self.config["data"]["test_month"]
        train = df[df[time_col] <= train_end].copy()
        validation = df[df[time_col].isin(val_months)].copy()
        test = df[df[time_col] == test_month].copy()
        if min(len(train), len(validation), len(test)) == 0:
            raise ValueError("Alguna partición temporal quedó vacía")
        return DataSplit(train=train, validation=validation, test=test)
