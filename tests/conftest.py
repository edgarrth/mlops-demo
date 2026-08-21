import pandas as pd
import pytest
from src.data import EXPECTED_COLUMNS


@pytest.fixture
def tiny_df():
    rows = []
    for i in range(60):
        month = 201506 if i < 30 else (201507 if i < 45 else 201509)
        sale = 1 if i % 10 == 0 else 0
        row = {c: 1 for c in EXPECTED_COLUMNS}
        row.update({
            "MES": month, "CLIENTE": i + 1, "FLAG_VENTA": sale,
            "SEXO": "M" if i % 2 else "F", "EST_CIVIL": "S",
            "REGION": "LIMA NORTE", "EDAD": 20 + i % 30,
            "LINEA_RENOVADO": 5000 + i * 10, "PLAZO_RENOVADO": 12,
        })
        rows.append(row)
    return pd.DataFrame(rows)
