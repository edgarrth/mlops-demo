from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data import FEATURES

MODEL_PATH = Path("artifacts/model.joblib")
MODEL = None


class PredictionRequest(BaseModel):
    LINEA_RENOVADO: float
    PLAZO_RENOVADO: float
    USO_LINEA_TOTAL_TC_T2: float | None = None
    USO_TRIM_LINEA_BBVA: float | None = None
    NR_ENTIDADES_TOTAL_T2: float
    DIFF_NRO_ENTIDA_TOTALES_T2_T12: float
    SDO_CONSUMO_T2: float | None = None
    RESENCIA_OFERTA_PLD_RENOVADO: float | None = None
    Ahorro_Sldo_Bco_T1: float | None = None
    PConsumo_Sldo_Bco_T1: float | None = None
    SDO_BCO_tot_sm_pasivo_Bco_6M: float | None = None
    EDAD: float
    SEXO: str
    EST_CIVIL: str
    ANTIGUEDAD_MES: float
    REGION: str
    FLAG_LIMA_PROVINCIA: float
    SUELDO_ESTIMADO: float | None = None
    CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD: float | None = None


class PredictionResponse(BaseModel):
    score_renovacion: float
    prediccion: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    if MODEL_PATH.exists():
        MODEL = joblib.load(MODEL_PATH)["model"]
    yield


app = FastAPI(title="API Renovación de Préstamo", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok" if MODEL is not None else "model_not_loaded"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    row = pd.DataFrame([request.model_dump()])[FEATURES]
    score = float(MODEL.predict_proba(row)[0, 1])
    return {"score_renovacion": score, "prediccion": int(score >= 0.5)}
