from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone

from src.config import load_config
from src.data import FEATURES, LoanData
from src.metrics import calculate_metrics, quality_gate
from src.model import Candidate, LoanRenewalModel


def log_mlflow(config: dict, params: dict, metrics: dict, model) -> bool:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("[MLflow] No instalado; el resto del pipeline continúa.")
        return False
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # MLflow 3 usa skops por defecto. Este pipeline de scikit-learn/imblearn
        # contiene numpy.dtype, que skops puede marcar como tipo no confiable.
        # Para este ejercicio académico usamos cloudpickle explícitamente, que
        # además mantiene compatibilidad con las versiones de MLflow del curso.
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            serialization_format="cloudpickle",
        )
    return True


def main() -> None:
    config = load_config()
    data = LoanData(config)
    df = data.load()
    validation = data.validate(df)
    split = data.temporal_split(df)

    trainer = LoanRenewalModel(config["model"]["random_state"])

    # Para mantener el ejercicio liviano, la comparación y el GridSearch usan
    # una muestra del train. El modelo ganador sí se reentrena luego con todo
    # train + validation antes de tocar el test final.
    sample_size = min(config["model"].get("experiment_sample_size", 20000), len(split.train))
    experiment_train = split.train.sample(n=sample_size, random_state=config["model"]["random_state"])
    candidates = trainer.compare(experiment_train, split.validation, config["model"]["top_fraction"])

    # Grid pequeño, suficiente para demostrar tuning en un trabajo de curso.
    search = trainer.tune_random_forest(experiment_train)
    tuned_scores = search.best_estimator_.predict_proba(split.validation[FEATURES])[:, 1]
    candidates.append(Candidate(
        "random_forest_gridsearch",
        search.best_estimator_,
        calculate_metrics(split.validation["FLAG_VENTA"], tuned_scores, config["model"]["top_fraction"]),
    ))

    champion = trainer.best(candidates)
    train_plus_val = pd.concat([split.train, split.validation], ignore_index=True)
    final_model = clone(champion.model)
    final_model.fit(train_plus_val[FEATURES], train_plus_val["FLAG_VENTA"])

    test_scores = final_model.predict_proba(split.test[FEATURES])[:, 1]
    test_metrics = calculate_metrics(split.test["FLAG_VENTA"], test_scores, config["model"]["top_fraction"])
    quality_gate(test_metrics, config)

    Path("artifacts").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump({"model": final_model, "features": FEATURES}, "artifacts/model.joblib")

    comparison = pd.DataFrame([{"model": c.name, **c.metrics} for c in candidates])
    comparison.to_csv("reports/model_comparison.csv", index=False)

    ranking = split.test[["MES", "CLIENTE", "FLAG_VENTA"]].copy()
    ranking["score_renovacion"] = test_scores
    ranking = ranking.sort_values("score_renovacion", ascending=False)
    ranking.to_csv("artifacts/test_ranking.csv", index=False)

    result = {
        "data_validation": validation,
        "champion": champion.name,
        "best_rf_params": search.best_params_,
        "validation_metrics": champion.metrics,
        "test_metrics": test_metrics,
    }
    Path("artifacts/metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    mlflow_ok = log_mlflow(config, {"champion": champion.name, **search.best_params_}, test_metrics, final_model)
    print(json.dumps({**result, "quality_gate": "PASSED", "mlflow_logged": mlflow_ok}, indent=2))


if __name__ == "__main__":
    main()
