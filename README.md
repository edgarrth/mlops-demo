# Caso MLOps - Renovación de Préstamo

Proyecto de **renovación de préstamo** desde un notebook hasta un flujo MLOps.

## 1. Objetivo

La financiera quiere priorizar a los clientes con mayor probabilidad de renovar un préstamo. El modelo no se usa solo como clasificador 0/1: también entrega `score_renovacion` para ordenar clientes y apoyar al call center.

El dataset contiene 87,556 registros y 22 columnas. La clase positiva representa cerca del 4%, por lo que Accuracy no es suficiente. El proyecto usa PR-AUC, ROC-AUC y Lift@10%.

## 2. Qué conceptos de las clases de MLOps aplica

4. Split temporal train/validation/test.
5. Manejo de desbalance con `class_weight` y SMOTE.
6. Comparación de modelos: Logistic Regression y Random Forest.
7. GridSearchCV pequeño para tuning.
8. MLflow para registrar parámetros, métricas y modelo.
9. Quality gate.
10. DVC como ejemplo de versionado del dataset.
11. Batch scoring para crear ranking de clientes.
12. FastAPI para inferencia online.
13. Docker / Docker Compose.
14. Tests con pytest.
15. GitHub Actions para CI, entrenamiento y monitoreo.
16. Drift con PSI, KS y Evidently cuando está instalado.

## 3. Estructura

```text
renovacion-prestamo-mlops-estudiante/
├── api/
│   └── main.py
├── artifacts/
├── data/raw/
│   ├── Dataset Renovacion_prestamo.csv
│   └── Dataset Renovacion_prestamo.csv.dvc
├── references/original/
├── reports/monitoring/
├── scripts/
│   ├── validate_data.py
│   ├── train.py
│   ├── score_batch.py
│   └── monitor.py
├── src/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── metrics.py
│   └── monitoring.py
├── tests/
├── .github/workflows/
├── config.yaml
├── dvc.yaml
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

## 4. Clases principales

### `LoanData` — `src/data.py`

Responsabilidad: trabajar con el dataset.

- `load()`: lee el CSV con separador `;`.
- `validate()`: comprueba columnas, target binario, registros y duplicados.
- `temporal_split()`: separa los meses para simular un escenario real donde se aprende del pasado y se predice el futuro.

Split usado:

- Train: 201501–201506.
- Validation: 201507–201508.
- Test final: 201509.

### `DataSplit` — `src/data.py`

Es un `dataclass` simple que agrupa `train`, `validation` y `test`.

### `LoanRenewalModel` — `src/model.py`

Responsabilidad: crear y entrenar modelos.

- `_preprocessor()`: imputación de nulos + estandarización numérica + One-Hot Encoding.
- `candidates()`: crea tres alternativas sencillas:
  - Logistic Regression con `class_weight=balanced`.
  - Random Forest con `class_weight=balanced`.
  - Random Forest + SMOTE.
- `compare()`: entrena los candidatos y los evalúa en validation.
- `tune_random_forest()`: ejecuta un GridSearch pequeño de 2 combinaciones y 2 folds sobre una muestra del train.
- `best()`: escoge el modelo con mejor PR-AUC en validation.

El preprocessing queda dentro del pipeline. Esto evita calcular imputaciones o categorías usando el test final.

### `Candidate` — `src/model.py`

`dataclass` que guarda nombre, modelo entrenado y métricas de un candidato.

### `DriftMonitor` — `src/monitoring.py`

Compara datos de referencia contra datos recientes.

Para cada variable numérica calcula:

- PSI: cambio de distribución.
- KS p-value: prueba estadística de cambio de distribución.

El script intenta además generar un reporte Evidently si la versión instalada lo permite. Si Evidently no está disponible, PSI y KS continúan funcionando.

### `PredictionRequest` — `api/main.py`

Modelo Pydantic con las variables que recibe `/predict`. FastAPI lo usa para validar el JSON de entrada.

### `PredictionResponse` — `api/main.py`

Respuesta de la API:

- `score_renovacion`: probabilidad estimada.
- `prediccion`: 1 si score >= 0.5, de lo contrario 0.

Para campañas se recomienda usar principalmente el **score y ranking**, no solo la clase.

## 5. Flujo de entrenamiento

```text
CSV
 ↓
Validación
 ↓
Split temporal
 ↓
Train 201501-201506
 ↓
Comparar modelos
 ├─ Logistic balanced
 ├─ Random Forest balanced
 └─ Random Forest + SMOTE
 ↓
GridSearch pequeño Random Forest
 ↓
Evaluar en 201507-201508
 ↓
Seleccionar por PR-AUC
 ↓
Reentrenar con Train + Validation
 ↓
Test final 201509
 ↓
Quality Gate
 ↓
Guardar modelo + métricas + ranking
 ↓
MLflow
```

## 6. Quality Gate

Los valores iniciales están en `config.yaml`:

```yaml
quality_gate:
  min_pr_auc: 0.06
  min_roc_auc: 0.62
  min_lift_at_10: 1.50
```

Si el modelo final no los supera, `scripts/train.py` termina con error. En GitHub Actions eso hace fallar el workflow.

Estos umbrales son académicos y deben ajustarse con negocio en un proyecto real.

## 7. Ejecución local

### Crear ambiente

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


### Validar datos

```bash
make validate
```

### Ejecutar tests

```bash
make test
```

### Entrenar

```bash
make train
```

Genera:

- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/test_ranking.csv`
- `reports/model_comparison.csv`

### Abrir MLflow

Si MLflow está instalado, el entrenamiento registra el run en `mlflow.db`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abrir `http://localhost:5000`.

### Batch scoring

```bash
make score
```

Genera `artifacts/campaign_ranking.csv` ordenado por `score_renovacion`.

### API

Primero debe existir `artifacts/model.joblib`.

```bash
make api
```

- Health: `http://localhost:8000/health`
- Swagger: `http://localhost:8000/docs`

### Monitoring

```bash
make monitor
```

Compara enero-junio contra septiembre y genera reportes en `reports/monitoring/`.

## 8. DVC

Se incluye el descriptor `.dvc` del dataset como ejemplo del curso.

El `dvc.yaml` muestra dos etapas: `validate` y `train`.

```bash
dvc repro
```

## 9. Docker

Después de entrenar:

```bash
docker compose up --build
```

La API queda en `http://localhost:8000`.

Detener:

```bash
docker compose down
```

## 10. GitHub Actions

Hay tres Actions deliberadamente simples.

### Action 1: `CI` — `.github/workflows/ci.yml`

**Cuándo se ejecuta:**

- cada `push`;
- cada Pull Request;
- manualmente con `workflow_dispatch`.

**Qué hace:**

1. descarga el repositorio;
2. instala Python 3.11;
3. instala dependencias;
4. valida que los `.py` compilen;
5. ejecuta pytest + coverage;
6. valida el dataset.

**Cómo ejecutarlo manualmente:**

1. subir el proyecto a GitHub;
2. entrar al repositorio;
3. abrir **Actions**;
4. seleccionar **CI**;
5. pulsar **Run workflow**.

También se ejecuta automáticamente al hacer:

```bash
git add .
git commit -m "change"
git push
```

### Action 2: `Train model` — `.github/workflows/train.yml`

**Cuándo se ejecuta:**

- manualmente;
- al hacer push a `main` si cambia código, configuración o datos.

**Qué hace:**

1. instala dependencias;
2. ejecuta entrenamiento real;
3. aplica el quality gate;
4. levanta FastAPI con el modelo recién generado;
5. llama `/health` como smoke test;
6. guarda `artifacts/` y `reports/` como artifacts de GitHub.

**Cómo ejecutarlo:**

GitHub → **Actions** → **Train model** → **Run workflow**.

Si PR-AUC, ROC-AUC o Lift@10% están debajo de los umbrales, el Action falla. Ese es el quality gate de ML.

### Action 3: `Monitor drift` — `.github/workflows/monitor.yml`

**Cuándo se ejecuta:**

- manualmente;
- automáticamente cada lunes a las 13:00 UTC (08:00 Perú).

**Qué hace:**

1. toma enero-junio como referencia;
2. usa septiembre como lote reciente del ejercicio;
3. calcula PSI y KS;
4. intenta crear Evidently HTML;
5. sube `reports/monitoring/` como artifact.

**Cómo ejecutarlo:**

GitHub → **Actions** → **Monitor drift** → **Run workflow**.
