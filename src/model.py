from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC
from .metrics import calculate_metrics


@dataclass
class Candidate:
    name: str
    model: object
    metrics: dict


class LoanRenewalModel:
    """Construye, compara y selecciona modelos de propensión."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def _preprocessor(self) -> ColumnTransformer:
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        return ColumnTransformer([
            ("num", numeric, NUMERIC),
            ("cat", categorical, CATEGORICAL),
        ])

    def candidates(self) -> dict[str, object]:
        logistic = Pipeline([
            ("prep", self._preprocessor()),
            ("model", LogisticRegression(max_iter=250, class_weight="balanced", random_state=self.random_state)),
        ])
        forest = Pipeline([
            ("prep", self._preprocessor()),
            ("model", RandomForestClassifier(
                n_estimators=50, max_depth=8, min_samples_leaf=2,
                class_weight="balanced", random_state=self.random_state, n_jobs=-1,
            )),
        ])
        forest_smote = ImbPipeline([
            ("prep", self._preprocessor()),
            ("smote", SMOTE(random_state=self.random_state, k_neighbors=1)),
            ("model", RandomForestClassifier(
                n_estimators=50, max_depth=8, min_samples_leaf=2,
                random_state=self.random_state, n_jobs=-1,
            )),
        ])
        return {
            "logistic_balanced": logistic,
            "random_forest_balanced": forest,
            "random_forest_smote": forest_smote,
        }

    def compare(self, train: pd.DataFrame, validation: pd.DataFrame, top_fraction: float) -> list[Candidate]:
        X_train, y_train = train[FEATURES], train["FLAG_VENTA"]
        X_val, y_val = validation[FEATURES], validation["FLAG_VENTA"]
        results = []
        for name, model in self.candidates().items():
            model.fit(X_train, y_train)
            scores = model.predict_proba(X_val)[:, 1]
            results.append(Candidate(name, model, calculate_metrics(y_val, scores, top_fraction)))
        return results

    def tune_random_forest(self, train: pd.DataFrame) -> GridSearchCV:
        base = self.candidates()["random_forest_balanced"]
        grid = {
            "model__n_estimators": [40, 60],
            "model__max_depth": [8],
        }
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(base, grid, scoring="average_precision", cv=cv, n_jobs=1)
        search.fit(train[FEATURES], train["FLAG_VENTA"])
        return search

    @staticmethod
    def best(candidates: list[Candidate]) -> Candidate:
        return max(candidates, key=lambda c: c.metrics["pr_auc"])
