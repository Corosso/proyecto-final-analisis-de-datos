"""Modelado predictivo (Fase F4): clasificación + calibración + odds decimales.

Objetivo
--------
Predecir el ganador de un partido profesional de Valorant a partir de las
features construidas en F3 (``data/processed/features.parquet``) y expresar el
resultado como **odds decimales** (favorito / evens / underdog).

Target
------
Binario ``y`` (1 = gana ``team1``, 0 = gana ``team2``).

Contrato de features (F3)
-------------------------
El input es un DataFrame con, al menos::

    match_id, team1, team2, date (str), date_ts (epoch numérico),
    y (int 0/1) y todas las features numéricas con prefijo ``f_``.

Las columnas ``f_*`` se detectan **de forma dinámica** (``startswith("f_")``)
para que el pipeline no se rompa si F3 cambia la lista de features.

Flujo del pipeline
------------------
1. Split temporal (pasado -> futuro) sin fuga; por defecto 80/20 sobre
   ``date_ts``/``date``, configurable también por fecha de corte.
2. Preprocesamiento dentro de un ``Pipeline`` (imputación por mediana y, para
   modelos lineales/KNN, ``StandardScaler``) para evitar fuga en el escalado.
3. Tuning con ``RandomizedSearchCV`` + validación cruzada temporal
   (``TimeSeriesSplit``) sobre el set de TRAIN.
4. Evaluación sobre TEST (nunca visto): accuracy, precision, recall, F1,
   ROC-AUC, matriz de confusión, Brier y log-loss.
5. Calibración de probabilidades (Platt/isotónica vía
   ``CalibratedClassifierCV``) y comparación del Brier antes/después.
6. Conversión a odds y tier de favoritismo con :func:`prob_to_odds`.
7. Entrenamiento final del mejor modelo calibrado y guardado de artefactos
   (``best_model.joblib`` + ``model_metadata.json``).

Dependencias
------------
scikit-learn, pandas, numpy, matplotlib y joblib son obligatorias;
``xgboost`` y ``lightgbm`` son opcionales (se importan con try/except).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from config import get_data_dir

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.base import clone  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, TimeSeriesSplit  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# --- Modelos opcionales -----------------------------------------------------
try:  # xgboost puede no estar instalado
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - depende del entorno
    XGBClassifier = None

try:  # lightgbm puede no estar instalado o carecer de libgomp
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - depende del entorno
    LGBMClassifier = None

logger = logging.getLogger("vlr.models")

if XGBClassifier is None:  # pragma: no cover - depende del entorno
    logger.warning("xgboost no disponible; se omitirá del catálogo de modelos.")
if LGBMClassifier is None:  # pragma: no cover - depende del entorno
    logger.warning("lightgbm no disponible; se omitirá del catálogo de modelos.")

# Rutas por defecto (resueltas según el entorno: Colab / WSL / local).
DEFAULT_FEATURES_PATH = get_data_dir() / "processed" / "features.parquet"
DEFAULT_OUTPUT_DIR = get_data_dir() / "processed"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "best_model.joblib"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "model_metadata.json"
DEFAULT_RESULTS_PATH = DEFAULT_OUTPUT_DIR / "model_results.csv"
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"

FEATURE_PREFIX = "f_"
EPS = 1e-6  # límite para evitar odds infinitos

# Tier de favoritismo según el odds (ambos equipos usan la misma escala).
TIER_BINS = [
    (1.40, "Favorito claro"),
    (1.80, "Favorito"),
    (2.20, "Evens/parejo"),
    (3.00, "Underdog"),
    (float("inf"), "Underdog claro"),
]

# Carga de datos

def load_features(path: str | Path) -> pd.DataFrame:
    """Carga ``features.parquet`` (o CSV de respaldo) y valida el contrato F3.

    Devuelve el DataFrame con las columnas ``f_*`` detectadas dinámicamente y
    el target ``y`` como entero 0/1.
    """
    path = Path(path)
    parquet = path.with_suffix(".parquet")
    csv = path.with_suffix(".csv")
    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        raise FileNotFoundError(
            f"No existe {parquet} ni {csv}. "
            "La fase F3 (feature engineering) aún no ha generado el archivo; "
            "puedes usar data sintética con make_synthetic_data()."
        )

    # Validación del contrato mínimo.
    required = {"match_id", "team1", "team2", "y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Columnas obligatorias ausentes: {missing}")

    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise ValueError("No se encontraron features con prefijo 'f_' en el dataset.")

    # Target binario limpio (descarta filas con y nulo).
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    logger.info(
        "features.parquet cargado: %d filas, %d features (f_*).",
        len(df), len(feature_cols),
    )
    return df


def _time_key(df: pd.DataFrame, date_col: str | None) -> pd.Series:
    """Devuelve una clave temporal ordenable (numérica) para el split.

    Prioriza ``date_ts`` (epoch) y, si no existe, intenta parsear ``date``.
    """
    if date_col is not None:
        cols = [date_col]
    elif "date_ts" in df.columns:
        cols = ["date_ts"]
    elif "date" in df.columns:
        cols = ["date"]
    else:
        raise ValueError("No hay columna temporal ('date_ts' ni 'date') para el split.")

    col = cols[0]
    if col == "date_ts" or col in df.columns:
        num = pd.to_numeric(df[col], errors="coerce")
        if num.notna().mean() > 0.5:
            return num
    # Fallback: parsear fechas legibles.
    return pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9


def temporal_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    cutoff: str | float | None = None,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal sin fuga: entrenamiento (pasado) y test (futuro).

    Ordena por ``date_ts``/``date`` y divide manteniendo el orden cronológico.
    Usa ``test_frac`` (~80/20) o una ``cutoff`` explícita (epoch o fecha ISO).
    """
    key = _time_key(df, date_col)
    order = np.argsort(key.to_numpy(), kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    key = key.iloc[order].reset_index(drop=True)
    if cutoff is not None:
        if isinstance(cutoff, str):
            cutoff = pd.Timestamp(cutoff).timestamp()
        train = df[key <= cutoff]
        test = df[key > cutoff]
        if train.empty or test.empty:
            raise ValueError("El cutoff deja vacío el set de train o de test.")
    else:
        n_test = max(1, int(round(len(df) * test_frac)))
        train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    logger.info(
        "Split temporal: train=%d (hasta %s), test=%d (desde %s).",
        len(train),
        _range_label(train, date_col)[1],
        len(test),
        _range_label(test, date_col)[0],
    )
    return train, test


def _range_label(df: pd.DataFrame, date_col: str | None) -> tuple[str, str]:
    """Devuelve (primera, última) fecha legible de un subconjunto (para logs)."""
    if df.empty:
        return "n/a", "n/a"

    def fmt(row: pd.Series) -> str:
        if "date" in df.columns and pd.notna(row["date"]):
            return str(row["date"])
        col = date_col or ("date_ts" if "date_ts" in df else None)
        if col is None:
            return "n/a"
        try:
            return datetime.fromtimestamp(
                float(row[col]), tz=timezone.utc
            ).strftime("%Y-%m-%d")
        except (TypeError, ValueError, OSError):
            return str(row[col])

    return fmt(df.iloc[0]), fmt(df.iloc[-1])

# Dataset sintético (smoke test / demostración cuando F3 no ha corrido)

def make_synthetic_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Genera un dataset sintético con la estructura del contrato F3.

    Incluye señal real (la brecha de habilidad entre equipos decide el partido)
    para que los modelos aprendan algo y la calibración sea evaluable.
    """
    rng = np.random.default_rng(seed)
    team_pool = [f"Team{i:02d}" for i in range(20)]
    skill = {t: float(rng.normal(0, 1)) for t in team_pool}
    dates = pd.date_range("2024-01-01", periods=n, freq="D").to_pydatetime()

    rows = []
    for i in range(n):
        t1, t2 = rng.choice(team_pool, size=2, replace=False)
        s1, s2 = skill[t1], skill[t2]
        gap = s1 - s2
        p = 1.0 / (1.0 + np.exp(-1.5 * gap))
        y = int(rng.random() < p)
        f_rating_diff = gap + rng.normal(0, 0.5)
        f_form_diff = gap + rng.normal(0, 0.8)
        f_winrate_diff = gap + rng.normal(0, 0.6)
        f_h2h_winrate = 1.0 / (1.0 + np.exp(-gap)) + rng.normal(0, 0.1)
        if i % 17 == 0:  # nulos deliberados para ejercitar el imputador
            f_rating_diff = np.nan
        rows.append({
            "match_id": f"syn_{i:05d}",
            "team1": t1,
            "team2": t2,
            "date": dates[i].strftime("%Y-%m-%d"),
            "date_ts": dates[i].timestamp(),
            "y": y,
            "f_rating_diff": f_rating_diff,
            "f_form_diff": f_form_diff,
            "f_winrate_diff": f_winrate_diff,
            "f_h2h_winrate": f_h2h_winrate,
        })
    return pd.DataFrame(rows)

# Odds

def tier_from_odds(odds: float) -> str:
    """Tier de favoritismo según la escala de odds decimales."""
    for upper, label in TIER_BINS:
        if odds <= upper:
            return label
    return "Underdog claro"


def prob_to_odds(p: float) -> tuple[float, float, str]:
    """Convierte ``P(team1 gana)`` en odds "fair" y tier de team1.

    Devuelve ``(odds_A, odds_B, tier_A)`` con ``odds_A = 1/P(team1)`` y
    ``odds_B = 1/(1-P(team1))``. La probabilidad se acota a ``[EPS, 1-EPS]``
    para evitar divisiones por cero.
    """
    p = float(np.clip(p, EPS, 1.0 - EPS))
    odds_a = 1.0 / p
    odds_b = 1.0 / (1.0 - p)
    return odds_a, odds_b, tier_from_odds(odds_a)

# Catálogo de modelos y utilidades de pipelines

def _make_pipeline(model, scale: bool) -> Pipeline:
    """Pipeline de preprocesamiento + modelo, sin fuga en el escalado.

    Imputación por mediana siempre; ``StandardScaler`` solo para modelos que lo
    necesitan (lineales/KNN). Los árboles y boosting no requieren escala.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _model_catalog(random_state: int = 42) -> list[dict[str, Any]]:
    """Lista de modelos a probar: (nombre, estimador, escala?, grilla de tuning)."""
    catalog: list[dict[str, Any]] = [
        {
            "name": "LogReg",
            "model": LogisticRegression(max_iter=2000, random_state=random_state),
            "scale": True,
            "grid": {
                "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        {
            "name": "KNN",
            "model": KNeighborsClassifier(),
            "scale": True,
            "grid": {
                "model__n_neighbors": [3, 5, 7, 10, 15, 20],
                "model__weights": ["uniform", "distance"],
            },
        },
        {
            "name": "RandomForest",
            "model": RandomForestClassifier(random_state=random_state, n_jobs=1),
            "scale": False,
            "grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [5, 8, 12, 20],
                "model__min_samples_leaf": [2, 4, 8],
            },
        },
    ]
    if XGBClassifier is not None:
        catalog.append({
            "name": "XGBoost",
            "model": XGBClassifier(
                eval_metric="logloss", random_state=random_state,
                n_jobs=1, tree_method="hist",
            ),
            "scale": False,
            "grid": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
            },
        })
    if LGBMClassifier is not None:
        catalog.append({
            "name": "LightGBM",
            "model": LGBMClassifier(
                random_state=random_state, n_jobs=1, verbose=-1,
            ),
            "scale": False,
            "grid": {
                "model__n_estimators": [100, 200, 300],
                "model__num_leaves": [15, 31, 63],
                "model__learning_rate": [0.01, 0.1, 0.2],
            },
        })
    return catalog


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """ROC-AUC tolerante a sets con una sola clase."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))

# Calibración de probabilidades

def _fit_calibrated(
    base_estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "sigmoid",
    cv_splits: int = 5,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Calibra un modelo ya entrenado (Platt/isotónica) vía ``CalibratedClassifierCV``.

    Usa ``ensemble=False``: la calibración se estima con predicciones
    out-of-fold (``cross_val_predict`` con ``KFold``) y, para predecir, el
    estimador base se reentrena con TODOS los datos de entrenamiento. Esto
    devuelve un objeto estándar de scikit-learn (fácil de serializar con
    joblib). El split temporal estricto ya se garantiza en el split
    train/test y en el tuning (``TimeSeriesSplit``); aquí la validación
    cruzada solo estima el mapa monótono de calibración.
    """
    cal = CalibratedClassifierCV(
        estimator=clone(base_estimator),
        method=method,
        cv=KFold(n_splits=cv_splits, shuffle=True, random_state=random_state),
        ensemble=False,
    )
    cal.fit(X_train, y_train)
    return cal

# Pipeline principal

def train_evaluate(
    df: pd.DataFrame | str | Path | None = None,
    *,
    test_frac: float = 0.2,
    cutoff: str | float | None = None,
    date_col: str | None = None,
    n_iter: int = 12,
    cv_splits: int = 3,
    random_state: int = 42,
    calib_method: str = "sigmoid",
    calibrate: bool = True,
    tune: bool = True,
    n_jobs: int = 1,
    max_rows: int | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    return_full: bool = False,
) -> pd.DataFrame | dict[str, Any]:
    """Entrena, calibra y evalúa los modelos; devuelve la tabla comparativa.

    ``df`` puede ser un DataFrame o una ruta (str/Path) a ``features.parquet``.
    Si es una ruta inexistente lanza ``FileNotFoundError`` con un mensaje claro.

    Con ``return_full=True`` devuelve un dict con todo lo necesario para
    graficar (probas calibradas, sets, modelos) además de ``results_df``.
    """
    # 1) Resolución del input.
    if df is None:
        df = DEFAULT_FEATURES_PATH
    if isinstance(df, (str, Path)):
        df = load_features(df)
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("df debe ser un DataFrame o una ruta a features.parquet.")

    # 2) Features dinámicas + split temporal.
    feature_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise ValueError("No se encontraron features con prefijo 'f_'.")

    # Submuestreo opcional: conserva los partidos más recientes (tras ordenar
    # cronológicamente) para experimentos rápidos sin romper el split temporal.
    if max_rows is not None and len(df) > max_rows:
        key = _time_key(df, date_col)
        df = df.iloc[np.argsort(key.to_numpy(), kind="stable")] \
            .reset_index(drop=True).tail(max_rows).reset_index(drop=True)

    train, test = temporal_split(df, test_frac=test_frac, cutoff=cutoff, date_col=date_col)

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train["y"].to_numpy()
    y_test = test["y"].to_numpy()

    tscv = TimeSeriesSplit(n_splits=cv_splits)
    results: list[dict[str, Any]] = []
    full: dict[str, Any] = {"models": {}}

    for spec in _model_catalog(random_state=random_state):
        name = spec["name"]
        logger.info("Entrenando %s ...", name)
        pipe = _make_pipeline(spec["model"], spec["scale"])

        # 3) Tuning sobre TRAIN con CV temporal.
        if tune:
            search = RandomizedSearchCV(
                pipe,
                spec["grid"],
                n_iter=min(n_iter, _grid_size(spec["grid"])),
                cv=tscv,
                scoring="roc_auc",
                n_jobs=n_jobs,
                random_state=random_state,
                refit=True,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params = search.best_params_
        else:
            best = pipe.fit(X_train, y_train)
            best_params = {}

        # 4) Predicciones sin calibrar.
        proba_uncal = best.predict_proba(X_test)[:, 1]
        y_pred = (proba_uncal >= 0.5).astype(int)

        # 5) Calibración de probabilidades (Platt/isotónica).
        proba_cal = proba_uncal
        cal = None
        if calibrate:
            cal = _fit_calibrated(
                best, X_train, y_train, method=calib_method,
                cv_splits=min(cv_splits, 5), random_state=random_state,
            )
            proba_cal = cal.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": _safe_auc(y_test, proba_uncal),
            "brier_uncal": brier_score_loss(y_test, proba_uncal),
            "logloss_uncal": log_loss(y_test, np.clip(proba_uncal, EPS, 1 - EPS)),
            "brier_cal": brier_score_loss(y_test, proba_cal),
            "logloss_cal": log_loss(y_test, np.clip(proba_cal, EPS, 1 - EPS)),
        }

        row = {
            "model": name,
            **metrics,
            "best_params": json.dumps(best_params, default=str),
        }
        results.append(row)
        full["models"][name] = {
            "model": best,
            "calibrated": cal if calibrate else None,
            "proba_uncal": proba_uncal,
            "proba_cal": proba_cal,
            "y_pred": y_pred,
            "best_params": best_params,
        }
        logger.info(
            "%s -> AUC=%.3f Brier(uncal)=%.4f Brier(cal)=%.4f",
            name, metrics["roc_auc"], metrics["brier_uncal"], metrics["brier_cal"],
        )

    results_df = pd.DataFrame(results)

    # 6) Mejor modelo por Brier calibrado (tie-break: ROC-AUC).
    results_df["_rank"] = results_df["brier_cal"].rank()
    results_df["_auc_rank"] = results_df["roc_auc"].rank(ascending=False)
    results_df = results_df.sort_values(["_rank", "_auc_rank"]).drop(
        columns=["_rank", "_auc_rank"]
    ).reset_index(drop=True)
    best_name = results_df.iloc[0]["model"]
    best_entry = full["models"][best_name]
    best_model = best_entry["calibrated"] if calibrate else best_entry["model"]

    # 7) Guardado de artefactos.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "model_results.csv", index=False)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": "y (1 = gana team1, 0 = gana team2)",
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "best_model": best_name,
        "calib_method": calib_method if calibrate else None,
        "split": {
            "test_frac": test_frac,
            "cutoff": cutoff,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_first": _range_label(train, date_col)[0],
            "train_last": _range_label(train, date_col)[1],
            "test_first": _range_label(test, date_col)[0],
            "test_last": _range_label(test, date_col)[1],
        },
        "best_params": best_entry["best_params"],
        "metrics": {k: results_df.iloc[0][k] for k in
                    ("accuracy", "precision", "recall", "f1", "roc_auc",
                     "brier_uncal", "brier_cal", "logloss_uncal", "logloss_cal")},
    }
    joblib.dump(best_model, output_dir / "best_model.joblib")
    with open(output_dir / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Mejor modelo: %s (Brier calibrado=%.4f).", best_name,
                metadata["metrics"]["brier_cal"])

    full.update({
        "results_df": results_df,
        "train": train,
        "test": test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "best_name": best_name,
        "best_model": best_model,
        "metadata": metadata,
        "output_dir": output_dir,
    })
    return full if return_full else results_df


def _grid_size(grid: dict) -> int:
    """Tamaño máximo de una grilla (producto de cardinalidades)."""
    size = 1
    for values in grid.values():
        size *= len(values)
    return size

# Predicción con odds

def load_best_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Carga el modelo calibrado guardado y sus metadatos."""
    model_path, metadata_path = Path(model_path), Path(metadata_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe {model_path}. Corre train_evaluate() primero."
        )
    model = joblib.load(model_path)
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as fh:
            metadata = json.load(fh)
    return model, metadata


def predict_match(
    team1: str,
    team2: str,
    features: Mapping[str, float] | pd.DataFrame | pd.Series,
    model: Any = None,
    metadata: Mapping[str, Any] | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Predice ``P(team1 gana)`` y devuelve odds y tier para ambos equipos.

    ``features`` debe contener las columnas ``f_*`` que el modelo usó en el
    entrenamiento (las provee la fase F3). En producción se calculan a partir
    de las stats actuales de los equipos.
    """
    if model is None:
        model, loaded_meta = load_best_model()
        if metadata is None:
            metadata = loaded_meta

    if feature_cols is None:
        feature_cols = (metadata or {}).get("feature_cols")
    if feature_cols is None:
        raise ValueError(
            "No se pudo determinar feature_cols; pásalas explícitamente o "
            "asegúrate de que model_metadata.json contenga 'feature_cols'."
        )

    if isinstance(features, pd.DataFrame):
        row = features.reindex(columns=feature_cols)
    elif isinstance(features, pd.Series):
        row = features.to_frame().T.reindex(columns=feature_cols)
    elif isinstance(features, Mapping):
        missing = [c for c in feature_cols if c not in features]
        if missing:
            raise ValueError(f"Faltan features requeridas: {missing}")
        row = pd.DataFrame([features]).reindex(columns=feature_cols)
    else:
        raise TypeError("features debe ser dict, Series o DataFrame de una fila.")

    p = float(np.clip(model.predict_proba(row)[0, 1], EPS, 1.0 - EPS))
    odds_a, odds_b, tier_a = prob_to_odds(p)
    tier_b = tier_from_odds(odds_b)
    winner = team1 if p >= 0.5 else team2

    return {
        "team1": team1,
        "team2": team2,
        "p_team1": p,
        "prediction": winner,
        "odds_team1": odds_a,
        "odds_team2": odds_b,
        "tier_team1": tier_a,
        "tier_team2": tier_b,
    }

# Gráficos de diagnóstico

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de confusión",
    save_path: str | Path | None = None,
) -> Any:
    """Dibuja y opcionalmente guarda la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["Gana team2", "Gana team1"],
    ).plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_calibration(
    y_true: np.ndarray,
    proba_uncal: np.ndarray,
    proba_cal: np.ndarray,
    title: str = "Curva de calibración",
    save_path: str | Path | None = None,
) -> Any:
    """Diagrama de calibración (reliability) antes/después de calibrar."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for proba, label, style in [
        (proba_uncal, "Sin calibrar", "r--"),
        (proba_cal, "Calibrado", "g-"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, style, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k:", label="Perfecta")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia real de victoria")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig


def plot_roc(y_true, proba, title="Curva ROC", save_path=None):
    """Dibuja la curva ROC de un modelo."""
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax, name=title)
    ax.plot([0, 1], [0, 1], "k:", label="Aleatorio")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    return fig

# CLI

def main(argv: list[str] | None = None) -> int:
    """Punto de entrada con argparse para ejecutar el pipeline F4."""
    parser = argparse.ArgumentParser(description="Pipeline de modelado F4 (Valorant).")
    parser.add_argument("--input", type=str, default=None,
                        help="Ruta a features.parquet (default: data/processed/features.parquet).")
    parser.add_argument("--synthetic", action="store_true",
                        help="Genera data sintética si el input no existe.")
    parser.add_argument("--n-synthetic", type=int, default=600)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--cutoff", type=str, default=None)
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--calib-method", type=str, default="sigmoid",
                        choices=["sigmoid", "isotonic"])
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Backend headless para guardar PNGs (solo en CLI; en notebooks se usa inline).
    matplotlib.use("Agg")

    if args.input:
        if Path(args.input).exists():
            df: pd.DataFrame | str = args.input
        elif args.synthetic:
            logger.warning("Input no encontrado (%s); usando data sintética.", args.input)
            df = make_synthetic_data(n=args.n_synthetic, seed=args.seed)
        else:
            logger.error(
                "No existe %s. Pasa --synthetic para usar data sintética o espera "
                "a que F3 genere features.parquet.", args.input,
            )
            return 2
    else:
        default = DEFAULT_FEATURES_PATH
        if default.exists():
            df = str(default)
        elif args.synthetic:
            logger.warning("Usando data sintética (features.parquet aún no existe).")
            df = make_synthetic_data(n=args.n_synthetic, seed=args.seed)
        else:
            logger.error(
                "No existe %s. Pasa --synthetic o genera el archivo con F3.", default,
            )
            return 2

    full = train_evaluate(
        df,
        test_frac=args.test_frac,
        cutoff=args.cutoff,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        random_state=args.seed,
        calib_method=args.calib_method,
        calibrate=not args.no_calibrate,
        tune=not args.no_tune,
        n_jobs=args.n_jobs,
        max_rows=args.max_rows,
        output_dir=args.output_dir,
        return_full=True,
    )

    print("\n=== Tabla comparativa (test) ===\n")
    print(full["results_df"].round(4).to_string(index=False))

    # Guarda gráficos de diagnóstico.
    figs = Path(args.output_dir) / "figures"
    best = full["best_name"]
    entry = full["models"][best]
    plot_confusion_matrix(
        full["y_test"], entry["y_pred"],
        title=f"Matriz de confusión — {best}",
        save_path=figs / f"confusion_{best}.png",
    )
    plot_calibration(
        full["y_test"], entry["proba_uncal"], entry["proba_cal"],
        title=f"Calibración — {best}",
        save_path=figs / f"calibration_{best}.png",
    )
    plt.close("all")

    print(f"\nArtefactos guardados en {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
