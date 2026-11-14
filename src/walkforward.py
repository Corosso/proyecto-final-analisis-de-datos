"""Walk-forward (Capa 2) — probabilidades out-of-sample honestas por partido.

Objetivo
--------
Generar ``P(team1 gana)`` para (casi) todos los partidos de
``features.parquet`` **sin fuga temporal**: cada partido se predice con un modelo
entrenado **solo con partidos anteriores** a él. De ahí se derivan los odds
decimales (``1/P``) y el *tier* de favoritismo, que se exportan para Power BI.

Método
------
Se usa ``sklearn.model_selection.TimeSeriesSplit`` con una **ventana
expandible** (expanding window): el fold ``k`` entrena sobre los primeros ``k``
bloques temporales y predice sobre el bloque ``k+1`` (futuro). Es el mismo
esquema "pasado -> futuro" que el split 80/20 de F4, pero repetido a lo largo de
toda la historia, de modo que **casi todos** los partidos reciben una predicción
out-of-sample.

- **Modelo**: ``RandomForestClassifier(n_estimators=100, max_depth=12,
  min_samples_leaf=8, random_state=42, n_jobs=1)`` — los hiperparámetros
  tuneados del mejor modelo (ver ``model_metadata.json``), sin calibración
  Platt (el RF ya produce probabilidades razonables y la calibración anidada
  queda fuera del alcance de esta fase).
- **Imputación**: mediana por fold (fit sobre train, transform sobre test) por
  rigor; en la práctica las features no tienen nulos.
- **Cobertura OOS**: el primer bloque temporal (los partidos más antiguos,
  ~1/11 del total ≈ 9 %) **no** recibe predicción porque no hay pasado para
  entrenar; esas filas quedan con ``prob_team1 = NaN`` (lo honesto).

El resultado se guarda en ``data/processed/probabilities.parquet`` y ``.csv`` con
las columnas ``match_id, prob_team1, odds_team1, odds_team2, tier``.

Uso
---
    uv run --python .venv python -m src.walkforward \
        --input data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

# Helpers reutilizados de F4 (misma escala de odds/tier). Se importan desde el
# paquete ``src`` (relativo) con fallback a la ruta absoluta del paquete.
try:  # ejecutado como ``python -m src.walkforward``
    from .models import EPS, load_features, tier_from_odds
except ImportError:  # pragma: no cover - ejecutado como script suelto
    from src.models import EPS, load_features, tier_from_odds

logger = logging.getLogger("vlr.walkforward")

# Hiperparámetros tuneados del mejor modelo (ver model_metadata.json).
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 8,
    "random_state": 42,
    "n_jobs": 1,
}

DEFAULT_OUTPUT_DIR = Path("data/processed")


def walk_forward_probabilities(
    features_path: str | Path,
    n_splits: int = 10,
    class_weight: str | dict | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Calcula ``P(team1 gana)`` out-of-sample para cada partido (walk-forward).

    Devuelve un DataFrame con ``match_id, prob_team1, odds_team1, odds_team2,
    tier``. Las filas sin pasado suficiente (primer bloque temporal) tienen
    ``prob_team1 = NaN`` (y por tanto odds/tier vacíos).

    Parámetros
    ----------
    features_path : ruta a ``features.parquet`` (contrato F3).
    n_splits : nº de folds del ``TimeSeriesSplit`` (default 10). El primer
        ~``1/(n_splits+1)`` de los partidos quedará sin predicción OOS.
    class_weight : opcional; ``"balanced"`` si se quisiera compensar el
        desbalance. Por defecto ``None`` (consistente con el mejor modelo).
    output_dir : carpeta donde guardar ``probabilities.parquet``/``.csv``.
    """
    # 1) Carga y orden cronológico (estable) para el split temporal.
    df = load_features(features_path)
    df = df.sort_values(["date_ts", "match_id"], kind="stable").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.int64)
    n = len(df)

    # 2) Walk-forward con ventana expandible.
    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba = np.full(n, np.nan, dtype=np.float64)

    logger.info(
        "Walk-forward: %d partidos, %d features, %d folds (expanding window).",
        n, len(feature_cols), n_splits,
    )
    t_start = time.time()
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te = X[te_idx]

        # Imputación por mediana fit en TRAIN (sin fuga), transform en test.
        imputer = SimpleImputer(strategy="median")
        X_tr_i = imputer.fit_transform(X_tr)
        X_te_i = imputer.transform(X_te)

        model = RandomForestClassifier(**RF_PARAMS, class_weight=class_weight)
        t0 = time.time()
        model.fit(X_tr_i, y_tr)
        proba[te_idx] = model.predict_proba(X_te_i)[:, 1]

        logger.info(
            "  fold %02d/%d: train=%6d test=%6d  fit=%.1fs  (acumulado %.1fs)",
            fold, n_splits, len(tr_idx), len(te_idx),
            time.time() - t0, time.time() - t_start,
        )

    # 3) Derivación de odds y tier (respetando NaN).
    out = pd.DataFrame({"match_id": df["match_id"].to_numpy(), "prob_team1": proba})
    p = np.clip(proba, EPS, 1.0 - EPS)
    out["odds_team1"] = 1.0 / p
    out["odds_team2"] = 1.0 / (1.0 - p)
    # tier solo para filas con predicción; NaN -> cadena vacía.
    # Se evalúa sobre el odds de team1 (1/p), igual que prob_to_odds.
    out["tier"] = np.where(np.isnan(proba), "", np.vectorize(tier_from_odds)(1.0 / p))

    # 4) Métricas out-of-sample (sobre filas con predicción).
    mask = ~np.isnan(proba)
    coverage = mask.mean()
    if mask.sum() > 0 and len(np.unique(y[mask])) > 1:
        from sklearn.metrics import brier_score_loss, roc_auc_score
        auc = float(roc_auc_score(y[mask], proba[mask]))
        brier = float(brier_score_loss(y[mask], proba[mask]))
    else:
        auc, brier = float("nan"), float("nan")
    logger.info(
        "Cobertura OOS: %.1f%% (%d/%d) · ROC-AUC=%.4f · Brier=%.4f",
        coverage * 100, mask.sum(), n, auc, brier,
    )

    # 5) Guardado (parquet + CSV).
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_dir / "probabilities.parquet", index=False)
    out.to_csv(output_dir / "probabilities.csv", index=False)
    logger.info("Guardado en %s (probabilities.parquet / .csv)", output_dir)

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward OOS (probabilidades + odds).")
    parser.add_argument("--input", type=str, default=None,
                        help="Ruta a features.parquet (default: data/processed/features.parquet).")
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--class-weight", type=str, default=None,
                        choices=[None, "balanced"])
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.input:
        features_path = args.input
    else:
        features_path = "data/processed/features.parquet"

    out = walk_forward_probabilities(
        features_path,
        n_splits=args.n_splits,
        class_weight=args.class_weight,
        output_dir=args.output_dir,
    )
    print("\n=== Walk-forward OOS ===")
    print(out[["prob_team1", "odds_team1", "odds_team2", "tier"]].describe().round(4).to_string())
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
