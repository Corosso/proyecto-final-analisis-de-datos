# F4 · Modelado predictivo — ganador y odds decimales

## 1. Objetivo

Predecir el **ganador** de un partido profesional de Valorant a partir de las
features construidas en F3 y expresar la predicción como **odds decimales**
(favorito / evens / underdog).

- **Target**: binario `y` (1 = gana `team1`, 0 = gana `team2`).
- **Salida**: para cada partido, una probabilidad calibrada `P(team1 gana)`
  → odds "fair" `odds_A = 1/P`, `odds_B = 1/(1-P)` → tier de favoritismo.

## 2. Contrato de features (F3) e integración

El pipeline lee `data/processed/features.parquet`. Contrato esperado:

| columna | tipo | descripción |
|---|---|---|
| `match_id` | str | identificador del partido |
| `team1`, `team2` | str | equipos |
| `date` | str | fecha legible |
| `date_ts` | numérico | epoch (para el split temporal) |
| `y` | int 0/1 | target (1 = gana team1) |
| `f_*` | numérico | features (9 en F3) |

Las columnas `f_*` se **detectan dinámicamente** (`[c for c in df if
c.startswith('f_')]`), de modo que si F3 agrega, quita o renombra features, el
pipeline no se rompe. En la corrida real se detectaron 9 features:

```
f_h2h_winrate, f_rating_diff, f_rank_diff, f_form_diff, f_winrate_diff,
f_mapscore_diff, f_experience_diff, f_rest_diff, f_streak_diff
```

La función de carga (`load_features`) valida las columnas obligatorias
(`match_id`, `team1`, `team2`, `y`), fuerza `y` a entero y descarta filas con
`y` nulo. Si el archivo no existe (F3 en progreso), lanza un mensaje claro y se
puede usar data sintética (`make_synthetic_data`) para validar el flujo.

## 3. Metodología

### 3.1 Split temporal sin fuga

- Se ordenan los partidos por `date_ts` (o `date` si no hay epoch).
- Se divide en **entrenamiento (pasado)** y **test (futuro)**, por defecto
  **80/20**, configurable con `test_frac` o con una fecha de corte `cutoff`.
- **No** se usa `train_test_split` aleatorio: en series temporales fugaría
  información del futuro al pasado.

### 3.2 Preprocesamiento

Dentro de un `Pipeline` para evitar fuga en el escalado:

- `SimpleImputer(strategy="median")` para nulos y valores no finitos
  (se reemplazan `inf`/`-inf` por `NaN` antes de imputar).
- `StandardScaler` solo para modelos que lo requieren (Regresión Logística y
  KNN). Los árboles y el boosting no escalan.

### 3.3 Modelos

| modelo | rol | escala |
|---|---|---|
| Regresión Logística (`max_iter=2000`) | baseline | sí |
| KNN | vecinos próximos | sí |
| Random Forest | ensemble | no |
| XGBoost | boosting (opcional) | no |
| LightGBM | boosting (opcional) | no |

`xgboost` y `lightgbm` se importan con `try/except`: si no están instalados (o
falta `libgomp` para LightGBM), se omiten del catálogo con un warning.

### 3.4 Tuning

`RandomizedSearchCV` sobre el set de **TRAIN** con validación cruzada temporal
`TimeSeriesSplit(n_splits=3)` y métrica `roc_auc`. Grillas probadas:

| modelo | hiperparámetros |
|---|---|
| LogReg | `C ∈ {0.001…10}`, `class_weight ∈ {None, balanced}` |
| KNN | `n_neighbors ∈ {3,5,7,10,15,20}`, `weights ∈ {uniform, distance}` |
| Random Forest | `n_estimators ∈ {100,200}`, `max_depth ∈ {5,8,12,20}`, `min_samples_leaf ∈ {2,4,8}` |
| XGBoost | `n_estimators ∈ {100,200,300}`, `max_depth ∈ {3,5,7}`, `learning_rate ∈ {0.01,0.1,0.2}`, `subsample ∈ {0.8,1.0}` |
| LightGBM | `n_estimators ∈ {100,200,300}`, `num_leaves ∈ {15,31,63}`, `learning_rate ∈ {0.01,0.1,0.2}` |

### 3.5 Evaluación (test nunca visto)

- Clasificación: accuracy, precision, recall, F1, ROC-AUC y matriz de confusión.
- **Calibración** (clave para los odds): **Brier score** y **log-loss**, más
  curva/diagrama de calibración.

### 3.6 Calibración de probabilidades

Se aplica `CalibratedClassifierCV` (Platt/sigmoid por defecto, isotónica
opcional) con `ensemble=False`: el mapa de calibración se estima con
predicciones out-of-fold (validación cruzada con `KFold`) y, para predecir, el
estimador base se reentrena con **todos** los datos de entrenamiento. Se
compara el Brier antes/después.

> Nota: `cross_val_predict` no admite `TimeSeriesSplit`; por eso la calibración
> usa `KFold`. El orden temporal estricto ya se garantiza en el split train/test
> y en el tuning.

## 4. Resultados (corrida sobre los últimos 20 000 partidos reales)

Split: `train = 16 000` · `test = 4 000` (80/20). Métricas sobre test:

| modelo | accuracy | precision | recall | F1 | ROC-AUC | Brier (sin cal.) | Brier (cal.) | log-loss (cal.) |
|---|---|---|---|---|---|---|---|---|
| RandomForest | 0.687 | 0.701 | 0.832 | 0.761 | 0.732 | 0.2023 | **0.2022** | 0.5903 |
| XGBoost | 0.689 | 0.699 | 0.845 | 0.765 | 0.728 | 0.2035 | 0.2032 | 0.5930 |
| LightGBM | 0.689 | 0.693 | 0.862 | 0.768 | 0.728 | 0.2053 | 0.2034 | 0.5934 |
| LogReg | 0.616 | 0.706 | 0.617 | 0.658 | 0.670 | 0.2266 | 0.2192 | 0.6283 |
| KNN | 0.642 | 0.669 | 0.798 | 0.728 | 0.661 | 0.2237 | 0.2214 | 0.6329 |

**Mejor modelo**: RandomForest (Brier calibrado 0.2022, ROC-AUC 0.732).
La calibración reduce el Brier en todos los modelos (más notoriamente en
LogReg y KNN, que parten peor calibrados).

## 5. Odds y tier de favoritismo

Escala aplicada por `prob_to_odds(p) -> (odds_A, odds_B, tier)`:

| odds | tier |
|---|---|
| ≤ 1.40 | Favorito claro |
| 1.41 – 1.80 | Favorito |
| 1.81 – 2.20 | Evens/parejo |
| 2.21 – 3.00 | Underdog |
| > 3.00 | Underdog claro |

`predict_match(team1, team2, features)` devuelve `P(team1)`, odds de ambos
equipos y sus tiers usando el mejor modelo calibrado.

## 6. Walk-forward out-of-sample (probabilidades por partido)

Para la Capa 2 se generó `P(team1 gana)` **out-of-sample** para (casi) todos los
partidos, sin fuga temporal, en `src/walkforward.py`:

- **Método**: ventana expandible con `TimeSeriesSplit(n_splits=10)`. En cada fold
  se entrena sobre los bloques temporales previos y se predice `predict_proba`
  sobre el bloque siguiente (futuro).
- **Modelo**: `RandomForestClassifier(n_estimators=100, max_depth=12,
  min_samples_leaf=8, random_state=42, n_jobs=1)` — los hiperparámetros tuneados
  del mejor modelo (sin calibración Platt anidada). `class_weight=None` (el
  desbalance es leve, ~59 % / 41 %, y el mejor modelo no lo usó).
- **Cobertura OOS**: **90,9 %** (97 490 / 107 242 partidos). El ~9 % más antiguo
  (primer bloque temporal, 9 752 partidos) no tiene pasado para entrenar y queda
  con `prob_team1 = NULL` (lo honesto).
- **Resultado OOS**: **ROC-AUC = 0.7164**, **Brier = 0.2078** (ligeramente mejor
  que el 0.707 del split 80/20, coherente con evaluar sobre más datos).

Salida: `data/processed/probabilities.{parquet,csv}` con `match_id, prob_team1,
odds_team1, odds_team2, tier`, que `src/bi_export.py` une al `fact_matches`.

## 7. Artefactos generados

- `data/processed/best_model.joblib` — mejor modelo calibrado (joblib).
- `data/processed/model_metadata.json` — features usadas, split, fecha, best params.
- `data/processed/model_results.csv` — tabla comparativa de modelos.
- `data/processed/probabilities.{parquet,csv}` — probabilidades OOS por partido
  (walk-forward, Capa 2) con `match_id, prob_team1, odds_team1, odds_team2, tier`.
- `data/processed/figures/confusion_<modelo>.png` y `calibration_<modelo>.png`.

## 8. Decisiones y riesgos

- **Split temporal estricto** (pasado → futuro) para no sobreestimar el
  desempeño; el ROC-AUC reportado es sobre partidos "futuros" nunca vistos.
- **Mejor modelo por Brier calibrado** (con tie-break por ROC-AUC), coherente
  con el objetivo de producir odds bien calibrados, no solo clases correctas.
- **Calibración Platt** como default (isotónica requiere ≥ ~1000 muestras de
  calibración; con datos chicos puede sobreajustar).
- **Desbalance leve**: `y=1` es ~59% de los partidos. Se probó
  `class_weight="balanced"` en LogReg, pero los árboles/boosting dominan.
- **AUC moderado (~0.73)**: las features de F3 capturan parte de la señal
  (rating, forma, winrate, h2h) pero el ganador de un partido de esports tiene
  alta varianza intrínseca. Para el uso como odds esto se traduce en cuotas
  "informativas" más que "exactas".
- **Serialización**: el modelo calibrado es un objeto estándar de scikit-learn
  (`CalibratedClassifierCV`), lo que garantiza que `joblib.load` funcione sin
  clases propias.
- **Rendimiento**: `n_jobs=1` por defecto (el backend `loky` con `n_jobs>1`
  tiene overhead alto en este entorno para jobs pequeños); usar `--max-rows`
  para experimentos rápidos.

## 9. Conclusiones preliminares

1. Los modelos de árboles/boosting (RandomForest, XGBoost, LightGBM) superan
   claramente a los lineales/KNN (ROC-AUC ≈ 0.73 vs ≈ 0.66).
2. La calibración mejora el Brier en todos los casos, requisito indispensable
   para traducir probabilidades a odds honestos.
3. El modelo final (RandomForest calibrado) es el punto de partida para generar
   cuotas; queda pendiente validar su rentabilidad real contra una casa de
   apuestas (backtesting de odds).

## 10. Cómo reproducir

```bash
cd "<repo>"
export PATH="$HOME/.local/bin:$PATH"
# full (todos los datos) — puede tardar varios minutos por KNN/RF:
uv run --python .venv python -m src.models \
  --input data/processed/features.parquet
# demo rápida sobre los últimos 20 000 partidos:
uv run --python .venv python -m src.models \
  --input data/processed/features.parquet --max-rows 20000
# smoke test con data sintética (si F3 no ha corrido):
uv run --python .venv python -m src.models --synthetic
# walk-forward OOS (Capa 2) — probabilidades/odds por partido (~2 min):
uv run --python .venv python -m src.walkforward \
  --input data/processed/features.parquet
# export del modelo estrella a CSVs (Power BI):
uv run --python .venv python -m src.bi_export --input-dir data
```

Desde código:

```python
import sys; sys.path.insert(0, "src")
from models import load_features, train_evaluate, predict_match

df = load_features("data/processed/features.parquet")
results = train_evaluate(df, return_full=True)          # dict completo
results_df = train_evaluate(df)                          # solo la tabla

pred = predict_match("Team A", "Team B", {
    "f_rating_diff": 0.3, "f_rank_diff": 12.0, "f_form_diff": 0.5,
    "f_winrate_diff": 0.05, "f_mapscore_diff": 0.1, "f_experience_diff": -2.0,
    "f_rest_diff": 1.0, "f_streak_diff": 1.0, "f_h2h_winrate": 0.6,
})
```
