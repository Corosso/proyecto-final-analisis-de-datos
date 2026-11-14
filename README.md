# Proyecto final Análisis de Datos — Predicción de partidos de Valorant

Modelo predictivo que estima el ganador de un próximo partido de Valorant,
expresado como **odds decimales tipo casa de apuestas** (favorito / evens / underdog).

## Objetivo

Predecir el resultado de partidos profesionales de Valorant usando aprendizaje
supervisado, a partir de datos históricos extraídos de [vlr.gg](https://www.vlr.gg/)
mediante la API no oficial [vlrggapi](https://github.com/axsddlr/vlrggapi).

El *target* es binario (¿ganó el equipo A?), pero la predicción se expresa como
**odds** y se categoriza en una escala de favoritismo.

## Target y escala de odds

1. El modelo predice la probabilidad calibrada `P(A gana)`.
2. Se convierte a odds "fair": `odds_A = 1 / P(A)`, `odds_B = 1 / (1 - P(A))`.
3. Se mapea a una escala interpretable:

| Odds del equipo | Etiqueta |
|---|---|
| ≤ 1.40 | Favorito claro |
| 1.41 – 1.80 | Favorito |
| 1.81 – 2.20 | Evens / parejo |
| 2.21 – 3.00 | Underdog |
| > 3.00 | Underdog claro |

> Los odds **reales de bookmakers** se agregan como mejora futura (fuente extra de datos).

## Fuente de datos

- **vlrggapi**: API REST no oficial (Python/FastAPI) que scrapea vlr.gg.
- La instancia hosteada está caída → se **clona el repo y se usa localmente** para descargar los datos.
- Endpoints clave: `/v2/match?q=results`, `/v2/match/{id}/details`, `/v2/team/{id}/stats`,
  `/v2/team/{id}/matches`, `/v2/team/{id}/roster`, `/v2/rankings`, `/v2/stats`.
- Rate limits (endpoints caros): **20 req/min**. El scraper hace throttling, reintentos y checkpoint.

## Fases

| Fase | Descripción | Entregable |
|---|---|---|
| F0 | Setup en Google Colab | `notebooks/00_setup_api.ipynb` |
| F1 | ETL (scraping completo, checkpoint/resume) | `src/scraper.py`, `src/etl.py`, `notebooks/01`, `02` |
| F2 | EDA (limpieza y visualización) | `notebooks/03_eda.ipynb` |
| F3 | Feature engineering (diferenciales + split temporal) | `src/features.py`, `notebooks/04` |
| F4 | Modelado (varios modelos + tuning + calibración) | `src/models.py`, `notebooks/05_modelado.ipynb` |
| F5 | Export BI (modelo estrella para Power BI) | `notebooks/06_bi_export.ipynb` |
| F6 | Documentación e informe | `docs/` |

## Estructura

```
.
├── data/
│   ├── raw/        # respuestas JSON crudas
│   ├── interim/    # tablas intermedias (Parquet/CSV)
│   ├── processed/  # dataset final de features
│   └── bi/         # modelo estrella para Power BI
├── src/            # código reutilizable (scraper, etl, features, models)
├── notebooks/      # notebooks de Colab por fase
└── docs/           # diccionario de datos, informe, referencias de API
```

## Stack

Python 3.11+, pandas, numpy, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, pyarrow.
Persistencia en Google Drive (Colab). BI en Power BI Desktop.

## Ejecución (Colab)

1. Sube/clona este repo a Colab (o monta Drive).
2. Abre `notebooks/00_setup_api.ipynb`: monta Drive, clona `vlrggapi`, instala dependencias.
3. Sigue `01` → `06` en orden.
