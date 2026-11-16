# Predicción del ganador de partidos de Valorant

Modelo de aprendizaje supervisado que estima **quién gana un partido profesional de
Valorant** y lo expresa como **odds decimales** (favorito / evens / underdog), a partir
de datos históricos de [vlr.gg](https://www.vlr.gg) extraídos con la API no oficial
[`vlrggapi`](https://github.com/axsddlr/vlrggapi).

> Proyecto final de la asignatura **Análisis de datos** (ETL · EDA · BI · ML).

---

## Estructura del proyecto

```
.
├── local/                ← scripts .py: setean la API y scrapean (solo en tu máquina)
│   ├── 01_setup_api.py   ← clona vlrggapi, instala deps y arranca el servidor
│   ├── 02_etl_matches.py ← scrapea la Capa 1 (partidos + rankings + equipos)
│   └── 03_etl_match_details.py  ← OPCIONAL: Capa 2 (stats por jugador)
├── colab/                ← notebooks: análisis en Colab (con datos ya scrapeados)
│   ├── 01_eda.ipynb                ← análisis exploratorio
│   ├── 02_visualizacion.ipynb      ← gráficas de mapas/equipos/marcadores
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modelado.ipynb
│   └── 05_bi_export.ipynb
├── src/                  ← módulos compartidos (los importan local/ y colab/)
├── data/                 ← datos de entrega (versionados): interim/ + processed/ + bi/
├── docs/                 ← documentación técnica
├── dashboard.pbix        ← dashboard de Power BI
├── requirements.txt
└── README.md
```

## ¿Qué hace cada paquete?

| Paquete / archivo | Responsabilidad |
|---|---|
| `local/` | **Ejecución local** del scraping. Son scripts `.py` que no necesitan Jupyter. |
| `local/01_setup_api.py` | Clona `vlrggapi`, instala sus dependencias (`uv sync`) y arranca el servidor en `127.0.0.1:3001`. |
| `local/02_etl_matches.py` | Scrapea la **Capa 1**: listado de partidos, rankings regionales y datos de equipo (perfiles, stats por mapa, historial, rosters). Idempotente con checkpoint. |
| `local/03_etl_match_details.py` | **Opcional**. Scrapea la **Capa 2** (stats por jugador/mapa). No es necesaria para el modelo; ver `docs/MEJORAS.md`. |
| `colab/` | **Ejecución en Google Colab** del análisis, con los datos ya scrapeados (subidos a Drive). |
| `colab/01_eda.ipynb` | Análisis exploratorio: calidad de datos, distribuciones, correlaciones. |
| `colab/02_visualizacion.ipynb` | Gráficas de mapas, equipos y marcadores (sin re-scrapear). |
| `colab/03_feature_engineering.ipynb` | Genera el dataset de features `f_*` + target `y`. |
| `colab/04_modelado.ipynb` | Entrena y evalúa los modelos, con odds y tier. |
| `colab/05_bi_export.ipynb` | Exporta el modelo estrella (CSVs) para Power BI. |
| `src/config.py` | Resuelve la ruta de datos según el entorno (Colab / WSL / local). |
| `src/scraper.py` | Cliente HTTP de la API: throttle, reintentos y checkpoint. |
| `src/etl.py` | Orquestación del scraping (Capas 1 y 2). |
| `src/eda.py` | Helpers de carga y visualización para el EDA. |
| `src/features.py` | Feature engineering (features `f_*`, sin fuga de datos). |
| `src/models.py` | Modelado, evaluación y conversión probabilidad → odds. |
| `src/walkforward.py` | Probabilidades out-of-sample por partido. |
| `src/bi_export.py` | Construye el modelo estrella y exporta los CSVs. |
| `data/` | Datos de entrega: `interim/` (tablas), `processed/` (features + modelo), `bi/` (CSVs). |
| `docs/` | `EJECUCION.md` (runbook), `MEJORAS.md` (mejoras), `diccionario_datos.md`, `api_endpoints.md`, `features.md`, `modelado.md`. |

## Pipeline (cómo fluyen los datos)

```
vlr.gg ── vlrggapi ──► src/scraper.py ──► src/etl.py ──► data/interim/*.parquet
                                                              │
                                    src/features.py ──────────┘
                                                              ▼
                                              data/processed/features.parquet
                                                              │
                     src/models.py ──► best_model.joblib       │
                     src/walkforward.py ──► probabilities.parquet
                                                              │
                     src/bi_export.py ──► data/bi/*.csv ──► Power BI
```

Cada fase lee archivos de la fase anterior y escribe para la siguiente.

## Cómo ejecutar

### En Colab (análisis)

1. Sube a Google Drive la carpeta del proyecto en
   `My Drive/Analisis de datos/Proyecto final` (incluyendo `data/`, `src/` y `colab/`).
2. Abre un notebook de `colab/` desde Drive y ejecuta las celdas en orden.
   La primera celda monta Drive; la segunda apunta `ROOT` a tu carpeta.
3. Orden recomendado: `01_eda` → `02_visualizacion` → `03_feature_engineering` →
   `04_modelado` → `05_bi_export`.

### En local (scraping)

```bash
# 1. Dependencias del proyecto
uv venv .venv && uv pip install -r requirements.txt

# 2. Levanta la API (una vez)
python local/01_setup_api.py

# 3. Scrapea la Capa 1
python local/02_etl_matches.py

# (Opcional) Capa 2 — lento, no necesario
python local/03_etl_match_details.py --limit 500
```

> Requisitos locales: Python 3.11+, `uv` y `git`. Ver `docs/EJECUCION.md` para el detalle.

## Resultados clave

- **Datos**: 108 448 partidos · 2 992 eventos · 20 902 equipos · 1 614 rankeados.
- **Modelo**: RandomForest con **AUC ≈ 0.71** (split temporal) y probabilidades
  out-of-sample (walk-forward) con cobertura del 90.9 %.
- **Salida**: probabilidad + odds (`1/p`) + tier (Favorito / Evens / Underdog).

Ver `docs/modelado.md` para la evaluación completa y `docs/diccionario_datos.md` para
el modelo de BI.
