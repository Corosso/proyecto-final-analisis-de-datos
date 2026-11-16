# Guía de ejecución (local y Colab)

Este proyecto tiene **dos modos de ejecución** claramente separados:

| Modo | Carpeta | Qué hace | Cuándo |
|---|---|---|---|
| **Local** | `local/` | Setea la API y **scrapea** los datos | Una sola vez, en tu máquina (WSL) |
| **Colab** | `colab/` | **Analiza** los datos ya scrapeados | Siempre que quieras explorar/modelar |

Los datos scrapeados se versionan en `data/` y se copian a Google Drive para Colab.

---

## 1. Ejecución local (scraping)

### Requisitos

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (gestor de paquetes)
- `git`

### Pasos

```bash
# 0. (una vez) Entorno del proyecto
uv venv .venv
uv pip install -r requirements.txt

# 1. (una vez) Levanta la API local (clona vlrggapi + arranca el servidor)
python local/01_setup_api.py

# 2. Scrapea la Capa 1 (partidos + rankings + equipos). Idempotente.
python local/02_etl_matches.py

# 3. (OPCIONAL) Capa 2: stats por jugador. Lento, no necesario.
python local/03_etl_match_details.py --limit 500
```

> `02_etl_matches.py` usa un **checkpoint** (`data/_checkpoint/`): si se interrumpe,
> vuelve a ejecutar el mismo comando y retoma donde quedó.

### Dónde se guardan los datos

La ruta de datos la resuelve `src/config.py` automáticamente:

1. Si está la variable `VALORANT_DATA_DIR` → usa esa ruta.
2. Si corre en **Colab** → `/content/drive/MyDrive/Analisis de datos/Proyecto final/data`.
3. Si existe `/home/coro/valorant-data` (**WSL**) → esa (evita los locks de OneDrive).
4. Si no → `data/` dentro del repo.

Para forzar una ruta:

```bash
VALORANT_DATA_DIR=/ruta/a/mis/datos python local/02_etl_matches.py
```

---

## 2. Ejecución en Colab (análisis)

1. Sube a Google Drive la carpeta del proyecto a
   `My Drive/Analisis de datos/Proyecto final` (necesitas `data/`, `src/` y `colab/`).
2. Abre un notebook de `colab/` desde Drive.
3. Ejecuta la **primera celda** (monta Drive y autoriza el acceso).
4. Ejecuta el resto. La celda 2 apunta `ROOT` a tu carpeta de Drive (si tu carpeta
   tiene otro nombre, cambia `ROOT` ahí).

Orden recomendado:

```
01_eda → 02_visualizacion → 03_feature_engineering → 04_modelado → 05_bi_export
```

### Qué datos necesita cada notebook (subir a Drive)

| Notebook | Necesita de `data/` |
|---|---|
| `01_eda` | `interim/` (matches, rankings, teams, team_stats, team_profiles, team_rosters) + `processed/features.parquet` |
| `02_visualizacion` | `interim/` (matches, team_stats, rankings, teams) |
| `03_feature_engineering` | `interim/` (matches, rankings, teams, team_stats, team_profiles) |
| `04_modelado` | `processed/features.parquet` (+ `processed/best_model.joblib` para predecir) |
| `05_bi_export` | `interim/` + `processed/probabilities.parquet` |

---

## 3. Estructura de `data/`

```
data/
├── interim/      # tablas del scraping (matches, rankings, teams, team_stats, ...)
├── processed/    # features.parquet, best_model.joblib, probabilities.parquet
└── bi/           # CSVs del modelo estrella para Power BI
```

> `data/raw/` (JSONs crudos del scraping) **no** se versiona ni se sube a Drive:
> son cientos de MB y no hacen falta para el análisis.

---

## 4. Referencias

- `docs/api_endpoints.md` — endpoints de la API y sus esquemas.
- `docs/features.md` — definición de cada feature `f_*`.
- `docs/modelado.md` — metodología y resultados del modelo.
- `docs/diccionario_datos.md` — modelo estrella para Power BI.
- `docs/MEJORAS.md` — ideas para mejorar el modelo.
