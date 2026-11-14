# Diccionario de datos — Modelo estrella (F5 · Business Intelligence)

> Proyecto: **Predicción del ganador de partidos profesionales de Valorant**.
> Fuente: datos de [vlr.gg](https://www.vlr.gg/) procesados por las fases ETL (F1),
> feature engineering (F3) y modelado (F4).
> Salida: **CSVs en `data/bi/`** listos para abrir en **Power BI Desktop**.

---

## 1. Resumen del modelo

Modelo de datos en **estrella** (schema de copo de nieve no es necesario): una tabla
de hechos `fact_matches` rodeada de dos dimensiones `dim_team` y `dim_event`.

| Tabla | Rol | Granularidad | Filas |
|---|---|---|---|
| `dim_team` | Dimensión | 1 fila por equipo | 20 286 |
| `dim_event` | Dimensión | 1 fila por torneo | 2 992 |
| `fact_matches` | Hechos | 1 fila por partido | 107 242 |
| `dim_map` | Dimensión (tabla suelta) | 1 fila por (equipo, mapa) | 20 124 |

CSVs exportados en **UTF-8 con BOM** (`utf-8-sig`) para que Power BI/Excel detecten
los acentos correctamente.

```
                 ┌─────────────────────┐
                 │      dim_team       │
                 │─────────────────────│
                 │ team_id         (PK)│◄────────────────────────┐
                 │ team_name · logo    │                         │
                 │ team_name_norm      │        ┌────────────────┼────┐
                 │ region · country    │        │                │    │
                 │ country_code/name   │        │                │    │
                 │ rank · rating       │        │                │    │
                 │ profile_rating/rank │        │ (team1_id)     │    │ (team2_id)
                 │ record · wins/losses│        │                │    │
                 │ earnings_usd        │        │                │    │
                 │ total_winnings_usd  │        │                │    │
                 │ roster_size · streak│        │                │    │
                 └─────────────────────┘        │                │    │
                         ▲                      │                │    │
                         │ (winner_id)          │                │    │
                         │                      ▼                ▼    │
┌─────────────────────┐  │            ┌─────────────────────────────────┐
│      dim_event      │  │            │          fact_matches          │
│─────────────────────│  │            │─────────────────────────────────│
│ event_id        (PK)│◄─┼────────────│ match_id      (clave natural)   │
│ event_name          │  │            │ team1_id           (FK dim_team)│
│ num_partidos        │  └────────────│ team2_id           (FK dim_team)│
│ fecha_min · fecha_max│             │ winner_id          (FK dim_team)│
└─────────────────────┘  (event_id)  │ event_id           (FK dim_event)│
                                     │ stage · fecha · date_ts         │
                                     │ score1 · score2 · margin        │
                                     │ y (resultado real)              │
                                      │ prob_team1 · odds · tier        │
                                     │ f_* (features del modelo)     │
                                     └─────────────────────────────────┘
```

`dim_map` es una **tabla suelta** (sin FK al fact, porque el fact no guarda el
mapa del partido):

```
┌───────────────────────┐
│        dim_map        │
│───────────────────────│
│ team_id · team_name   │
│ map · games           │
│ wins · losses         │
│ win_pct · atk/def %   │
└───────────────────────┘
```

---

## 2. Reglas de normalización y supuestos

### 2.1 Normalización de nombres de equipo

- `norm_name(s)`: **NFKD → eliminar acentos → minúsculas → puntuación/símbolos a
  espacio → colapsar espacios → recortar**. Así `"LEVIATÁN"`, `"Leviatán"` y
  `"leviatan"` colapsan a la misma clave `"leviatan"`.
- Es **idéntica** a `features.norm_name` (fase F3), de modo que las claves de
  `dim_team` son coherentes con las features `f_*` del modelo.
- `norm_key(s)`: si `norm_name` devuelve cadena vacía (nombres íntegramente
  **no latinos**: CJK, cirílico, árabe…), conserva el original normalizado en
  Unicode (NFKC + minúsculas + espacios colapsados). Evita fusionar equipos
  distintos como `"雨のち晴れ"` y `"АОГИРИ"`. Solo afecta a 6 equipos de rankings
  que no aparecen en ningún partido del fact.

### 2.2 Supuestos y decisiones

1. **`team_id` estable**: no existe un id real de equipo en el dataset, así que se
   genera un entero `1..N` ordenando alfabéticamente la clave normalizada. Es
   determinista (estable entre ejecuciones mientras no cambie el universo de equipos).
2. **Duplicados de `rankings` entre regiones**: un mismo equipo puede aparecer en
   varias regiones (p. ej. `"101 Outlaws"` en `ap` y `gc`). Se conserva la entrada
   de **mejor `rank`** (menor número) y, en empate, la de **mayor `rating`**.
3. **Universo de `dim_team`**: unión de (a) equipos que aparecen en `features`
   (los que referenciará el fact) y (b) equipos de `rankings`. Hay ~18 700 equipos
   que solo existen en partidos (equipos pequeños sin ranking); sus atributos de
   ranking (`region`, `rating`, `rank`, …) quedan vacíos.
4. **`y` es el resultado real (ground truth)**, no la predicción: `1` = gana
   `team1`, `0` = gana `team2`. La **probabilidad por partido** del modelo se
   obtiene por **walk-forward out-of-sample** (ventana expandible, 10 folds) y se
   exporta en `fact_matches` como `prob_team1`/`odds_team1`/`odds_team2`/`tier`
   (ver §6.5 y `docs/modelado.md`). El ~9 % más antiguo no tiene pasado para
   entrenar y queda con `prob_team1 = NULL` (lo honesto).
5. **Empates y fechas inválidas**: la fase F3 ya descartó partidos con
   `winner ∈ {draw, None}`, equipos placeholder (`forfeit`, `bye`, `tbd`, …) y
   ~26 filas con fecha relativa o artefacto de epoch. Por eso `fact_matches`
   (107 242) tiene menos filas que `matches` (108 448).
6. **`dim_event`** se construye desde `matches.event` (todos los torneos, 2 992).
   `num_partidos` y el rango de fechas se calculan sobre `matches`; 4 torneos
   quedan sin rango de fechas porque sus partidos tenían fecha relativa.
7. **Tablas de equipo** (`teams`, `team_profiles`, `team_stats`): ya disponibles.
   Se incorporan en Capa 2: `team_profiles` enriquece `dim_team` (logo, país,
   rating de perfil, ganancias y tamaño de plantilla) y `team_stats` + `teams`
   alimentan la dimensión `dim_map` (win rate por mapa). Solo los ~1 548 equipos
   con perfil rankeado tienen estos atributos; el resto quedan vacíos.
8. **`dim_map` es una tabla suelta**: su clave natural es `(team_id, map)` y usa el
   `team_id` de vlr.gg (string). **No** se conecta a `fact_matches` porque el fact
   no guarda en qué mapa se jugó cada partido; sirve para análisis independientes
   de rendimiento por mapa.

---

## 3. Tabla `dim_team`

Dimensión de equipos. Clave primaria: `team_id`.

| Columna | Tipo | Descripción |
|---|---|---|
| `team_id` | entero | **PK**. Id numérico estable generado por nombre normalizado. |
| `team_name` | texto | Nombre de display. Preferencia: nombre oficial de `rankings`; si no está rankeado, la forma más frecuente en los partidos. |
| `team_name_norm` | texto | Clave normalizada (`norm_key`). Para trazabilidad / depuración. |
| `logo` | texto | URL del logo del equipo (desde `team_profiles`). Vacío si no rankeado. |
| `region` | texto | Región VLR (`na`, `eu`, `gc`, `ap`, `br`, `kr`, `jp`, `cn`, `col`, `oce`, …). Vacío si el equipo no está en rankings. |
| `country` | texto | País del equipo (nombre completo; preferencia `rankings`, relleno desde `team_profiles`). |
| `country_code` | texto | Código ISO de 2 letras del país (desde `team_profiles.country`). Vacío si no rankeado. |
| `country_name` | texto | Nombre completo del país (desde `team_profiles.country_name`). |
| `rank` | entero | Posición en el ranking mundial (menor = mejor). Vacío si no rankeado. |
| `rating` | entero | Rating VLR (0–2000). Vacío si no rankeado. |
| `profile_rating` | entero | Rating actual del perfil del equipo (`team_profiles.rating['rating']`). Vacío si no rankeado. |
| `profile_rank` | entero | Posición actual del perfil (`team_profiles.rating['rank']`). Vacío si no rankeado. |
| `record` | texto | Récord histórico `"W–L"` (victorias–derrotas). |
| `wins` | entero | Nº de victorias (parseado de `record`). |
| `losses` | entero | Nº de derrotas (parseado de `record`). |
| `earnings_usd` | decimal | Ganancias acumuladas en USD (parseado de `"$1,191,150"`; relleno desde `total_winnings`). |
| `total_winnings_usd` | decimal | Ganancias totales del perfil (parseado de `"$1,583,636"` → `1583636`). |
| `roster_size` | entero | Nº de jugadores activos no-staff en la plantilla (`roster`). Vacío si no rankeado. |
| `streak` | texto | Racha actual, p. ej. `"8W"` / `"1L"`. |
| `streak_value` | entero | Magnitud de la racha (nº de partidos). |
| `streak_type` | texto | Sentido de la racha: `"W"` (victorias) o `"L"` (derrotas). |
| `en_ranking` | entero 0/1 | `1` si el equipo está en el ranking actual; `0` en caso contrario. |

---

## 4. Tabla `dim_event`

Dimensión de torneos/eventos. Clave primaria: `event_id`.

| Columna | Tipo | Descripción |
|---|---|---|
| `event_id` | entero | **PK**. Id numérico generado por título de evento (orden alfabético). |
| `event_name` | texto | Título del torneo (p. ej. `"Game Changers 2026: Championship São Paulo"`). |
| `num_partidos` | entero | Nº de partidos registrados del torneo (en `matches`). |
| `fecha_min` | texto (ISO `YYYY-MM-DD`) | Primera fecha de partidos del torneo. |
| `fecha_max` | texto (ISO `YYYY-MM-DD`) | Última fecha de partidos del torneo. |

---

## 5. Tabla `dim_map` (tabla suelta)

Win rate por mapa y por equipo. Clave natural: par `(team_id, map)`. **No se
relaciona con `fact_matches`** (el fact no guarda el mapa del partido); es una
tabla independiente para análisis de rendimiento por mapa.

| Columna | Tipo | Descripción |
|---|---|---|
| `team_id` | texto | Id de equipo en vlr.gg (string). Clave (junto a `map`). |
| `team_name` | texto | Nombre del equipo, resuelto vía `teams.parquet`; si no está, se usa `team_id`. |
| `map` | texto | Nombre del mapa (`Ascent`, `Bind`, `Haven`, `Split`, …). |
| `games` | entero | Nº de partidos del equipo en ese mapa. |
| `wins` | entero | Partidos ganados en ese mapa. |
| `losses` | entero | Partidos perdidos en ese mapa. |
| `win_pct` | decimal (0–1) | % de victorias en el mapa (parseado de `"67%"` → `0.67`). `NULL` si `games = 0`. |
| `atk_rwin_pct` | decimal (0–1) | % de rondas ganadas en ataque (round win rate). |
| `def_rwin_pct` | decimal (0–1) | % de rondas ganadas en defensa (round win rate). |

---

## 6. Tabla `fact_matches`

Tabla de hechos. Una fila = un partido. Clave natural: `match_id` (único).

### 6.1 Claves y descriptores

| Columna | Tipo | Descripción |
|---|---|---|
| `match_id` | texto | Id del partido en vlr.gg (clave natural, única). |
| `team1_id` | entero | **FK** → `dim_team.team_id` (equipo A). |
| `team2_id` | entero | **FK** → `dim_team.team_id` (equipo B). |
| `winner_id` | entero | **FK** → `dim_team.team_id` (equipo ganador). |
| `event_id` | entero | **FK** → `dim_event.event_id`. |
| `team1` | texto | Nombre original del equipo A (trazabilidad). |
| `team2` | texto | Nombre original del equipo B (trazabilidad). |
| `winner` | texto | `"team1"` o `"team2"`. |
| `stage` | texto | Fase del torneo (p. ej. `"Playoffs"`, `"Group Stage"`, `"Swiss Stage"`). |
| `status` | texto | Estado del partido (`"Completed"`). |
| `source` | texto | Origen de la fila (`"event_matches"` / `"results"`). |

### 6.2 Tiempo

| Columna | Tipo | Descripción |
|---|---|---|
| `date` | texto | Fecha original en texto (p. ej. `"Sat, April 25, 2020"`). |
| `fecha` | texto (ISO `YYYY-MM-DD`) | Fecha del partido (derivada de `date_ts`, UTC). **Usar esta en Power BI** para time intelligence. |
| `date_ts` | entero | Timestamp epoch en segundos (para el split temporal del modelo). |

### 6.3 Medidas

| Columna | Tipo | Descripción |
|---|---|---|
| `score1` | decimal | Rondas ganadas por `team1`. |
| `score2` | decimal | Rondas ganadas por `team2`. |
| `margin` | decimal | `score1 − score2`. Positivo = `team1` dominó el marcador. |
| `y` | entero 0/1 | **Resultado real** (target del modelo): `1` = gana `team1`, `0` = gana `team2`. |

### 6.4 Features del modelo (inputs `f_*`)

Todas numéricas; salvo `f_h2h_winrate`, representan el **diferencial `team1 − team2`**
(> 0 favorece a `team1`; excepción: `f_rank_diff`, ver tabla). Definición completa
en `docs/features.md`.

| Columna | Definición |
|---|---|
| `f_h2h_winrate` | % victorias de `team1` contra `team2` en enfrentamientos previos (0–1). |
| `f_rating_diff` | `rating(team1) − rating(team2)`. |
| `f_rank_diff` | `rank(team1) − rank(team2)`. **Señal invertida**: `< 0` favorece a `team1`. |
| `f_form_diff` | % victorias últimos 5 partidos (team1 − team2). |
| `f_winrate_diff` | % victorias últimos 20 partidos (team1 − team2). |
| `f_mapscore_diff` | Margen medio de puntos últimos 5 partidos (team1 − team2). |
| `f_experience_diff` | Nº partidos previos (team1 − team2). |
| `f_rest_diff` | Días de descanso (team1 − team2). |
| `f_streak_diff` | Racha firmada (team1 − team2). |

### 6.5 Predicciones (walk-forward out-of-sample)

Probabilidades de victoria y odds derivados, obtenidas con el **walk-forward de
ventana expandible** (10 folds) de `src/walkforward.py`. Cada partido se predice
con un modelo entrenado **solo con partidos anteriores**; por eso el ~9 % más
antiguo no tiene predicción (ver `docs/modelado.md`).

| Columna | Tipo | Descripción |
|---|---|---|
| `prob_team1` | decimal (0–1) | `P(team1 gana)` out-of-sample (RandomForest calibrado por construcción temporal). `NULL` en el ~9 % más antiguo. |
| `odds_team1` | decimal | Odds "fair" de `team1`: `1/prob_team1`. |
| `odds_team2` | decimal | Odds "fair" de `team2`: `1/(1-prob_team1)`. |
| `tier` | texto | Tier de favoritismo de `team1` según `odds_team1`: `Favorito claro` (≤1.40), `Favorito` (1.41–1.80), `Evens/parejo` (1.81–2.20), `Underdog` (2.21–3.00), `Underdog claro` (>3.00). |

---

## 7. Relaciones (para Power BI)

| Relación | Desde (FK) | Hacia (PK) | Cardinalidad | Activa |
|---|---|---|---|---|
| 1 | `fact_matches.team1_id` | `dim_team.team_id` | Muchos→1 | No (inactiva) |
| 2 | `fact_matches.team2_id` | `dim_team.team_id` | Muchos→1 | No (inactiva) |
| 3 | `fact_matches.winner_id` | `dim_team.team_id` | Muchos→1 | Sí |
| 4 | `fact_matches.event_id` | `dim_event.event_id` | Muchos→1 | Sí |

> `dim_team` es una **dimensión de rol** (role-playing dimension): las tres
> columnas `team1_id`, `team2_id` y `winner_id` apuntan a la misma tabla. Power BI
> crea tres relaciones; solo una puede estar activa a la vez. Usa
> `USERELATIONSHIP(fact[team1_id], dim_team[team_id])` en medidas DAX para análisis
> "desde el equipo A" o "desde el equipo B".

> **`dim_map` no se relaciona con el fact**: el fact no guarda el mapa del
> partido, así que no existe FK. Si en el futuro `fact_matches` incorporara la
> columna `map`, se podría relacionar `dim_map` (clave `(team_id, map)`) con el
> fact vía `team_id` + `map`. Por ahora es una tabla independiente para análisis
> de rendimiento por mapa/equipo.

---

## 8. Cómo cargar en Power BI Desktop

1. **Abrir Power BI Desktop** (Windows) → *Inicio* → *Obtener datos* → **Texto/CSV**.
2. Selecciona, uno a uno, los cuatro archivos de `data/bi/`:
   `dim_team.csv`, `dim_event.csv`, `fact_matches.csv` y `dim_map.csv` → *Cargar*.
   (El BOM UTF-8 hace que los acentos se detecten solos; si preguntara, elige
   **origen de archivo: 65001 UTF-8**.)
3. **Verificar tipos**: en *Transformar datos*, confirma que Power BI detectó:
   - `team_id`, `event_id`, `rank`, `rating`, `profile_rating`, `profile_rank`,
     `wins`, `losses`, `streak_value`, `roster_size`, `en_ranking`, `date_ts`,
     `y` → **número entero**.
   - `earnings_usd`, `total_winnings_usd`, `score1`, `score2`, `margin`,
     `prob_team1`, `odds_team1`, `odds_team2`, `win_pct`, `atk_rwin_pct`,
     `def_rwin_pct`, `f_*` → **número decimal**.
   - `fecha`, `fecha_min`, `fecha_max` → **fecha** (si no, cambia el tipo).
   - el resto → **texto**.
4. **Crear relaciones** (Power BI suele detectarlas solo; si no):
   - `fact_matches.team1_id` → `dim_team.team_id` (muchos→1)
   - `fact_matches.team2_id` → `dim_team.team_id` (muchos→1)
   - `fact_matches.winner_id` → `dim_team.team_id` (muchos→1, **activa**)
   - `fact_matches.event_id` → `dim_event.event_id` (muchos→1, **activa**)
   - `dim_map` queda **sin relación** (tabla suelta); úsala en páginas propias.
5. **Medidas DAX útiles**:

   ```dax
   Partidos        = COUNTROWS(fact_matches)
   WinrateTeam1    = DIVIDE(CALCULATE(COUNTROWS(fact_matches), fact_matches[y]=1), COUNTROWS(fact_matches))
   PartidosTeamA   = CALCULATE(COUNTROWS(fact_matches), USERELATIONSHIP(fact_matches[team1_id], dim_team[team_id]))
   VictoriasEquipo = CALCULATE(COUNTROWS(fact_matches), fact_matches[winner_id] = SELECTEDVALUE(dim_team[team_id]))
   MargenPromedio  = AVERAGE(fact_matches[margin])
   ```

---

## 9. Pendientes (Capa 2) — completados

- Sí **`dim_map`** (win rate por mapa por equipo): construido desde
  `team_stats.parquet` + `teams.parquet` (ver §5 y la tabla `dim_map`). Es una
  **tabla suelta** sin FK al fact (el fact no guarda el mapa del partido).
- Sí **Probabilidad por partido** (`P(team1 gana)`): generada con **walk-forward
  out-of-sample** (`src/walkforward.py`, 10 folds de ventana expandible), derivada
  a `odds_team1`/`odds_team2`/`tier` y exportada en `fact_matches` (ver §6.5).
  Cobertura OOS: 90,9 % (los partidos más antiguos, ~9 %, quedan `NULL`).
- Sí **Tablas de equipo** (`teams`, `team_profiles`): `dim_team` enriquecido con
  `logo`, `country_code`/`country_name`, `profile_rating`, `profile_rank`,
  `total_winnings_usd` y `roster_size` (ver §3).
