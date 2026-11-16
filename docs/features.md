# Feature Engineering (F3) — Diccionario de features

Módulo: `src/features.py`. Genera `processed/features.parquet` (+ `features.csv` para
Power BI) a partir de `interim/matches.parquet` y `interim/rankings.parquet`
(opcionalmente `teams`, `team_stats`, `team_profiles` si ya existen).

## Convenciones

- **Target** `y` ∈ {0, 1}: `1` si gana `team1`, `0` si gana `team2`. Se descartan
  `winner` ∈ {draw, None} y equipos *placeholder* (`forfeit`, `bye`, `tbd`,
  `n/a`, `null`, `none`, …).
- **Todas las features** tienen prefijo `f_`, son numéricas y representan un
  **diferencial `team1 − team2`** salvo `f_h2h_winrate` (probabilidad directa).
- **Señal**: en general, `> 0` favorece a `team1`, `< 0` favorece a `team2`.
  Excepción documentada: `f_rank_diff` (ver abajo).

## Tabla de features

| Feature | Definición / fórmula | Fuente | Fuga |
|---|---|---|---|
| `f_rating_diff` | `rating(team1) − rating(team2)`. Rating VLR global (0–2000). Equipo sin ranking → rating `0` (peor posible). | `rankings.rating` | Foto actual* |
| `f_rank_diff` | `rank(team1) − rank(team2)`. **Rank menor = mejor** ⇒ señal invertida: `f_rank_diff < 0` favorece a `team1`. Equipo sin ranking → `worst_rank = max(rank)+1` (peor que todos) para mantener la señal coherente. | `rankings.rank` | Foto actual* |
| `f_form_diff` | `forma(team1) − forma(team2)`, con `forma = % victorias en los últimos 5 partidos previos`. Sin historial → `0.5`. | `matches` (rolling) | Libre |
| `f_winrate_diff` | `winrate(team1) − winrate(team2)`, con `winrate = % victorias en los últimos 20 partidos previos`. Sin historial → `0.5`. | `matches` (rolling) | Libre |
| `f_h2h_winrate` | % de victorias de `team1` contra `team2` en enfrentamientos **previos** (0–1). Sin historial → `0.5`. | `matches` (acumulado por par) | Libre |
| `f_mapscore_diff` | `margen_medio(team1) − margen_medio(team2)`, con `margen = score a favor − score en contra`, promediado sobre los últimos 5 previos. Captura dominancia (no solo ganar/perder). | `matches` (rolling) | Libre |
| `f_experience_diff` | `nº partidos previos(team1) − nº partidos previos(team2)` (experiencia/rodaje). | `matches` (cumcount) | Libre |
| `f_rest_diff` | `días_descanso(team1) − días_descanso(team2)`, con descanso = días desde el partido previo del equipo. | `matches` (diff fechas) | Libre |
| `f_streak_diff` | `racha(team1) − racha(team2)`, con racha firmada (+N victorias seguidas / −N derrotas) antes del partido. | `matches` (acumulado) | Libre |

\* **Foto actual**: `rankings` es una instantánea del ranking actual; no hay ranking
histórico en el dataset. Para partidos antiguos estas dos features usan información
posterior (sesgo de supervivencia). Es una limitación conocida y aceptada para esta
iteración; el resto de features históricas son estrictamente libres de fuga.

## Features opcionales (si existen las tablas de equipo)

| Feature | Definición | Fuente | Fuga |
|---|---|---|---|
| `f_map_winrate_diff` | `win_pct` medio en los mapas jugados (team1 − team2). Sin datos → `0.5`. | `team_stats.win_pct` | Foto actual* |
| `f_profile_rating_diff` | `rating` del perfil de equipo (team1 − team2). | `team_profiles.rating.rating` | Foto actual* |
| `f_winnings_diff` | `log1p(total_winnings)` (team1 − team2). | `team_profiles.total_winnings` | Foto actual* |

\* Igual que las dos features de ranking, estas tres usan el valor vigente en la
fecha de descarga. `f_map_winrate_diff` es la más afectada: el `win_pct` de toda la
trayectoria del equipo incluye los mismos partidos que se predicen. En total,
**5 de las 12 features son de foto actual** y 7 son históricas libres de fuga.

Estas columnas solo aparecen si `interim/team_stats.parquet` y/o
`interim/team_profiles.parquet` existen al ejecutar `build_features`. El código las
detecta con `Path.exists()` y las añade automáticamente.

## Reglas anti-fuga implementadas

1. Se parsea `date` (string `"Thu, October 22, 2026"`) + `time` (`"1:00 PM"`) a un
   timestamp único; se ordena cronológicamente (desempate por `match_id` numérico).
2. El historial por equipo se apila en formato largo (una fila por equipo-partido) y
   se calcula con `shift(1)` + `rolling(N, min_periods=1)`: el partido **actual** y
   los **posteriores** nunca entran en la ventana.
3. El head-to-head se acumula en una sola pasada cronológica por par de equipos;
   solo se lee el estado **anterior** al partido.
4. Se conservan `match_id`, `team1`, `team2`, `date`, `date_ts` (epoch, segundos),
   `event`, `stage`, `score1`, `score2` para trazabilidad y para el **split temporal**
   (que se hace en la fase de modelado, no aquí).

## Decisiones y supuestos

- **Fechas**: las 25 filas con fecha relativa (`"20h 49m ago"`, endpoint `q=results`)
  y la fila con `"Wed, December 31, 1969"` (artefacto de epoch) se **descartan**
  por no tener fecha absoluta fiable (~26 filas de ~108k).
- **`unix_timestamp`**: en este dataset es en realidad un *datetime string*
  (`"2026-11-06 18:45:00"`), no epoch; se usa solo como respaldo cuando `date` no
  parsea.
- **Nombres de equipo**: se normalizan (minúsculas, sin acentos, puntuación → espacio)
  para agrupar historial y unir rankings, evitando partir un equipo por `"LEVIATÁN"`
  vs `"leviatan"`. El nombre original se conserva en `team1`/`team2`.
- **Rankings duplicados entre regiones**: se toma el `rating` máximo y el `rank`
  mínimo por equipo.
- **Equipo sin ranking**: `rating → 0` y `rank → max(rank)+1` (tratado como peor que
  todos los rankeados). Documentado arriba.
- **Imputación sin historial**: `form`/`winrate`/`h2h` → `0.5`; `mapscore`/`rest` → `0`.

## Contrato del `features.parquet` (para la fase de modelado)

| Columna | Tipo | Descripción |
|---|---|---|
| `match_id` | `str` | id del partido (vlr.gg) |
| `team1` | `str` | nombre original del equipo A |
| `team2` | `str` | nombre original del equipo B |
| `date` | `str` | fecha original (string) |
| `date_ts` | `int64` | timestamp epoch en segundos (para split temporal) |
| `event` | `str` | torneo |
| `stage` | `str` | fase del torneo |
| `score1` | `float64` | marcador de team1 (metadata, **no usar como feature**) |
| `score2` | `float64` | marcador de team2 (metadata, **no usar como feature**) |
| `y` | `int8` | target: 1 si gana team1, 0 si gana team2 |
| `f_*` | `float64` | features diferenciales (ver tabla de arriba) |

> **Para modelar**: usar SOLO las columnas `f_*` como features, `y` como target y
> `date_ts` para el split temporal. `score1`/`score2` son resultado del partido
> (disponibles solo *a posteriori*) y **no** deben entrar como features.
