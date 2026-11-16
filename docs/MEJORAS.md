# Opciones de mejora del modelo

Ideas documentadas para mejorar la predicción en futuras iteraciones. Ninguna es
necesaria para el entregable actual.

## 1. Capa 2 — features por jugador (la más impactante)

**Qué es**: scrapear los *detalles* de cada partido (`/v2/match/{id}/details`) para
obtener estadísticas **por jugador y por mapa** (kills/muertes/asistencias, ACS, rating).

**Por qué no está hecho**: es carísimo. ~108 448 partidos × 20 req/min ≈ **días** de
scraping continuo.

**Qué aportaría**:
- Features a nivel de jugador (rating promedio del roster, ACS del jugador estrella,
  etc.).
- Un `dim_player` y un `fact` a nivel de mapa en el BI.
- Posible mejora del AUC (las features por jugador suelen sumar señal).

**Cómo hacerlo** (si se decide):

```bash
python local/03_etl_match_details.py --limit 500      # prueba
python local/03_etl_match_details.py                  # completo (días)
```

Luego re-ejecutar `03_feature_engineering` y `04_modelado` con las tablas nuevas
(`match_details`, `match_maps`, `match_player_stats`).

## 2. Ranking histórico (quitar el sesgo de "foto actual")

`f_rating_diff` y `f_rank_diff` usan el ranking **actual** de vlr.gg, no el ranking en
la fecha del partido. Para partidos antiguos eso es "información futura" (leakage leve).

**Mejora**: scrapear un ranking histórico (snapshots por fecha) o, al menos, reconocer
el sesgo y evaluar el modelo **sin** esas dos features para cuantificar cuánto aportan
(de hecho ya son las de menor correlación con `y`: ~0.10 y −0.12).

## 3. Normalizar BO1 vs BO3 en `f_mapscore_diff`

El EDA detectó que los marcadores mezclan **dos escalas**:
- **70.7%** por mapas (BO3/BO5, valores 0–3).
- **29.3%** por rondas (BO1, hasta 25).

`f_mapscore_diff` mezcla ambas escalas. **Mejora**: normalizar el margen según el tipo
de serie (o añadir un flag `es_bo1`) para no mezclar magnitudes.

## 4. Excluir `gc` (Game Changers) del filtro de rankeados

El EDA mostró que **`gc` (Game Changers) es la "región" más grande del ranking**
(508 equipos = 31.5%), más que cualquier región geográfica. Son equipos de un circuito
que casi no juega contra el VCT principal.

**Mejora**: excluir `region == "gc"` (y quizá otros circuitos separados) al filtrar
"equipos rankeados", o entrenar un modelo separado por circuito. Esto quita ruido si el
objetivo es predecir el circuito principal.

## 5. Descartar `f_rest_diff`

`f_rest_diff` (días de descanso) tiene correlación con el target ≈ **0** (−0.004). Es
candidata a descartar para simplificar el modelo sin perder rendimiento.

## 6. Otras ideas menores

- **LightGBM**: en WSL falta `libgomp` y se omite; en Colab sí está disponible. Probar
  si añade algo sobre XGBoost/RandomForest.
- **Calibración**: el walk-forward usa RandomForest crudo; añadir calibración
  (Platt/isotónica) por fold podría mejorar el Brier.
- **Más historial por equipo**: `max_team_match_pages` está en 3; subirlo da más forma
  histórica (a costa de más scraping).
