# Referencia de la API — vlrggapi (v2.0.0)

Referencia de los endpoints usados por el scraper, con esquemas **verificados contra el
código fuente** de [axsddlr/vlrggapi](https://github.com/axsddlr/vlrggapi) (`master`) y
contra su `API.md`. No se inventaron campos: los ejemplos JSON están copiados del
`API.md` real o deducidos de los *parsers* del scraper.

> **Estado**: la instancia hosteada `https://vlrggapi.vercel.app` está caída. Se clona el
> repo y se corre localmente (`python main.py` → `http://127.0.0.1:3001`). Ver
> `local/01_setup_api.py`.

---

## 1. Generalidades

- **Base URL local**: `http://127.0.0.1:3001`
- **Versión**: `GET /version` → `{"version": "2.0.0", "default_api": "v2"}`
- **Envelope V2**: las respuestas `200` siguen `{"status": "success", "data": {...}}`.
  El scraper **desenvuelve** y devuelve `data`.
- **Errores**: `400` con `{"detail": "<mensaje>"}` para validación; `429` con
  `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`; `502/504/500` para fallos
  de *upstream* (vlr.gg); `503` para *circuit breaker* abierto.
- **Caché**: caché TTL en memoria por endpoint (ver tabla). No hay que golpear el mismo
  recurso más veces de las necesarias.

### Forma de `data` (patrón dominante)

Casi todos los *list* endpoints devuelven:

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [ ... ],
    "meta": { ... }          // opcional (paginación)
  }
}
```

Algunos devuelven un objeto único en lugar de `segments` (`/v2/team/{id}/stats` sí usa
`segments`; `/v2/team` y `/v2/team/{id}/roster` meten **un solo objeto** en `segments[0]`).

---

## 2. Rate limits (crítico)

Ventana deslizante **por IP y por tier**. Un sub-recurso hereda el tier del padre.
Contadores por proceso (multi-worker = contadores independientes).

| Tier | Límite | Rutas |
|---|---|---|
| `cheap` | 200/min | `/health`, `/version` |
| `moderate` | 60/min | `/news`, `/events`, `/event/{id}`, `/events/matches`, `/rankings`, `/search`, `/match` con `q=upcoming` o `q=live_score` |
| `expensive` | **20/min** | `/match/details`, `/match/{id}/details`, `/match/{id}/logs`, `/stats`, `/player/*`, `/team/*`, `/match` con `q=results` o `q=upcoming_extended` |

**Headers**: las respuestas llevan `X-RateLimit-Limit` y `X-RateLimit-Remaining`. El
`429` lleva `Retry-After` (segundos) y el body `{"detail": {"error": "Rate limit
exceeded", "tier": "...", "retry_after": N, "limit": M}}`.

El scraper (`src/scraper.py`) aplica un intervalo mínimo por tier (≈ `60/limit` con
margen) y reintenta con *backoff* respetando `Retry-After`.

---

## 3. Tabla resumen de endpoints usados

| Endpoint | Tier | Caché | Parámetros clave | Esencial |
|---|---|---|---|---|
| `GET /version` | cheap | — | — | soporte |
| `GET /v2/health` | cheap | — | — | soporte |
| `GET /v2/match?q=results` | expensive | 60s | `q`, `num_pages`, `from_page`, `to_page`, `max_retries`, `request_delay`, `timeout` | **sí** (histórico reciente) |
| `GET /v2/match?q=upcoming` | moderate | 5 min | `q` | doc (inferencia futura) |
| `GET /v2/match?q=live_score` | moderate | 30s | `q` | doc (inferencia futura) |
| `GET /v2/events?q=completed` | moderate | 30 min | `q`, `page` | **sí** (enumerar eventos) |
| `GET /v2/events/matches?event_id=` | moderate | 10 min | `event_id` | **sí** (histórico completo) |
| `GET /v2/match/{id}/details` | expensive | 5 min | `light`, `include` | **sí** (features por jugador) |
| `GET /v2/team/{id}` | expensive | 30 min | — | **sí** (rating/rank) |
| `GET /v2/team/{id}/stats` | expensive | 10 min | `date_start` | **sí** (win rates por mapa) |
| `GET /v2/team/{id}/matches` | expensive | 10 min | `page` (1–100) | **sí** (forma reciente) |
| `GET /v2/team/{id}/roster` | expensive | 30 min | — | útil |
| `GET /v2/rankings?region=` | moderate | 1 hr | `region` | útil |
| `GET /v2/stats?region=&timespan=` | expensive | 30 min | `region`, `timespan`, `event_id`, `map`, `min_rounds`, `min_rating` | útil (F3) |
| `GET /v2/search?q=` | moderate | 5 min | `q` | soporte (nombre→id) |

---

## 4. Esquemas exactos (ejemplos reales)

### 4.1 `GET /version`

```json
{ "version": "2.0.0", "default_api": "v2" }
```

### 4.2 `GET /v2/health`

```json
{
  "status": "success",
  "data": {
    "service": { "status": "Healthy" },
    "http_client": { "status": "Healthy", "status_code": null }
  }
}
```

### 4.3 `GET /v2/match?q=results`

Paginado. **Ventana de páginas máx. 20 por request**, `num_pages`/`from_page`/`to_page`
acotados a `1..100` (ver §5). Cada segmento (campo `match_id` **sí** está presente, y
`match_page` es URL completa; el ejemplo del `API.md` legacy está desactualizado en eso):

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "team1": "Team Vitality",
        "team2": "Gentle Mates",
        "score1": "0",
        "score2": "2",
        "flag1": "flag_eu",
        "flag2": "flag_fr",
        "time_completed": "2h 44m ago",
        "round_info": "Regular Season-Week 4",
        "tournament_name": "Champions Tour 2024: EMEA Stage 1",
        "match_page": "https://www.vlr.gg/318931/team-vitality-vs-gentle-mates-...",
        "tournament_icon": "https://owcdn.net/img/...",
        "page_number": 1,
        "match_id": "318931"
      }
    ],
    "meta": {
      "page_range": "1-1",
      "total_pages_requested": 1,
      "successful_pages": 1,
      "failed_pages": [],
      "total_matches": 50
    }
  }
}
```

Campos por segmento (según `_parse_results_page` en `api/scrapers/matches.py`):
`team1, team2, score1, score2, flag1, flag2, time_completed, round_info,
tournament_name, match_page, tournament_icon, page_number, match_id`.
`score1`/`score2` son **strings** (pueden ir vacíos en partidos sin jugar/forfeits).

### 4.4 `GET /v2/events?q=completed&page=1`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "title": "VCT 2026: Pacific Stage 2",
        "event_id": "2776",
        "status": "completed",
        "prize": "$250,000",
        "dates": "Jul 15—Sep 6",
        "region": "kr",
        "thumb": "https://owcdn.net/img/...",
        "url_path": "https://www.vlr.gg/event/2776/vct-2026-pacific-stage-2"
      }
    ]
  }
}
```

Sin `meta` de paginación: el fin se detecta cuando `segments` llega vacío.

### 4.5 `GET /v2/events/matches?event_id=2095`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "match_id": "595657",
        "url": "https://www.vlr.gg/595657/...",
        "date": "Sun, April 24, 2024",
        "time": "3:00 PM",
        "unix_timestamp": "2024-04-24 19:00:00",
        "status": "Completed",
        "note": "",
        "stage": "Playoffs",
        "event_series": "Grand Final",
        "team1": { "name": "Sentinels", "score": "2", "is_winner": true },
        "team2": { "name": "Cloud9", "score": "1", "is_winner": false }
      }
    ]
  }
}
```

`is_winner` da el ganador directamente. `unix_timestamp` es UTC o `""` si no hay
countdown. `team1`/`team2` son objetos (nombre, score, is_winner), **no strings**.

### 4.6 `GET /v2/match/{match_id}/details`

Parámetros: `light` (bool, salta tabs de performance/economía) e `include`
(csv de campos top-level). `include` acepta: `event, date, map_vetos, status, teams,
streams, vods, maps, head_to_head, performance, economy, economy_by_map`; `match_id`
siempre se mantiene. Un *fieldset* sin `performance`/`economy`/`economy_by_map` salta
esas peticiones *upstream* (más rápido). Ejemplo recortado (un mapa, un jugador por
equipo):

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "match_id": "773435",
        "teams": [
          { "id": "21093", "name": "Gentle Mates GC", "tag": "", "logo": "https://owcdn.net/img/...png", "score": "2", "is_winner": true },
          { "id": "14278", "name": "Shopify Rebellion Gold", "tag": "", "logo": "https://owcdn.net/img/...png", "score": "1", "is_winner": false }
        ],
        "maps": [
          {
            "map_name": "Split",
            "picked_by": "Gentle Mates GC",
            "duration": "1:06:04",
            "score": { "team1": 11, "team2": 13 },
            "score_ct": { "team1": "2", "team2": "3" },
            "score_t": { "team1": "9", "team2": "10" },
            "score_ot": { "team1": "", "team2": "" },
            "players": {
              "team1": [
                { "name": "Marti", "agent": "", "rating": "1.09", "acs": "229", "kills": "20",
                  "deaths": "17", "assists": "4", "kd_diff": "+3", "kast": "83%", "adr": "143",
                  "hs_pct": "45%", "fk": "2", "fd": "3", "fk_diff": "-1" }
              ],
              "team2": [
                { "name": "Noia", "agent": "", "rating": "1.39", "acs": "262", "kills": "22",
                  "deaths": "12", "assists": "3", "kd_diff": "+10", "kast": "88%", "adr": "177",
                  "hs_pct": "44%", "fk": "1", "fd": "2", "fk_diff": "-1" }
              ]
            },
            "rounds": [
              { "round_num": 1, "winner": "team2", "side": "t", "win_type": "elim" }
            ],
            "performance": { "kill_matrix": [], "advanced_stats": [] },
            "economy": []
          }
        ]
      }
    ]
  }
}
```

Notas de esquema:
- `data.segments[0]` es el objeto del partido (un solo elemento). **Importante**: la forma
  de `score` de mapa **varía entre versiones**: la forma *path* devuelve `score` como
  objeto `{team1: int, team2: int}` con `score_ct`/`score_t`/`score_ot` separados; la
  forma *query* legacy (`/v2/match/details?match_id=`) devuelve `score` como
  `{team1: {total, ct, t}, team2: {...}}`. **Usar la forma path** (`/v2/match/{id}/details`),
  que es la canónica y la que usa este scraper.
- `teams[].id` es el **team ID de vlr.gg** (fuente autoritativa para mapear nombre→id).
- `maps[].players.team1/team2` = stats por jugador y por mapa (K/D/A, ACS, rating, ADR, KAST…).
- `maps[].rounds` = resultado ronda a ronda; `head_to_head` = enfrentamientos previos.
- `maps[]` no jugados devuelven `rounds: []`.

### 4.7 `GET /v2/team/{team_id}/stats`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "map": "Bind",
        "games": 22,
        "win_pct": "55%",
        "wins": 12,
        "losses": 10,
        "atk_first": 11,
        "def_first": 11,
        "atk_rwin_pct": "44%",
        "atk_rw": 102,
        "atk_rl": 131,
        "def_rwin_pct": "60%",
        "def_rw": 142,
        "def_rl": 93
      }
    ]
  }
}
```

### 4.8 `GET /v2/team/{team_id}`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "id": "14278",
        "name": "Shopify Rebellion Gold",
        "tag": "SRG",
        "successor": "",
        "logo": "https://owcdn.net/img/...png",
        "country": "ca",
        "country_name": "Canada",
        "description": "Shopify Rebellion GoldSRGshopifyrebellion.gg@SRVALCanada",
        "social_links": [ { "platform": "other", "url": "https://shopifyrebellion.gg/" } ],
        "rating": { "rank": "172", "rating": "1446", "peak_rating": "", "streak": "81W60L" },
        "roster": [ { "id": "9185", "alias": "dodonut", "real_name": "...", "country": "th", "is_captain": false, "role": "", "is_staff": false } ],
        "event_placements": [ { "event": "...", "series": "...", "placement": "17th", "prize": "", "date": "2026", "url": "..." } ],
        "total_winnings": "$608,997"
      }
    ]
  }
}
```

### 4.9 `GET /v2/team/{team_id}/matches`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "match_id": "773435",
        "url": "https://www.vlr.gg/773435/...",
        "event": "GC 26: ChampionshipUBF",
        "date": "2026/10/30",
        "time": "2:00 pm",
        "team1": { "name": "Shopify Rebellion Gold", "tag": "SRG", "logo": "..." },
        "team2": { "name": "Gentle Mates GC", "tag": "M8GC", "logo": "..." },
        "score": "1:2",
        "result": "loss"
      }
    ],
    "meta": { "page": 1 }
  }
}
```

`page` va de 1 a 100. `result` es `win`/`loss` desde la perspectiva del equipo consultado.

### 4.10 `GET /v2/team/{team_id}/roster`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "active": [ { "id": "9185", "alias": "dodonut", "real_name": "...", "country": "th", "is_captain": false, "role": "", "is_staff": false } ],
        "staff": [],
        "former": [],
        "benched": [],
        "transactions": []
      }
    ]
  }
}
```

### 4.11 `GET /v2/rankings?region=na`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      { "rank": "1", "team": "Sentinels", "country": "United States", "last_played": "22h ago", "record": "7-3", "earnings": "$295,500", "logo": "//owcdn.net/img/..." }
    ]
  }
}
```

### 4.12 `GET /v2/stats?region=americas&timespan=30`

```json
{
  "status": "success",
  "data": {
    "status": 200,
    "segments": [
      {
        "player": "corey", "org": "TTR", "rating": "1.18", "average_combat_score": "235.2",
        "kill_deaths": "1.19", "kill_assists_survived_traded": "72%", "average_damage_per_round": "158.4",
        "kills_per_round": "0.81", "assists_per_round": "0.29", "first_kills_per_round": "0.19",
        "first_deaths_per_round": "0.13", "headshot_percentage": "26%", "clutch_success_percentage": "28%",
        "clutch_attempts": "9/57"
      }
    ]
  }
}
```

### 4.13 `GET /v2/search?q=tenz`

```json
{
  "status": "success",
  "data": {
    "segments": {
      "query": "tenz",
      "results": {
        "players": [ { "id": "9", "name": "TenZ", "img": "...", "description": "", "tag": "" } ],
        "teams": [ { "id": "16647", "name": "TenZ and Friends", "img": "...", "description": "", "tag": "(inactive)" } ],
        "events": []
      }
    }
  }
}
```

> **Ojo**: `/v2/search` **no** lleva `data.status`, sino `data.segments` directamente.

---

## 5. Paginación y límites (decisiones de diseño)

1. **`/v2/match?q=results` solo alcanza las páginas 1–100** (`from_page`/`to_page`/`num_pages`
   acotados a `le=100`, ventana ≤ 20 páginas por request). ≈ 100 páginas × ~50 partidos =
   ~5.000 partidos recientes. **No alcanza todo el histórico.**
2. **Para el histórico completo** se usa la vía de eventos: `GET /v2/events?q=completed&page=N`
   (moderate, 60/min) para enumerar eventos y `GET /v2/events/matches?event_id=X`
   (moderate) para los partidos de cada evento. Esta vía sí recorre años.
3. **Team IDs**: ni `q=results` ni `events/matches` traen `team_id` (solo nombres). La fuente
   autoritativa de `team_id` es `match/{id}/details → teams[].id`. Para no pagar un detail
   por partido en la Capa 1, se resuelve nombre→id con `/v2/search` (moderate) y se corrige
   en Capa 2 con los details.
4. **Fallos = 502 total**: si dentro de un request paginado falla **una** página, el endpoint
   devuelve 502 y se pierde el request entero. Por eso el scraper hace **checkpoint por
   ventana** y usa ventanas pequeñas (configurable `pages_per_request`).
5. `page` de equipos/jugadores acotado a 1–100; `request_delay` ∈ [0.5, 5.0]s,
   `max_retries` ≤ 3, `timeout` ≤ 45s en `/v2/match`.

---

## 6. Códigos de región

**`/v2/rankings`** (14): `na, eu, ap, la, la-s, la-n, oce, kr, mn, gc, br, cn, jp, col`.

**`/v2/stats`** (6): `all, americas, emea, pacific, china, intl` (con aliases obsoletos
`na/br→americas`, `eu→emea`, `ap/kr/jp/oce→pacific`, `cn→china`).

---

## 7. Supuestos y riesgos

| # | Supuesto / riesgo | Mitigación |
|---|---|---|
| R1 | `q=results` limitado a 100 páginas → no cubre todo el histórico. | Usar vía `events` + `events/matches` como fuente primaria del histórico. |
| R2 | `events` sin `meta` de total → el fin se detecta con `segments == []`. | El ETL corta al recibir una página vacía. |
| R3 | `score1/score2` (results) son strings y a veces vacíos; `unix_timestamp` puede ser `""`. | Parse defensivo (int con fallback, NaN/nulo). |
| R4 | `events` paginado acotado a 100 páginas; si hubiera >~5.000 eventos se cortaría. | Documentado; pro Valorant está muy por debajo. |
| R5 | `search` puede no resolver nombres cambiados/equipos inactivos. | `match details` corrige `team_id` en Capa 2. |
| R6 | Shape de `maps[].score` difiere entre path y query form. | Usar solo path form (`/v2/match/{id}/details`). |
| R7 | `429` se resuelve con `Retry-After`; el tier es por IP y por proceso. | Throttling por tier + backoff; reintentos respetando header. |
| R8 | Un request paginado grande puede 502 si falla una página. | Ventanas pequeñas + checkpoint por ventana. |
| R9 | `X-RateLimit-Remaining` es informativo (no usar en `429`). | Solo log; la autoridad es el throttling local + `Retry-After`. |
