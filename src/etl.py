"""Orquestación ETL por capas (Fase F1) sobre vlrggapi.

Flujo general::

    Capa 1 (rápida, nivel equipo)  ->  datos de partidos + equipos + rankings
    Capa 2 (lenta, por partido)    ->  detalles (stats por jugador/mapa)

Las funciones de colección son *idempotentes*: consultan el ``Checkpoint`` del
cliente y solo descargan lo que falta, guardando cada respuesta cruda en
``data/raw/`` y las tablas consolidadas en ``data/interim/`` (Parquet con
fallback a CSV).

Esquema canónico de la tabla de partidos (``matches``)::

    match_id, team1, team2, score1, score2, winner, event, stage, date, source

``winner`` es "team1" | "team2" | "draw" | None. El target binario de fases
posteriores (¿ganó el equipo A?) se deriva de aquí.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from scraper import (
    MAX_PAGE,
    MAX_PAGE_WINDOW,
    RANKING_REGIONS,
    STATS_REGIONS,
    Checkpoint,
    VLRClient,
    ensure_dir,
    read_json,
    write_json_atomic,
)

try:  # tqdm es opcional pero muy útil en Colab
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback sin barra
    def tqdm(it: Iterable, **_: Any) -> Iterable:
        return it

logger = logging.getLogger("vlr.etl")


# ---------------------------------------------------------------------------
# Utilidades de tablas y archivos
# ---------------------------------------------------------------------------

def save_table(df: pd.DataFrame, prefix: str | Path, index: bool = False) -> Path:
    """Guarda un DataFrame en Parquet; si falla, en CSV. Devuelve la ruta."""
    prefix = Path(prefix)
    ensure_dir(prefix.parent)
    try:
        out = prefix.with_suffix(".parquet")
        df.to_parquet(out, index=index)
        return out
    except Exception as exc:  # noqa: BLE001 - pyarrow ausente, etc.
        logger.warning("No se pudo escribir Parquet (%s); usando CSV.", exc)
        out = prefix.with_suffix(".csv")
        df.to_csv(out, index=index)
        return out


def load_table(path: str | Path) -> pd.DataFrame:
    """Carga Parquet o CSV según lo que exista."""
    path = Path(path)
    parquet = path.with_suffix(".parquet")
    csv = path.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No existe {parquet} ni {csv}")


def _to_int(value: Any) -> int | None:
    """Convierte un valor (string de score, etc.) a int o None si no aplica."""
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def _winner(score1: int | None, score2: int | None) -> str | None:
    if score1 is None or score2 is None:
        return None
    if score1 > score2:
        return "team1"
    if score2 > score1:
        return "team2"
    return "draw"


def _segments(data: dict) -> list:
    """Extrae la lista ``segments`` de un ``data`` desenvuelto."""
    if not isinstance(data, dict):
        return []
    segs = data.get("segments")
    return segs if isinstance(segs, list) else []


# ---------------------------------------------------------------------------
# Capa 1a — Resultados recientes (GET /v2/match?q=results)
# ---------------------------------------------------------------------------

def collect_match_results(client: VLRClient, max_pages: int = MAX_PAGE,
                          pages_per_request: int = 5) -> list[dict]:
    """Descarga las primeras ``max_pages`` de resultados (ventanas con checkpoint).

    ``max_pages`` se acota a 100 (límite de la API). ``pages_per_request`` se
    acota a 20 (ventana máxima). Guarda crudos en ``data/raw/results/`` y
    devuelve la lista plana de segmentos (partidos).
    """
    max_pages = min(max_pages, MAX_PAGE)
    pages_per_request = min(max(pages_per_request, 1), MAX_PAGE_WINDOW)
    cp: Checkpoint = client.checkpoint
    done_ranges = {tuple(r) for r in cp.section("results_ranges", [])}
    raw_dir = ensure_dir(client.data_dir / "raw" / "results")

    all_rows: list[dict] = []
    start = 1
    pbar = tqdm(desc="q=results", total=max_pages, unit="pag")
    while start <= max_pages:
        end = min(start + pages_per_request - 1, max_pages)
        rng = (start, end)
        if rng not in done_ranges:
            data = client.match_results(from_page=start, to_page=end,
                                        request_delay=0.5)
            write_json_atomic(raw_dir / f"pages_{start:04d}_{end:04d}.json", data)
            done_ranges.add(rng)
            cp.set("results_ranges", sorted(done_ranges))
            cp.save()
        # Recargar segmentos del archivo crudo (no re-llamar a la API).
        cached = read_json(raw_dir / f"pages_{start:04d}_{end:04d}.json", {})
        rows = _segments(cached)
        all_rows.extend(rows)
        pbar.update(end - start + 1)
        start = end + 1
    pbar.close()

    # Reconstruir la lista completa en orden de página (por si se retomó).
    seen: dict[str, dict] = {}
    for r in all_rows:
        seen.setdefault(str(r.get("match_id")), r)
    return list(seen.values())


# ---------------------------------------------------------------------------
# Capa 1b — Histórico completo vía eventos (GET /v2/events + /v2/events/matches)
# ---------------------------------------------------------------------------

def collect_events(client: VLRClient, max_pages: int = MAX_PAGE) -> list[dict]:
    """Enumera eventos completados (moderate tier) hasta agotar la paginación.

    Devuelve la lista de eventos (``title``, ``event_id``, ...). Guarda crudos
    en ``data/raw/events/``.
    """
    cp: Checkpoint = client.checkpoint
    done = set(cp.section("events_pages", []))
    exhausted = bool(cp.get("events_exhausted", False))
    raw_dir = ensure_dir(client.data_dir / "raw" / "events")

    events: list[dict] = []
    page = 1
    while page <= max_pages:
        if exhausted:
            # Ya agotamos la paginación en una ejecución previa: solo
            # reconstruimos los eventos ya descargados desde la caché.
            if page not in done:
                break
            events.extend(_segments(read_json(raw_dir / f"events_{page:04d}.json", {})))
        elif page not in done:
            data = client.events(q="completed", page=page)
            segs = _segments(data)
            write_json_atomic(raw_dir / f"events_{page:04d}.json", data)
            done.add(page)
            cp.set("events_pages", sorted(done))
            if not segs:
                exhausted = True
                cp.set("events_exhausted", True)
            cp.save()
            events.extend(segs)
        else:
            events.extend(_segments(read_json(raw_dir / f"events_{page:04d}.json", {})))
        page += 1
    return events


def collect_event_matches(client: VLRClient, event_ids: Iterable[str | int],
                          event_title_map: dict[str, str] | None = None) -> list[dict]:
    """Descarga los partidos de cada evento (moderate tier).

    ``event_title_map``: opcional, ``{event_id: title}`` para rellenar el campo
    ``event`` de cada partido. Guarda crudos en ``data/raw/event_matches/``.
    """
    event_title_map = event_title_map or {}
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "event_matches")

    rows: list[dict] = []
    save_every = 25
    for i, eid in enumerate(tqdm(list(event_ids), desc="event_matches")):
        eid = str(eid)
        if cp.is_done("event_matches", eid):
            cached = read_json(raw_dir / f"event_{eid}.json", {})
            segs = _segments(cached)
        else:
            data = client.event_matches(eid)
            write_json_atomic(raw_dir / f"event_{eid}.json", data)
            cp.mark_done("event_matches", eid)
            if (i + 1) % save_every == 0:
                cp.save()
            segs = _segments(data)
        for s in segs:
            s = dict(s)
            s["event"] = event_title_map.get(eid, s.get("event", ""))
            rows.append(s)
    cp.save()
    return rows


# ---------------------------------------------------------------------------
# Capa 1c — Rankings (GET /v2/rankings)
# ---------------------------------------------------------------------------

def collect_rankings(client: VLRClient,
                     regions: Iterable[str] = RANKING_REGIONS) -> list[dict]:
    """Descarga rankings regionales (moderate tier)."""
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "rankings")
    rows: list[dict] = []
    for region in tqdm(list(regions), desc="rankings"):
        if cp.is_done("rankings", region):
            data = read_json(raw_dir / f"{region}.json", {})
        else:
            data = client.rankings(region)
            write_json_atomic(raw_dir / f"{region}.json", data)
            cp.mark_done("rankings", region)
            cp.save()
        for s in _segments(data):
            s = dict(s)
            s["region"] = region
            rows.append(s)
    return rows


# ---------------------------------------------------------------------------
# Resolución nombre -> team_id (GET /v2/search, moderate tier)
# ---------------------------------------------------------------------------

def resolve_team_ids(client: VLRClient, team_names: Iterable[str]) -> dict[str, str]:
    """Resuelve nombres de equipo a ids de vlr.gg vía ``/v2/search``.

    Devuelve ``{nombre: team_id}``. Elige el primer resultado de ``teams`` cuyo
    ``name`` coincida (case-insensitive). Guarda el mapa en el checkpoint y
    crudos en ``data/raw/search/``. Es una *mejor aproximación*: los ids
    autoritativos se obtienen en Capa 2 con ``match details``.
    """
    cp: Checkpoint = client.checkpoint
    resolved: dict[str, str] = dict(cp.section("search", {}))
    raw_dir = ensure_dir(client.data_dir / "raw" / "search")

    # Nombres que NO son equipos reales (partidos no jugados / forfeits).
    placeholders = {"", "forfeit", "bye", "tbd", "n/a", "null", "none"}

    def _safe_filename(name: str) -> str:
        """Nombre de archivo seguro para un nombre de equipo arbitrario."""
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        return clean or f"name_{hashlib.md5(name.encode()).hexdigest()[:12]}"

    names = sorted({str(n) for n in team_names if n})
    for name in tqdm(names, desc="search name->id"):
        key = name.lower()
        if key in resolved or key in placeholders:
            continue
        try:
            data = client.search(name)
        except Exception as exc:  # noqa: BLE001 - seguimos con el resto
            logger.warning("search(%s) falló: %s", name, exc)
            continue
        write_json_atomic(raw_dir / f"{_safe_filename(key)}.json", data)
        segs = data.get("segments", {}) if isinstance(data, dict) else {}
        teams = (segs.get("results", {}) or {}).get("teams", []) or []
        match = next((t for t in teams if str(t.get("name", "")).lower() == key), None)
        if match and match.get("id"):
            resolved[key] = str(match["id"])
        cp.set("search", resolved)
        cp.save()
    return resolved


# ---------------------------------------------------------------------------
# Capa 1d — Datos de equipo (GET /v2/team/{id}/*)
# ---------------------------------------------------------------------------

def collect_team_profiles(client: VLRClient, team_ids: Iterable[str | int]) -> list[dict]:
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "teams")
    rows: list[dict] = []
    for tid in tqdm(list(team_ids), desc="team profiles"):
        tid = str(tid)
        if cp.is_done("team_profiles", tid):
            data = read_json(raw_dir / f"profile_{tid}.json", {})
        else:
            data = client.team(tid)
            write_json_atomic(raw_dir / f"profile_{tid}.json", data)
            cp.mark_done("team_profiles", tid)
            cp.save()
        segs = _segments(data)
        if segs:
            rows.append(segs[0])
    return rows


def collect_team_stats(client: VLRClient, team_ids: Iterable[str | int]) -> list[dict]:
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "teams")
    rows: list[dict] = []
    for tid in tqdm(list(team_ids), desc="team stats"):
        tid = str(tid)
        if cp.is_done("team_stats", tid):
            data = read_json(raw_dir / f"stats_{tid}.json", {})
        else:
            data = client.team_stats(tid)
            write_json_atomic(raw_dir / f"stats_{tid}.json", data)
            cp.mark_done("team_stats", tid)
            cp.save()
        for s in _segments(data):
            s = dict(s)
            s["team_id"] = tid
            rows.append(s)
    return rows


def collect_team_matches(client: VLRClient, team_ids: Iterable[str | int],
                         max_pages_per_team: int = 3) -> list[dict]:
    """Historial de partidos por equipo (paginado). Por defecto 3 páginas/equipo."""
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "teams")
    done_by_team: dict[str, list[int]] = cp.section("team_matches", {})
    rows: list[dict] = []

    for tid in tqdm(list(team_ids), desc="team matches"):
        tid = str(tid)
        done_pages = set(done_by_team.setdefault(tid, []))
        for page in range(1, max_pages_per_team + 1):
            fname = raw_dir / f"matches_{tid}_p{page:02d}.json"
            if page in done_pages:
                data = read_json(fname, {})
            else:
                data = client.team_matches(tid, page=page)
                write_json_atomic(fname, data)
                done_pages.add(page)
                done_by_team[tid] = sorted(done_pages)
                cp.save()
            segs = _segments(data)
            if not segs:
                break  # fin de historial del equipo
            for s in segs:
                s = dict(s)
                s["team_id"] = tid
                rows.append(s)
    return rows


def collect_team_rosters(client: VLRClient, team_ids: Iterable[str | int]) -> list[dict]:
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "teams")
    rows: list[dict] = []
    for tid in tqdm(list(team_ids), desc="team rosters"):
        tid = str(tid)
        if cp.is_done("team_rosters", tid):
            data = read_json(raw_dir / f"roster_{tid}.json", {})
        else:
            data = client.team_roster(tid)
            write_json_atomic(raw_dir / f"roster_{tid}.json", data)
            cp.mark_done("team_rosters", tid)
            cp.save()
        segs = _segments(data)
        if segs:
            seg = dict(segs[0])
            seg["team_id"] = tid
            rows.append(seg)
    return rows


# ---------------------------------------------------------------------------
# Normalización -> tabla canónica de partidos
# ---------------------------------------------------------------------------

def build_matches_table(results_rows: list[dict],
                        event_rows: list[dict]) -> pd.DataFrame:
    """Combina ``q=results`` y ``events/matches`` en una única tabla deduplicada.

    Prioriza la fuente ``event_matches`` (tiene ``is_winner`` y ``stage``), y
    rellena huecos con ``results``. Deduplica por ``match_id``.
    """
    records: dict[str, dict] = {}

    # 1) event_matches (fuente autoritativa para winner/stage)
    for r in event_rows:
        mid = str(r.get("match_id", "")).strip()
        if not mid:
            continue
        t1 = (r.get("team1") or {})
        t2 = (r.get("team2") or {})
        score1 = _to_int(t1.get("score"))
        score2 = _to_int(t2.get("score"))
        w1 = bool(t1.get("is_winner"))
        w2 = bool(t2.get("is_winner"))
        winner = "team1" if w1 else ("team2" if w2 else _winner(score1, score2))
        records[mid] = {
            "match_id": mid,
            "team1": t1.get("name"),
            "team2": t2.get("name"),
            "score1": score1,
            "score2": score2,
            "winner": winner,
            "event": r.get("event"),
            "stage": r.get("stage") or r.get("event_series"),
            "date": r.get("date"),
            "time": r.get("time"),
            "unix_timestamp": r.get("unix_timestamp"),
            "status": r.get("status"),
            "source": "event_matches",
        }

    # 2) results (rellena partidos no vistos por eventos)
    for r in results_rows:
        mid = str(r.get("match_id", "")).strip()
        if not mid or mid in records:
            continue
        score1 = _to_int(r.get("score1"))
        score2 = _to_int(r.get("score2"))
        records[mid] = {
            "match_id": mid,
            "team1": r.get("team1"),
            "team2": r.get("team2"),
            "score1": score1,
            "score2": score2,
            "winner": _winner(score1, score2),
            "event": r.get("tournament_name"),
            "stage": r.get("round_info"),
            "date": r.get("time_completed"),
            "time": None,
            "unix_timestamp": None,
            "status": "completed",
            "source": "results",
        }

    df = pd.DataFrame.from_records(list(records.values()))
    if df.empty:
        df = pd.DataFrame(columns=[
            "match_id", "team1", "team2", "score1", "score2", "winner",
            "event", "stage", "date", "time", "unix_timestamp", "status", "source",
        ])
    return df


# ---------------------------------------------------------------------------
# Capa 2 — Detalles por partido (GET /v2/match/{id}/details)
# ---------------------------------------------------------------------------

# include por defecto para la Capa 2: todo menos tabs de performance/economía
# (más rápidos) salvo que se pidan explícitamente.
DEFAULT_DETAIL_INCLUDE = ["event", "date", "map_vetos", "status", "teams",
                          "streams", "vods", "maps", "head_to_head"]


def collect_match_details(client: VLRClient, match_ids: Iterable[str | int],
                          include: list[str] | None = None,
                          light: bool = False) -> list[dict]:
    """Descarga detalles completos por partido (expensive tier, 20/min).

    Devuelve la lista de objetos ``data.segments[0]`` (o ``data`` si no trae
    ``segments``). Guarda crudos en ``data/raw/details/``.
    """
    include = include if include is not None else DEFAULT_DETAIL_INCLUDE
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "details")
    out: list[dict] = []
    for mid in tqdm(list(match_ids), desc="match details"):
        mid = str(mid)
        fname = raw_dir / f"{mid}.json"
        if cp.is_done("match_details", mid):
            data = read_json(fname, {})
        else:
            data = client.match_details(mid, include=include, light=light)
            write_json_atomic(fname, data)
            cp.mark_done("match_details", mid)
            cp.save()
        segs = _segments(data)
        out.append(segs[0] if segs else data)
    return out


def flatten_match_details(details: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplana detalles en tres tablas: matches, maps y player_stats.

    - ``matches``: una fila por partido (equipos, ids, score, winner, vetoes).
    - ``maps``: una fila por mapa (score, picked_by, duration, winner del mapa).
    - ``player_stats``: una fila por jugador-mapa (K/D/A, ACS, rating, ADR...).
    """
    match_rows: list[dict] = []
    map_rows: list[dict] = []
    player_rows: list[dict] = []

    for d in details:
        if not isinstance(d, dict):
            continue
        mid = str(d.get("match_id", ""))
        teams = d.get("teams") or []
        team_map = {}
        for i, t in enumerate(teams, start=1):
            team_map[f"team{i}"] = {
                "id": t.get("id"),
                "name": t.get("name"),
                "score": _to_int(t.get("score")),
                "is_winner": bool(t.get("is_winner")),
                "logo": t.get("logo"),
            }
        t1 = team_map.get("team1", {})
        t2 = team_map.get("team2", {})
        winner = "team1" if t1.get("is_winner") else ("team2" if t2.get("is_winner") else None)
        match_rows.append({
            "match_id": mid,
            "team1_id": t1.get("id"),
            "team1_name": t1.get("name"),
            "team1_score": t1.get("score"),
            "team2_id": t2.get("id"),
            "team2_name": t2.get("name"),
            "team2_score": t2.get("score"),
            "winner": winner,
            "event": (d.get("event") or {}).get("name") if isinstance(d.get("event"), dict) else d.get("event"),
            "date": d.get("date"),
            "status": d.get("status"),
            "map_vetos": d.get("map_vetos"),
            "head_to_head": len(d.get("head_to_head") or []),
        })

        for mi, m in enumerate(d.get("maps") or [], start=1):
            m = m or {}
            score = m.get("score") or {}
            if isinstance(score, dict):
                s1 = score.get("team1")
                s2 = score.get("team2")
                # forma query legacy: {team1:{total,...}, team2:{total,...}}
                if isinstance(s1, dict):
                    s1 = s1.get("total")
                if isinstance(s2, dict):
                    s2 = s2.get("total")
            else:
                s1 = s2 = None
            s1i, s2i = _to_int(s1), _to_int(s2)
            map_rows.append({
                "match_id": mid,
                "map_index": mi,
                "map_name": m.get("map_name"),
                "picked_by": m.get("picked_by"),
                "duration": m.get("duration"),
                "score_team1": s1i,
                "score_team2": s2i,
                "map_winner": _winner(s1i, s2i),
            })

            players = m.get("players") or {}
            for side in ("team1", "team2"):
                for p in players.get(side) or []:
                    p = p or {}
                    row = {
                        "match_id": mid,
                        "map_index": mi,
                        "map_name": m.get("map_name"),
                        "side": side,
                        "player": p.get("name"),
                        "agent": p.get("agent"),
                        "rating": p.get("rating"),
                        "acs": _to_int(p.get("acs")),
                        "kills": _to_int(p.get("kills")),
                        "deaths": _to_int(p.get("deaths")),
                        "assists": _to_int(p.get("assists")),
                        "kd_diff": p.get("kd_diff"),
                        "kast": p.get("kast"),
                        "adr": p.get("adr"),
                        "hs_pct": p.get("hs_pct"),
                        "fk": _to_int(p.get("fk")),
                        "fd": _to_int(p.get("fd")),
                        "fk_diff": p.get("fk_diff"),
                    }
                    player_rows.append(row)

    matches_df = pd.DataFrame(match_rows)
    maps_df = pd.DataFrame(map_rows)
    players_df = pd.DataFrame(player_rows)
    return matches_df, maps_df, players_df


def details_team_map(details: list[dict]) -> pd.DataFrame:
    """Extrae el mapa autoritativo team_id ↔ team_name/logo desde los detalles."""
    seen: dict[str, dict] = {}
    for d in details:
        if not isinstance(d, dict):
            continue
        for t in d.get("teams") or []:
            tid = str(t.get("id", "")).strip()
            if tid and tid not in seen:
                seen[tid] = {
                    "team_id": tid,
                    "name": t.get("name"),
                    "tag": t.get("tag"),
                    "logo": t.get("logo"),
                }
    return pd.DataFrame(list(seen.values()))


# ---------------------------------------------------------------------------
# Orquestadores
# ---------------------------------------------------------------------------

def run_layer1(client: VLRClient, *,
               max_results_pages: int = MAX_PAGE,
               pages_per_request: int = 5,
               max_events_pages: int = MAX_PAGE,
               max_team_match_pages: int = 3,
               rankings: bool = True,
               team_data: bool = True,
               ranked_only: bool = True,
               progress: Callable[[str], None] = lambda msg: logger.info(msg)) -> dict[str, pd.DataFrame]:
    """Capa 1: partidos (results + eventos) + rankings + datos de equipo.

    Devuelve un dict de DataFrames y los guarda en ``data/interim/``.
    """
    interim = ensure_dir(client.data_dir / "interim")
    tables: dict[str, pd.DataFrame] = {}

    # --- Partidos recientes (results) ------------------------------------
    progress("Descargando resultados recientes (q=results)...")
    results_rows = collect_match_results(client, max_pages=max_results_pages,
                                         pages_per_request=pages_per_request)
    progress(f"  -> {len(results_rows)} partidos recientes")

    # --- Eventos y sus partidos (histórico completo) ---------------------
    progress("Enumerando eventos completados...")
    events = collect_events(client, max_pages=max_events_pages)
    progress(f"  -> {len(events)} eventos")
    event_titles = {str(e.get("event_id")): e.get("title", "") for e in events}
    event_ids = list(event_titles.keys())
    progress("Descargando partidos por evento...")
    event_rows = collect_event_matches(client, event_ids, event_titles)
    progress(f"  -> {len(event_rows)} partidos (histórico)")

    # --- Tabla canónica de partidos --------------------------------------
    matches = build_matches_table(results_rows, event_rows)
    tables["matches"] = matches
    save_table(matches, interim / "matches")
    progress(f"Tabla matches: {len(matches)} filas -> data/interim/matches")

    # --- Rankings --------------------------------------------------------
    if rankings:
        progress("Descargando rankings regionales...")
        rankings_rows = collect_rankings(client)
        tables["rankings"] = pd.DataFrame(rankings_rows)
        save_table(tables["rankings"], interim / "rankings")

    # --- Datos de equipo ---------------------------------------------------
    if team_data and not matches.empty:
        team_names = pd.unique(pd.concat([matches["team1"], matches["team2"]])
                               .dropna().astype(str))
        if ranked_only and "rankings" in tables and not tables["rankings"].empty:
            ranked = set(tables["rankings"]["team"].astype(str).str.lower())
            team_names = [n for n in team_names if str(n).lower() in ranked]
        progress(f"Resolviendo {len(team_names)} nombres de equipo a ids (search)...")
        name_to_id = resolve_team_ids(client, team_names)
        progress(f"  -> {len(name_to_id)} equipos resueltos")
        team_ids = sorted({v for v in name_to_id.values() if v})

        if team_ids:
            progress("Descargando perfiles/stats/matches/rosters de equipos...")
            profiles = collect_team_profiles(client, team_ids)
            tables["team_profiles"] = pd.DataFrame(profiles)
            save_table(tables["team_profiles"], interim / "team_profiles")

            stats = collect_team_stats(client, team_ids)
            tables["team_stats"] = pd.DataFrame(stats)
            save_table(tables["team_stats"], interim / "team_stats")

            tmatches = collect_team_matches(client, team_ids,
                                            max_pages_per_team=max_team_match_pages)
            tables["team_matches"] = pd.DataFrame(tmatches)
            save_table(tables["team_matches"], interim / "team_matches")

            rosters = collect_team_rosters(client, team_ids)
            tables["team_rosters"] = pd.DataFrame(rosters)
            save_table(tables["team_rosters"], interim / "team_rosters")

            # tabla de mapeo nombre->id para fases posteriores
            teams_df = pd.DataFrame(
                [{"name": n, "team_id": i} for n, i in name_to_id.items()])
            tables["teams"] = teams_df
            save_table(teams_df, interim / "teams")

    return tables


def run_layer2(client: VLRClient, match_ids: Iterable[str | int] | None = None, *,
               include: list[str] | None = None,
               light: bool = False,
               limit: int | None = None,
               progress: Callable[[str], None] = lambda msg: logger.info(msg)) -> dict[str, pd.DataFrame]:
    """Capa 2: detalles por partido + aplanado a matches/maps/player_stats.

    Si ``match_ids`` es None, se leen de ``data/interim/matches``. ``limit``
    acota el número de partidos (útil para pruebas).
    """
    interim = ensure_dir(client.data_dir / "interim")

    if match_ids is None:
        try:
            matches = load_table(interim / "matches")
            match_ids = matches["match_id"].dropna().astype(str).tolist()
        except FileNotFoundError:
            raise RuntimeError("No se encontró data/interim/matches; corre la Capa 1 primero.")

    match_ids = [str(m) for m in match_ids]
    if limit is not None:
        match_ids = match_ids[:limit]

    progress(f"Descargando detalles de {len(match_ids)} partidos...")
    details = collect_match_details(client, match_ids, include=include, light=light)

    progress("Aplanando detalles...")
    m_df, maps_df, players_df = flatten_match_details(details)
    team_map = details_team_map(details)

    tables = {
        "match_details": m_df,
        "match_maps": maps_df,
        "match_player_stats": players_df,
        "team_id_map": team_map,
    }
    for name, df in tables.items():
        save_table(df, interim / name)
    progress(f"Detalles: {len(m_df)} partidos, {len(maps_df)} mapas, "
             f"{len(players_df)} filas jugador-mapa.")
    return tables


def collect_stats_aggregate(client: VLRClient,
                            regions: Iterable[str] | None = None,
                            timespan: str = "all") -> list[dict]:
    """(Opcional, F3) Agregados por jugador vía ``GET /v2/stats`` (expensive)."""
    if regions is None:
        regions = [r for r in STATS_REGIONS if r != "all"]
    cp: Checkpoint = client.checkpoint
    raw_dir = ensure_dir(client.data_dir / "raw" / "stats")
    rows: list[dict] = []
    for region in tqdm(list(regions), desc="stats"):
        key = f"{region}_{timespan}"
        if cp.is_done("stats_agg", key):
            data = read_json(raw_dir / f"{key}.json", {})
        else:
            data = client.stats(region, timespan=timespan)
            write_json_atomic(raw_dir / f"{key}.json", data)
            cp.mark_done("stats_agg", key)
            cp.save()
        for s in _segments(data):
            s = dict(s)
            s["region"] = region
            rows.append(s)
    return rows
