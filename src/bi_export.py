"""Fase F5 · Inteligencia de Negocios — exportación del modelo estrella a CSVs.

Construye un **modelo de datos en estrella** a partir de ``interim/matches.parquet``,
``interim/rankings.parquet`` y ``processed/features.parquet`` (y, si ya existen, las
tablas de equipo ``teams`` / ``team_profiles`` / ``team_stats``) y lo exporta como
CSVs listos para abrir en **Power BI Desktop**:

- ``dim_team``    → ``data/bi/dim_team.csv``
- ``dim_event``   → ``data/bi/dim_event.csv``
- ``fact_matches`` → ``data/bi/fact_matches.csv``
- ``dim_map``     → ``data/bi/dim_map.csv`` (tabla suelta, win rate por mapa/equipo)

Reglas de diseño
----------------
1. Clave de ``dim_team`` **estable**: no hay ``team_id`` real en el dataset, así que
   se genera un ``team_id`` numérico por **nombre normalizado** (minúsculas, sin
   acentos, puntuación → espacio). La normalización es idéntica a ``features.norm_name``
   para que las claves sean coherentes con las features ``f_*``.
2. ``dim_event`` usa una clave generada ``event_id`` sobre el título del torneo.
3. ``fact_matches`` hereda la granularidad de ``features`` (una fila = un partido,
   ya con el target ``y`` y las 9 features ``f_*``); se enriquece con ``winner``,
   ``status`` y ``source`` desde ``matches`` y se añaden ``margin`` y ``winner_id``.
4. Si las tablas de equipo no existen (se están scrapeando), el export funciona
   igual con ``matches`` + ``rankings`` + ``features``; se deja constancia en el log.

Solo requiere ``pandas`` + ``numpy``.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vlr.bi")

# Directorio por defecto de los datos de entrada. Se puede sobreescribir con
# la variable de entorno VALORANT_DATA_DIR o con el flag --input-dir.
DEFAULT_INPUT_DIR = "data"

# Salida: data/bi/ dentro del repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "bi"

FEATURE_PREFIX = "f_"
PLACEHOLDER_TEAMS = {
    "", "forfeit", "bye", "tbd", "n/a", "null", "none", "nan",
    "tba", "to be determined", "to be announced", "unknown",
}

# Utilidades

def norm_name(s: Any) -> str:
    """Normaliza un nombre de equipo para usarlo como clave (idéntico a
    ``features.norm_name``): minúsculas, sin acentos, puntuación → espacio y
    espacios colapsados/recortados. ``"LEVIATÁN"`` → ``"leviatan"``."""
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    text = unicodedata.normalize("NFKD", str(s))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_key(s: Any) -> str:
    """Clave única estable por equipo.

    Usa ``norm_name``; si el nombre es íntegramente no-Latino (CJK, cirílico,
    árabe…) y colapsa a cadena vacía, conserva el original normalizado en Unicode
    (NFKC + minúsculas + espacios colapsados) para no fusionar equipos distintos
    (p. ej. ``"雨のち晴れ"`` y ``"АОГИРИ"``). Los equipos del ``fact_matches``
    siempre tienen clave no vacía, así que esto solo afecta a nombres exóticos
    de ``rankings`` que no participan en ningún partido del fact.
    """
    key = norm_name(s)
    if key:
        return key
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    raw = unicodedata.normalize("NFKC", str(s)).strip().lower()
    return re.sub(r"\s+", " ", raw).strip()


def _parse_money(s: Any) -> float:
    """``"$1,191,150"`` → ``1191150.0`` (NaN si no parsea)."""
    if s is None:
        return np.nan
    t = str(s).strip().replace("$", "").replace(",", "").replace(" ", "")
    if t in ("", "-", "—", "nan", "None"):
        return np.nan
    try:
        return float(t)
    except ValueError:
        return np.nan


def _parse_record(s: Any) -> tuple[float, float]:
    """``"62–35"`` → ``(62, 35)`` (victorias, derrotas). NaN si no parsea."""
    if s is None:
        return np.nan, np.nan
    parts = [p for p in re.split(r"[–—-]", str(s).strip()) if p.strip() != ""]
    if len(parts) >= 2:
        return (
            pd.to_numeric(parts[0], errors="coerce"),
            pd.to_numeric(parts[1], errors="coerce"),
        )
    return np.nan, np.nan


def _parse_streak(s: Any) -> tuple[float, str]:
    """``"8W"`` → ``(8, "W")``; ``"1L"`` → ``(1, "L")``."""
    if s is None:
        return np.nan, ""
    m = re.match(r"^\s*(\d+)\s*([WL])\s*$", str(s))
    if m:
        return float(m.group(1)), m.group(2)
    return np.nan, ""


def _parse_pct(s: Any) -> float:
    """``"67%"`` → ``0.67`` (NaN si no parsea: ``"N/A"``, vacío, ``"-"``…)."""
    if s is None:
        return np.nan
    t = str(s).strip().replace("%", "").strip()
    if t in ("", "-", "—", "N/A", "n/a", "nan", "None"):
        return np.nan
    try:
        return float(t) / 100.0
    except ValueError:
        return np.nan


def _parse_rating_dict(rating: Any) -> dict:
    """Normaliza el campo ``rating`` de ``team_profiles`` a un dict.

    En parquet ya llega como ``dict``; se tolera que venga como string JSON-like.
    """
    if isinstance(rating, dict):
        return rating
    if isinstance(rating, str):
        try:
            parsed = ast.literal_eval(rating)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def _profile_rating(rating: Any) -> float:
    """``{'rating': '1825'}`` → ``1825.0``; ``'N/A'``/vacío → ``NaN``."""
    return pd.to_numeric(
        str(_parse_rating_dict(rating).get("rating", "")), errors="coerce"
    )


def _profile_rank(rating: Any) -> float:
    """``{'rank': '9'}`` → ``9.0``; vacío → ``NaN``."""
    return pd.to_numeric(
        str(_parse_rating_dict(rating).get("rank", "")), errors="coerce"
    )


# Roles de staff (el flag ``is_staff`` del scraper está siempre en False, así que
# se distingue por ``role``). Un jugador "activo no-staff" es aquel cuyo rol no es
# de cuerpo técnico ni contiene "inactive".
_STAFF_ROLE_KEYWORDS = ("coach", "manager", "analyst", "staff")


def _roster_size(roster: Any) -> float:
    """Nº de jugadores activos no-staff en el ``roster`` (lista de dicts)."""
    if roster is None:
        return np.nan
    if isinstance(roster, str):
        try:
            roster = ast.literal_eval(roster)
        except (ValueError, SyntaxError):
            return np.nan
    if isinstance(roster, (list, tuple, np.ndarray)):
        items = list(roster)
    else:
        return np.nan
    count = 0
    for p in items:
        if not isinstance(p, dict):
            continue
        role = str(p.get("role") or "").lower()
        if "inactive" in role:  # suplentes inactivos / plantillas con baja
            continue
        if any(k in role for k in _STAFF_ROLE_KEYWORDS):  # cuerpo técnico
            continue
        count += 1
    return float(count)


def _read_parquet(path: Path) -> pd.DataFrame | None:
    """Lee un parquet si existe; si no, devuelve ``None`` (sin romper el export)."""
    if path is None or not Path(path).exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensivo
        logger.warning("No se pudo leer %s: %s", path, exc)
        return None

# Carga de entradas

def _resolve_input_dir(input_dir: str | Path | None) -> Path:
    if input_dir:
        return Path(input_dir)
    return Path(os.environ.get("VALORANT_DATA_DIR", DEFAULT_INPUT_DIR))


def load_inputs(input_dir: str | Path | None = None) -> dict[str, pd.DataFrame | None]:
    """Carga ``matches``, ``rankings`` y ``features`` (y tablas de equipo si existen)."""
    base = _resolve_input_dir(input_dir)
    matches = _read_parquet(base / "interim" / "matches.parquet")
    rankings = _read_parquet(base / "interim" / "rankings.parquet")

    features = _read_parquet(base / "processed" / "features.parquet")
    if features is None:
        # fallback: features.csv (misma carpeta)
        csv = base / "processed" / "features.csv"
        if csv.exists():
            features = pd.read_csv(csv, dtype={"match_id": str})
            logger.info("features leído desde features.csv (fallback)")

    # Tablas de equipo (opcionales, aún en scraping).
    teams = _read_parquet(base / "interim" / "teams.parquet")
    team_profiles = _read_parquet(base / "interim" / "team_profiles.parquet")
    team_stats = _read_parquet(base / "interim" / "team_stats.parquet")

    if matches is None:
        raise FileNotFoundError(
            f"No se encontró matches.parquet en {base / 'interim'}. "
            "Indica el directorio con VALORANT_DATA_DIR o --input-dir."
        )
    if rankings is None:
        raise FileNotFoundError(
            f"No se encontró rankings.parquet en {base / 'interim'}."
        )
    if features is None:
        raise FileNotFoundError(
            f"No se encontró features.parquet/features.csv en {base / 'processed'}."
        )

    return {
        "matches": matches,
        "rankings": rankings,
        "features": features,
        "teams": teams,
        "team_profiles": team_profiles,
        "team_stats": team_stats,
    }

# Dimensiones

def build_dim_team(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    features: pd.DataFrame,
    team_profiles: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Construye ``dim_team`` y devuelve el mapa ``nombre_normalizado -> team_id``.

    El universo de equipos es la unión de los equipos que aparecen en ``features``
    (los que realmente referenciará ``fact_matches``) y los de ``rankings``.
    Para equipos presentes en varias regiones en ``rankings`` se conserva la entrada
    de **mejor rank** (y, en empate, mayor rating).
    """
    # 1) Ranking deduplicado por clave única estable.
    rk = rankings.copy()
    rk["team_norm"] = rk["team"].map(norm_key)
    rk["_rank_num"] = pd.to_numeric(rk["rank"], errors="coerce")
    rk["_rating_num"] = pd.to_numeric(rk["rating"], errors="coerce")
    rk = (
        rk.sort_values(["_rank_num", "_rating_num"], ascending=[True, False], na_position="last")
        .drop_duplicates(subset="team_norm", keep="first")
        .reset_index(drop=True)
    )
    rk_name_map = dict(zip(rk["team_norm"], rk["team"]))

    # 2) Equipos presentes en features (los que referenciará el fact).
    ft = pd.unique(pd.concat([features["team1"], features["team2"]]))
    fact_teams = pd.DataFrame({"team": ft})
    fact_teams["team_norm"] = fact_teams["team"].map(norm_key)
    # Nombre canónico para equipos que no están en rankings: el más frecuente.
    fact_canon = fact_teams.groupby("team_norm")["team"].agg(
        lambda s: s.value_counts().index[0]
    )

    # 3) Universo final ordenado + team_id estable (orden alfabético del key).
    keys = sorted(set(fact_canon.index) | set(rk_name_map.keys()))
    dim = pd.DataFrame({"team_name_norm": keys})
    dim["team_id"] = np.arange(1, len(dim) + 1, dtype=np.int64)
    # team_name: nombre oficial de rankings si existe; si no, el canónico de features.
    dim["team_name"] = dim["team_name_norm"].map(rk_name_map).fillna(
        dim["team_name_norm"].map(fact_canon)
    )

    # 4) Atributos desde rankings (deduplicado).
    attr_cols = ["team_norm", "region", "country", "rank", "rating", "record",
                 "recent_record", "streak", "earnings"]
    attr = rk[attr_cols].rename(columns={"team_norm": "team_name_norm"})
    dim = dim.merge(attr, on="team_name_norm", how="left", validate="one_to_one")

    # 5) Enriquecimiento opcional desde team_profiles (si existe y tiene columnas esperadas).
    if team_profiles is not None:
        dim = _enrich_from_profiles(dim, team_profiles)

    # 5b) Garantiza la existencia de las columnas de perfil (aunque no haya profiles).
    for col in ["logo", "country_code", "country_name", "profile_rating",
                "profile_rank", "total_winnings_usd", "roster_size"]:
        if col not in dim.columns:
            dim[col] = np.nan

    # 6) Tipado / parsing de atributos numéricos.
    dim["rank"] = pd.to_numeric(dim["rank"], errors="coerce").astype("Int64")
    dim["rating"] = pd.to_numeric(dim["rating"], errors="coerce").astype("Int64")
    dim["earnings_usd"] = dim["earnings"].map(_parse_money)
    dim = dim.drop(columns=["earnings"])
    dim["profile_rating"] = pd.to_numeric(dim["profile_rating"], errors="coerce").astype("Int64")
    dim["profile_rank"] = pd.to_numeric(dim["profile_rank"], errors="coerce").astype("Int64")
    dim["total_winnings_usd"] = pd.to_numeric(dim["total_winnings_usd"], errors="coerce")
    dim["roster_size"] = pd.to_numeric(dim["roster_size"], errors="coerce").astype("Int64")

    rec = dim["record"].map(_parse_record)
    dim["wins"] = rec.map(lambda t: t[0]).astype("Int64")
    dim["losses"] = rec.map(lambda t: t[1]).astype("Int64")

    st = dim["streak"].map(_parse_streak)
    dim["streak_value"] = st.map(lambda t: t[0]).astype("Int64")
    dim["streak_type"] = st.map(lambda t: t[1])

    dim["en_ranking"] = dim["team_name_norm"].isin(set(rk_name_map.keys())).astype(np.int8)

    # Orden y columnas finales.
    final_cols = [
        "team_id", "team_name", "team_name_norm", "logo",
        "region", "country", "country_code", "country_name",
        "rank", "rating", "profile_rating", "profile_rank",
        "record", "wins", "losses", "earnings_usd", "total_winnings_usd",
        "roster_size", "streak", "streak_value", "streak_type", "en_ranking",
    ]
    dim = dim[final_cols].sort_values("team_id").reset_index(drop=True)

    # Mapa clave -> team_id para construir el fact.
    id_map = dict(zip(dim["team_name_norm"], dim["team_id"]))
    return dim, id_map


def _enrich_from_profiles(dim: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """Enriquece ``dim_team`` con atributos de ``team_profiles`` (Capa 2).

    Añade por ``team_name_norm`` (clave normalizada del ``name`` del perfil):
    ``logo``, ``country_code`` (ISO de ``country``), ``country_name``,
    ``profile_rating``, ``profile_rank``, ``total_winnings_usd`` y
    ``roster_size``. Solo los ~1 548 equipos con perfil rankeado tendrán estos
    valores; el resto quedan vacíos. Además rellena ``country`` y
    ``earnings_usd`` (de rankings) con los valores de perfil cuando faltan.
    """
    if profiles is None or "name" not in profiles.columns:
        return dim
    prof = profiles.copy()
    prof["team_name_norm"] = prof["name"].map(norm_key)
    prof = prof.drop_duplicates(subset="team_name_norm", keep="first")

    # Columnas nuevas derivadas del perfil.
    prof["logo"] = prof["logo"] if "logo" in prof.columns else None
    prof["country_code"] = prof["country"] if "country" in prof.columns else None
    prof["country_name"] = prof["country_name"] if "country_name" in prof.columns else None
    prof["profile_rating"] = prof["rating"].map(_profile_rating)
    prof["profile_rank"] = prof["rating"].map(_profile_rank)
    prof["total_winnings_usd"] = prof["total_winnings"].map(_parse_money) \
        if "total_winnings" in prof.columns else np.nan
    prof["roster_size"] = prof["roster"].map(_roster_size) if "roster" in prof.columns else np.nan

    keep = [
        "team_name_norm", "logo", "country_code", "country_name",
        "profile_rating", "profile_rank", "total_winnings_usd", "roster_size",
    ]
    dim = dim.merge(prof[keep], on="team_name_norm", how="left")

    # Relleno de huecos (no sobrescribe valores existentes de rankings).
    if "country" in dim.columns and "country_name" in dim.columns:
        dim["country"] = dim["country"].fillna(dim["country_name"])
    if "earnings_usd" in dim.columns and "total_winnings_usd" in dim.columns:
        dim["earnings_usd"] = dim["earnings_usd"].fillna(dim["total_winnings_usd"])
    return dim


def build_dim_event(matches: pd.DataFrame) -> pd.DataFrame:
    """Construye ``dim_event`` a partir de los eventos únicos de ``matches``."""
    ev = matches[["event"]].copy()
    ev["event_name"] = ev["event"].astype(str).str.strip()
    # rango de fechas del evento (parseando el string de fecha original).
    ev["_fecha"] = pd.to_datetime(matches["date"], errors="coerce")

    agg = (
        ev.groupby("event_name", dropna=False)["_fecha"]
        .agg(num_partidos="count", fecha_min="min", fecha_max="max")
        .reset_index()
    )
    agg = agg[agg["event_name"].str.len() > 0]
    agg = agg.sort_values("event_name").reset_index(drop=True)
    agg["event_id"] = np.arange(1, len(agg) + 1, dtype=np.int64)
    agg["num_partidos"] = agg["num_partidos"].astype(np.int64)
    agg["fecha_min"] = agg["fecha_min"].dt.strftime("%Y-%m-%d")
    agg["fecha_max"] = agg["fecha_max"].dt.strftime("%Y-%m-%d")

    return agg[["event_id", "event_name", "num_partidos", "fecha_min", "fecha_max"]]

# dim_map (tabla suelta, win rate por mapa/equipo)

def build_dim_map(
    team_stats: pd.DataFrame,
    teams: pd.DataFrame,
) -> pd.DataFrame:
    """Construye ``dim_map`` desde ``team_stats`` + ``teams``.

    Es una **tabla suelta** (sin FK a ``fact_matches``): el fact no guarda el
    mapa en el que se jugó cada partido, así que no se puede relacionar. Su
    clave natural es el par ``(team_id, map)``. ``team_id`` es el id vlr.gg del
    equipo (string, coherente con ``teams.parquet``), y ``team_name`` se
    resuelve desde ``teams`` (id → name); si no se encuentra, se usa el propio
    ``team_id``.
    """
    ts = team_stats.copy()
    # Resolución de nombre vía teams.parquet (team_id -> name).
    id_to_name = dict(zip(teams["team_id"].astype(str), teams["name"].astype(str)))
    ts["team_name"] = ts["team_id"].astype(str).map(id_to_name)
    ts["team_name"] = ts["team_name"].fillna(ts["team_id"].astype(str))

    ts["win_pct"] = ts["win_pct"].map(_parse_pct)
    ts["atk_rwin_pct"] = ts["atk_rwin_pct"].map(_parse_pct)
    ts["def_rwin_pct"] = ts["def_rwin_pct"].map(_parse_pct)

    cols = [
        "team_id", "team_name", "map", "games", "wins", "losses",
        "win_pct", "atk_rwin_pct", "def_rwin_pct",
    ]
    return ts[cols].sort_values(["team_id", "map"], kind="stable").reset_index(drop=True)

# Fact

def build_fact_matches(
    matches: pd.DataFrame,
    features: pd.DataFrame,
    team_id_map: dict[str, int],
    event_id_map: dict[str, int],
    probabilities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construye ``fact_matches`` (una fila = un partido) desde ``features`` + ``matches``.

    Si se pasa ``probabilities`` (salida del walk-forward), se unen por
    ``match_id`` las columnas ``prob_team1``/``odds_team1``/``odds_team2``/``tier``.
    """
    fact = features.copy()

    # Claves foráneas.
    fact["team1_id"] = fact["team1"].map(norm_key).map(team_id_map).astype("Int64")
    fact["team2_id"] = fact["team2"].map(norm_key).map(team_id_map).astype("Int64")
    fact["event_id"] = fact["event"].astype(str).str.strip().map(event_id_map).astype("Int64")

    # winner desde matches (para trazabilidad; features ya excluyó empates).
    mwin = matches[["match_id", "winner", "status", "source", "time"]].copy()
    fact = fact.merge(mwin, on="match_id", how="left", validate="many_to_one")

    # winner_id: FK a dim_team del equipo ganador.
    fact["winner_id"] = np.where(
        fact["y"].astype(int) == 1, fact["team1_id"].astype(float), fact["team2_id"].astype(float)
    )
    fact["winner_id"] = fact["winner_id"].astype("Int64")

    # Medidas.
    fact["margin"] = fact["score1"] - fact["score2"]
    fact["fecha"] = pd.to_datetime(fact["date_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%d")

    # Probabilidades out-of-sample (walk-forward) por partido, si existen.
    prob_cols = ["prob_team1", "odds_team1", "odds_team2", "tier"]
    if probabilities is not None and not probabilities.empty:
        fact = fact.merge(
            probabilities[["match_id"] + prob_cols],
            on="match_id", how="left", validate="one_to_one",
        )
    else:
        for c in prob_cols:
            fact[c] = np.nan

    fact = fact.sort_values(["date_ts", "match_id"], kind="stable").reset_index(drop=True)

    # Orden de columnas: claves/medidas primero, features al final.
    feat_cols = [c for c in fact.columns if c.startswith(FEATURE_PREFIX)]
    base_cols = [
        "match_id", "team1_id", "team2_id", "winner_id", "event_id",
        "team1", "team2", "winner", "stage", "date", "fecha", "date_ts",
        "score1", "score2", "margin", "y",
        "prob_team1", "odds_team1", "odds_team2", "tier",
        "status", "source",
    ]
    fact = fact[base_cols + feat_cols]
    return fact

# Orquestación

def _load_probabilities(path: Path | None = None) -> pd.DataFrame | None:
    """Carga ``probabilities.parquet``/``.csv`` (salida del walk-forward) si existe."""
    if path is None:
        path = REPO_ROOT / "data" / "processed" / "probabilities.parquet"
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv, dtype={"match_id": str})
    return None


def export_star_schema(input_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Construye y devuelve el modelo estrella completo."""
    inputs = load_inputs(input_dir)
    matches = inputs["matches"]
    rankings = inputs["rankings"]
    features = inputs["features"]

    logger.info("matches=%d · rankings=%d · features=%d",
                len(matches), len(rankings), len(features))

    dim_team, team_id_map = build_dim_team(matches, rankings, features, inputs["team_profiles"])
    dim_event = build_dim_event(matches)

    event_id_map = dict(zip(dim_event["event_name"], dim_event["event_id"]))
    # Cualquier evento presente en features pero no en matches (no debería ocurrir)
    # se fuerza a existir en dim_event para no romper la FK.
    missing_events = set(features["event"].astype(str).str.strip()) - set(event_id_map)
    if missing_events:
        extra = pd.DataFrame({
            "event_name": sorted(missing_events),
            "num_partidos": 0,
            "fecha_min": None,
            "fecha_max": None,
        })
        extra["event_id"] = np.arange(len(dim_event) + 1, len(dim_event) + 1 + len(extra), dtype=np.int64)
        dim_event = pd.concat([dim_event, extra], ignore_index=True)
        event_id_map.update(dict(zip(dim_event["event_name"], dim_event["event_id"])))

    # Probabilidades OOS (walk-forward); opcionales.
    probabilities = _load_probabilities()
    if probabilities is not None:
        logger.info("probabilities=%d filas (walk-forward OOS).", len(probabilities))

    fact = build_fact_matches(matches, features, team_id_map, event_id_map, probabilities)

    # dim_map (tabla suelta) si existen team_stats + teams.
    dim_map = None
    if inputs["team_stats"] is not None and inputs["teams"] is not None:
        dim_map = build_dim_map(inputs["team_stats"], inputs["teams"])
        logger.info("dim_map=%d filas (win rate por mapa/equipo).", len(dim_map))

    tables: dict[str, pd.DataFrame] = {
        "dim_team": dim_team,
        "dim_event": dim_event,
        "fact_matches": fact,
    }
    if dim_map is not None:
        tables["dim_map"] = dim_map
    return tables


def write_csvs(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Escribe los CSVs (UTF-8 con BOM) y devuelve las rutas generadas."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, df in tables.items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written.append(path)
    return written


def validate_schema(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Chequeos de integridad del modelo estrella."""
    dim_team = tables["dim_team"]
    dim_event = tables["dim_event"]
    fact = tables["fact_matches"]

    team_ids = set(dim_team["team_id"])
    event_ids = set(dim_event["event_id"])

    checks: dict[str, Any] = {
        "dim_team_filas": len(dim_team),
        "dim_event_filas": len(dim_event),
        "fact_filas": len(fact),
        "team_id_unicos": dim_team["team_id"].is_unique,
        "event_id_unicos": dim_event["event_id"].is_unique,
        "match_id_unicos": fact["match_id"].is_unique,
        "fk_team1_ok": fact["team1_id"].isin(team_ids).all(),
        "fk_team2_ok": fact["team2_id"].isin(team_ids).all(),
        "fk_winner_ok": fact["winner_id"].isin(team_ids).all(),
        "fk_event_ok": fact["event_id"].isin(event_ids).all(),
        "sin_nulos_fk": fact[["team1_id", "team2_id", "winner_id", "event_id"]].notna().all().all(),
    }
    if "prob_team1" in fact.columns:
        checks["prob_cob_%"] = round(100 * fact["prob_team1"].notna().mean(), 2)
    if "dim_map" in tables:
        dm = tables["dim_map"]
        checks["dim_map_filas"] = len(dm)
        checks["dim_map_sin_nulos_id"] = dm["team_id"].notna().all() and dm["map"].notna().all()
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export del modelo estrella a CSVs para Power BI")
    parser.add_argument("--input-dir", default=None, help="Directorio de datos de entrada")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio de salida CSVs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tables = export_star_schema(args.input_dir)
    paths = write_csvs(tables, args.output_dir)

    checks = validate_schema(tables)

    print("\n=== Modelo estrella exportado (F5 · BI) ===")
    for name, df in tables.items():
        print(f"  {name:<14} {len(df):>8,} filas  ·  {len(df.columns):>2} columnas")
    print("\nCSVs generados:")
    for p in paths:
        print(f"  - {p}")
    print("\nChequeos de integridad:")
    for k, v in checks.items():
        print(f"  {k:<16} {v}")

    if not all(checks.values()):
        logger.warning("Algún chequeo de integridad falló (ver arriba).")
        return 1
    print("\nOK: modelo estrella listo para cargar en Power BI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
