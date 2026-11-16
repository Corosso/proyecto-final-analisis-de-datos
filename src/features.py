"""Feature engineering (Fase F3) — dataset listo para modelar, sin fuga de datos.

Genera un dataset ``processed/features.parquet`` (+ ``features.csv`` para Power BI)
a partir de ``interim/matches.parquet`` y ``interim/rankings.parquet`` (y, si ya
existen, las tablas de equipo ``teams``, ``team_stats`` y ``team_profiles``).

Reglas duras de diseño
----------------------
1. **Target binario** ``y``: 1 si gana ``team1``, 0 si gana ``team2``. Se descartan
   ``winner`` en {draw, None} y partidos con equipos placeholder (forfeit/bye/tbd...).
2. **Sin fuga (data leakage)**: toda feature *histórica* (forma, win rate,
   head-to-head, margen, experiencia, descanso, racha) se calcula en orden
   cronológico usando **solo** partidos anteriores al partido en cuestión
   (acumulados/rolling por equipo). El partido propio y los posteriores nunca se usan.
3. **Features diferenciales** ``team1 − team2`` con prefijo ``f_``, todas numéricas.
4. **Split temporal**: NO se hace aquí. Se conservan ``date`` (string), ``date_ts``
   (epoch) y ``match_id``/``team1``/``team2`` para que la fase de modelado haga el
   split y la trazabilidad.

Limitación conocida (documentada): ``rankings.rating`` y ``rankings.rank`` son una
*foto actual* (no hay ranking histórico en el dataset), por lo que ``f_rating_diff``
y ``f_rank_diff`` tienen un sesgo de "información futura" para partidos antiguos.
El resto de features históricas sí son estrictamente libres de fuga.
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import get_data_dir

logger = logging.getLogger("vlr.features")

# Constantes

N_FORM = 5       # ventana corta de forma reciente (nº de partidos previos)
N_WINRATE = 20   # ventana media de win rate histórico reciente
N_MAPSCORE = 5   # ventana corta para el margen medio de mapas/puntos

# Equipos que NO son equipos reales (partidos no jugados / forfeits / pendientes).
PLACEHOLDER_TEAMS = {
    "", "forfeit", "bye", "tbd", "n/a", "null", "none", "nan",
    "tba", "to be determined", "to be announced", "unknown",
}

MIN_YEAR = 2020  # Valorant empezó en 2020; fechas anteriores son artefactos/placeholder.

# Utilidades

def parse_date(s: Any) -> pd.Timestamp | None:
    """Convierte una cadena de fecha en ``pd.Timestamp`` (o ``None``).

    Soportes:
    - ``"Thu, October 22, 2026"`` (formato humano de vlr.gg)
    - ISO ``"2026-11-06 18:45:00"`` / ``"2026-11-06"``
    - ``None`` / vacío / ``NaN`` -> ``None``

    Las fechas relativas tipo ``"2d 0h ago"`` (endpoint ``q=results`` reciente)
    devuelven ``None`` y se descartan en el pipeline por no ser absolutas.
    """
    if s is None:
        return None
    if isinstance(s, float) and np.isnan(s):
        return None
    text = str(s).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(ts) else ts


def norm_name(s: Any) -> str:
    """Normaliza un nombre de equipo para usarlo como *clave* de agrupación.

    Minúsculas, sin acentos, puntuación -> espacio, espacios colapsados y
    recortados. Así ``"LEVIATÁN"``, ``"Leviatán"`` y ``"leviatan"`` colapsan a la
    misma clave (evita partir el historial de un equipo por diferencias de casing).
    """
    if s is None:
        return ""
    if isinstance(s, float) and np.isnan(s):
        return ""
    text = unicodedata.normalize("NFKD", str(s))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_placeholder(name: Any) -> bool:
    """True si ``name`` es un placeholder de equipo (forfeit/bye/tbd/...)."""
    return norm_name(name) in PLACEHOLDER_TEAMS

# Target

def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Añade el target binario ``y`` y descarta partidos no predecibles.

    - ``y = 1`` si ``winner == "team1"``, ``y = 0`` si ``winner == "team2"``.
    - Descarta ``winner`` en {draw, None/NaN}, equipos *placeholder*
      (forfeit, bye, tbd, ...) y ``team1 == team2``.

    Devuelve un DataFrame copiado (no muta el original).
    """
    out = df.copy()

    # target binario (NaN si no es team1/team2)
    out["y"] = np.where(out["winner"] == "team1", 1,
                        np.where(out["winner"] == "team2", 0, np.nan))
    out = out[out["y"].notna()].copy()
    out["y"] = out["y"].astype(int)

    # equipos inválidos
    bad = out["team1"].map(_is_placeholder) | out["team2"].map(_is_placeholder)
    out = out[~bad]

    # mismo equipo en ambos lados (error / duplicado)
    out = out[out["team1"].map(norm_name) != out["team2"].map(norm_name)]

    return out.reset_index(drop=True)

# Timestamps

def _resolve_timestamps(df: pd.DataFrame) -> pd.Series:
    """Fecha (+ hora) -> ``datetime64[ns]``; ``NaT`` donde no se pudo resolver.

    Orden de preferencia: ``date`` (string humano), luego ``unix_timestamp``
    (que en la práctica es un datetime string, no epoch), y suma la hora de
    ``time`` ("H:MM AM/PM") para desempatar partidos del mismo día.
    """
    d = pd.to_datetime(df["date"], errors="coerce")

    # fallback: unix_timestamp (realmente un datetime string en vlr.gg)
    if "unix_timestamp" in df.columns:
        u = pd.to_datetime(df["unix_timestamp"], errors="coerce")
        d = d.where(d.notna(), u)

    # hora del día
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], errors="coerce", format="%I:%M %p")
        if t.notna().any():
            hours = t.dt.hour.fillna(0).astype("int64")
            minutes = t.dt.minute.fillna(0).astype("int64")
            d = d + pd.to_timedelta(hours * 3600 + minutes * 60, unit="s")

    # Normalizar a nanosegundos (pandas puede devolver [us] según el input de
    # parquet); así ``.astype('int64')`` es consistente al calcular epoch.
    return d.astype("datetime64[ns]")

# Features históricas por equipo (sin fuga)

def _grolling(perf: pd.DataFrame, col: str, window: int) -> pd.Series:
    """Media móvil de ``col`` por equipo (``min_periods=1``), alineada al índice."""
    return (
        perf.groupby("team_key", sort=False)[col]
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _streak(team_keys: np.ndarray, won_shift: np.ndarray) -> np.ndarray:
    """Racha actual (firmada) ANTES de cada partido, por equipo.

    +N = N victorias seguidas; -N = N derrotas seguidas; 0 = sin historial.
    Se calcula sobre el resultado *previo* (``won_shift``), nunca el actual.
    """
    state: dict[str, tuple[int | None, int]] = {}  # equipo -> (último_resultado, racha)
    out = np.zeros(len(won_shift), dtype="int64")
    for i in range(len(won_shift)):
        key = team_keys[i]
        w = won_shift[i]
        if pd.isna(w):
            state[key] = (None, 0)
            out[i] = 0
            continue
        w = int(w)
        last, cur = state.get(key, (None, 0))
        if last is not None and w == last:
            cur += 1 if w == 1 else -1
        else:
            cur = 1 if w == 1 else -1
        state[key] = (w, cur)
        out[i] = cur
    return out


def _per_team_features(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (t1f, t2f): features históricas por equipo, libres de fuga.

    Se apilan los partidos en formato largo (una fila por equipo-partido), se
    ordenan cronológicamente y por cada equipo se calculan con ``shift(1)`` +
    rolling (nunca el partido actual):

    - ``form``    : % victorias en los últimos 5 previos (0.5 si sin historial).
    - ``winrate`` : % victorias en los últimos 20 previos (0.5 si sin historial).
    - ``mapscore``: margen medio (score a favor − en contra) últimos 5 previos.
    - ``experience``: nº de partidos previos del equipo.
    - ``rest``    : días desde el partido previo del equipo.
    - ``streak``  : racha firmada (+victorias / −derrotas) antes del partido.
    """
    score1 = pd.to_numeric(s["score1"], errors="coerce")
    score2 = pd.to_numeric(s["score2"], errors="coerce")
    mid_num = pd.to_numeric(s["match_id"], errors="coerce")

    t1 = pd.DataFrame({
        "match_id": s["match_id"].values,
        "team_key": s["team1_key"].values,
        "won": (s["y"].values == 1).astype("int8"),
        "margin": (score1 - score2).values,
        "dt": s["dt"].values,
        "mid_num": mid_num.values,
        "role": "team1",
    })
    t2 = pd.DataFrame({
        "match_id": s["match_id"].values,
        "team_key": s["team2_key"].values,
        "won": (s["y"].values == 0).astype("int8"),
        "margin": (score2 - score1).values,
        "dt": s["dt"].values,
        "mid_num": mid_num.values,
        "role": "team2",
    })

    perf = pd.concat([t1, t2], ignore_index=True)
    perf["dt"] = pd.to_datetime(perf["dt"])
    perf = perf.sort_values(["dt", "mid_num"], kind="mergesort").reset_index(drop=True)

    g = perf.groupby("team_key", sort=False)
    perf["won_shift"] = g["won"].shift(1)          # resultado del partido PREVIO
    perf["margin_shift"] = g["margin"].shift(1)
    perf["prev_dt"] = g["dt"].shift(1)
    perf["rest"] = (perf["dt"] - perf["prev_dt"]).dt.days

    perf["form"] = _grolling(perf, "won_shift", N_FORM)
    perf["winrate"] = _grolling(perf, "won_shift", N_WINRATE)
    perf["mapscore"] = _grolling(perf, "margin_shift", N_MAPSCORE)
    perf["experience"] = g.cumcount()               # nº de previos (0-based)
    perf["streak"] = _streak(perf["team_key"].values, perf["won_shift"].values)

    cols = ["match_id", "form", "winrate", "mapscore", "experience", "rest", "streak"]
    t1f = perf[perf["role"] == "team1"][cols].copy()
    t2f = perf[perf["role"] == "team2"][cols].copy()

    # imputación neutral antes de diferenciar (sin historial -> empate)
    for c in ("form", "winrate"):
        t1f[c] = t1f[c].fillna(0.5)
        t2f[c] = t2f[c].fillna(0.5)
    for c in ("mapscore", "rest"):
        t1f[c] = t1f[c].fillna(0.0)
        t2f[c] = t2f[c].fillna(0.0)

    t1f = t1f.rename(columns={c: f"t1_{c}" for c in cols if c != "match_id"})
    t2f = t2f.rename(columns={c: f"t2_{c}" for c in cols if c != "match_id"})
    return t1f, t2f


def _h2h_winrate(s: pd.DataFrame) -> np.ndarray:
    """% de victorias de team1 contra team2 en enfrentamientos PREVIOS (0.5 default).

    Recorre los partidos en orden cronológico manteniendo, por par (ordenado) de
    equipos, cuántas veces ganó cada uno. Para el partido actual solo se lee el
    estado acumulado hasta el partido anterior.
    """
    state: dict[tuple[str, str], dict] = {}
    out = np.full(len(s), 0.5, dtype="float64")
    t1s = s["team1_key"].values
    t2s = s["team2_key"].values
    ys = s["y"].values

    for i in range(len(s)):
        a, b = t1s[i], t2s[i]
        key = (a, b) if a < b else (b, a)
        st = state.get(key)
        if st is not None and st["n"] > 0:
            wins_t1 = st["wa"] if a == key[0] else st["wb"]
            out[i] = wins_t1 / st["n"]

        if st is None:
            st = {"wa": 0, "wb": 0, "n": 0}
            state[key] = st
        if ys[i] == 1:                       # ganó team1
            if a == key[0]:
                st["wa"] += 1
            else:
                st["wb"] += 1
        else:                                # ganó team2
            if a == key[0]:
                st["wb"] += 1
            else:
                st["wa"] += 1
        st["n"] += 1
    return out

# Rankings (foto estática) y enriquecimientos opcionales

def _ranking_maps(rankings_df: pd.DataFrame | None) -> tuple[dict, dict, int]:
    """(rating_map, rank_map, worst_rank) a partir de ``rankings``.

    - ``rating``: máximo por equipo (rating global, dedup entre regiones).
    - ``rank``: mínimo por equipo (mejor rank regional).
    - ``worst_rank``: valor para equipos SIN ranking (peor que todos).
    """
    if rankings_df is None or rankings_df.empty:
        return {}, {}, 100_000
    rk = rankings_df.copy()
    rk["_k"] = rk["team"].map(norm_name)
    rk = rk[rk["_k"] != ""]
    if "rating" in rk.columns:
        rk["_rating"] = pd.to_numeric(rk["rating"], errors="coerce")
    else:
        rk["_rating"] = np.nan
    if "rank" in rk.columns:
        rk["_rank"] = pd.to_numeric(rk["rank"], errors="coerce")
    else:
        rk["_rank"] = np.nan

    agg = rk.groupby("_k", as_index=False).agg(
        _rating=("_rating", "max"), _rank=("_rank", "min"))
    rating_map = dict(zip(agg["_k"], agg["_rating"]))
    rank_map = dict(zip(agg["_k"], agg["_rank"]))
    worst_rank = int(agg["_rank"].max()) + 1 if agg["_rank"].notna().any() else 100_000
    return rating_map, rank_map, worst_rank


def _name_to_id_map(teams_df: pd.DataFrame | None,
                    team_profiles_df: pd.DataFrame | None) -> dict[str, str]:
    """``{name_key: team_id}`` desde ``teams.parquet`` y/o ``team_profiles``."""
    mapping: dict[str, str] = {}
    if teams_df is not None and not teams_df.empty and {"name", "team_id"} <= set(teams_df.columns):
        for _, row in teams_df.iterrows():
            k = norm_name(row["name"])
            if k:
                mapping[k] = str(row["team_id"])
    if team_profiles_df is not None and not team_profiles_df.empty \
            and {"id", "name"} <= set(team_profiles_df.columns):
        for _, row in team_profiles_df.iterrows():
            k = norm_name(row["name"])
            if k:
                mapping.setdefault(k, str(row["id"]))
    return mapping


def _enrich_team_stats(out: pd.DataFrame, team_stats_df: pd.DataFrame,
                       teams_df: pd.DataFrame | None,
                       team_profiles_df: pd.DataFrame | None) -> pd.DataFrame:
    """``f_map_winrate_diff``: win_pct medio en mapas (team1 − team2), si existen datos."""
    ts = team_stats_df.copy()
    id_col = "team_id" if "team_id" in ts.columns else ("id" if "id" in ts.columns else None)
    if id_col is None or "win_pct" not in ts.columns:
        return out
    # win_pct viene como string "67%": quitar '%' y pasar a fracción 0–1.
    ts["win_pct"] = pd.to_numeric(
        ts["win_pct"].astype(str).str.replace("%", "", regex=False),
        errors="coerce") / 100.0
    win_by_id = ts.groupby(id_col)["win_pct"].mean().to_dict()

    id_map = _name_to_id_map(teams_df, team_profiles_df)
    win_by_name = {nk: win_by_id[tid] for nk, tid in id_map.items() if tid in win_by_id}
    if not win_by_name:
        return out

    w1 = out["team1_key"].map(win_by_name).fillna(0.5)
    w2 = out["team2_key"].map(win_by_name).fillna(0.5)
    out["f_map_winrate_diff"] = (w1 - w2).astype("float64")
    return out


def _enrich_profiles(out: pd.DataFrame, team_profiles_df: pd.DataFrame) -> pd.DataFrame:
    """``f_profile_rating_diff`` y ``f_winnings_diff`` desde ``team_profiles``."""
    tp = team_profiles_df.copy()
    if "name" not in tp.columns:
        return out
    tp["_k"] = tp["name"].map(norm_name)

    def _get_rating(v: Any) -> Any:
        return v.get("rating") if isinstance(v, dict) else v

    if "rating" in tp.columns:
        tp["_pr"] = tp["rating"].map(_get_rating)
        pr = tp.groupby("_k")["_pr"].apply(
            lambda s: pd.to_numeric(s, errors="coerce").max()).to_dict()
        if pr:
            p1 = out["team1_key"].map(pr).fillna(0.0)
            p2 = out["team2_key"].map(pr).fillna(0.0)
            out["f_profile_rating_diff"] = (p1 - p2).astype("float64")

    if "total_winnings" in tp.columns:
        # total_winnings viene como "$1,583,636": quitar '$' y ','.
        tp["_tw"] = pd.to_numeric(
            tp["total_winnings"].astype(str).str.replace(r"[$,]", "", regex=True),
            errors="coerce")
        tw = tp.groupby("_k")["_tw"].max().to_dict()
        if tw:
            w1 = out["team1_key"].map(tw).fillna(0.0)
            w2 = out["team2_key"].map(tw).fillna(0.0)
            # log1p: la variable es muy sesgada (un solo equipo con $136M).
            out["f_winnings_diff"] = (np.log1p(w1) - np.log1p(w2)).astype("float64")
    return out

# Orquestación de features diferenciales

def add_differential_features(
    df: pd.DataFrame,
    rankings_df: pd.DataFrame | None = None,
    team_stats_df: pd.DataFrame | None = None,
    team_profiles_df: pd.DataFrame | None = None,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Añade las features diferenciales ``f_*`` (team1 − team2) sin fuga.

    ``df`` debe contener ``match_id, team1, team2, score1, score2, y`` y ``date``
    (y ``time``/``unix_timestamp`` si existen). Devuelve el DataFrame ordenado
    cronológicamente con las features ``f_*`` añadidas (y columnas internas de
    trabajo, que ``build_features`` descarta en el export final).
    """
    out = df.copy()

    # claves normalizadas para agrupar historial por equipo
    out["team1_key"] = out["team1"].map(norm_name)
    out["team2_key"] = out["team2"].map(norm_name)

    # timestamp (fecha + hora)
    if "dt" not in out.columns:
        out["dt"] = _resolve_timestamps(out)

    # descartar fechas inválidas o artefactos previos a 2020
    out = out[out["dt"].notna()].copy()
    out = out[out["dt"].dt.year >= MIN_YEAR].copy()

    # dedup defensivo + orden cronológico (match_id numérico como desempate)
    out = out.drop_duplicates(subset=["match_id"]).copy()
    out["_mid_num"] = pd.to_numeric(out["match_id"], errors="coerce")
    out = out.sort_values(["dt", "_mid_num"], kind="mergesort",
                          na_position="last").reset_index(drop=True)

    # --- features históricas por equipo + head-to-head ---
    t1f, t2f = _per_team_features(out)
    out = out.merge(t1f, on="match_id", how="left")
    out = out.merge(t2f, on="match_id", how="left")
    out["f_h2h_winrate"] = _h2h_winrate(out)

    # --- rating / rank (foto estática de rankings) ---
    rating_map, rank_map, worst_rank = _ranking_maps(rankings_df)
    r1 = out["team1_key"].map(rating_map).fillna(0.0)
    r2 = out["team2_key"].map(rating_map).fillna(0.0)
    out["f_rating_diff"] = (r1 - r2).astype("float64")

    # rank: menor = mejor -> el equipo sin ranking recibe el PEOR rank (worst_rank),
    # para que la señal sea coherente (negativo favorece a team1).
    k1 = out["team1_key"].map(rank_map).fillna(worst_rank)
    k2 = out["team2_key"].map(rank_map).fillna(worst_rank)
    out["f_rank_diff"] = (k1 - k2).astype("float64")

    # --- diferenciales históricos (team1 − team2) ---
    out["f_form_diff"] = (out["t1_form"] - out["t2_form"]).astype("float64")
    out["f_winrate_diff"] = (out["t1_winrate"] - out["t2_winrate"]).astype("float64")
    out["f_mapscore_diff"] = (out["t1_mapscore"] - out["t2_mapscore"]).astype("float64")
    out["f_experience_diff"] = (out["t1_experience"] - out["t2_experience"]).astype("float64")
    out["f_rest_diff"] = (out["t1_rest"] - out["t2_rest"]).astype("float64")
    out["f_streak_diff"] = (out["t1_streak"] - out["t2_streak"]).astype("float64")

    # --- enriquecimientos opcionales (tablas de equipo, si ya existen) ---
    if team_stats_df is not None and not team_stats_df.empty:
        out = _enrich_team_stats(out, team_stats_df, teams_df, team_profiles_df)
    if team_profiles_df is not None and not team_profiles_df.empty:
        out = _enrich_profiles(out, team_profiles_df)

    return out

# Entrada de datos + export

def _read_table(prefix: Path, required: bool = False) -> pd.DataFrame | None:
    """Lee ``<prefix>.parquet`` (o ``.csv``) si existe; opcionalmente exige que exista."""
    prefix = Path(prefix)
    for p in (prefix.with_suffix(".parquet"), prefix.with_suffix(".csv")):
        if p.exists():
            try:
                return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("No se pudo leer %s: %s", p, exc)
                if required:
                    raise
                return None
    if required:
        raise FileNotFoundError(f"No existe {prefix}.parquet ni {prefix}.csv")
    return None


def build_features(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Orquesta todo el feature engineering y guarda el dataset final.

    Lee ``interim/*.parquet``, genera ``processed/features.parquet`` y
    ``processed/features.csv``, y devuelve el DataFrame.
    """
    data_dir = Path(data_dir or get_data_dir())
    interim = data_dir / "interim"
    processed = data_dir / "processed"

    matches = _read_table(interim / "matches", required=True)
    rankings_df = _read_table(interim / "rankings")
    teams_df = _read_table(interim / "teams")
    team_stats_df = _read_table(interim / "team_stats")
    team_profiles_df = _read_table(interim / "team_profiles")

    if rankings_df is None:
        logger.warning("rankings.parquet no encontrado: f_rating_diff/f_rank_diff = 0.")

    df = build_target(matches)
    n_before = len(df)
    df = add_differential_features(df, rankings_df, team_stats_df,
                                   team_profiles_df, teams_df)
    logger.info("Partidos tras target+filtrado de fechas: %d -> %d", n_before, len(df))

    # fecha como epoch (segundos) para el split temporal de modelado
    df["date_ts"] = (df["dt"].astype("datetime64[ns]").astype("int64") // 10**9).astype("int64")

    # columnas finales: metadatos + target + features
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    meta_cols = ["match_id", "team1", "team2", "date", "date_ts",
                 "event", "stage", "score1", "score2"]
    final_cols = [c for c in meta_cols if c in df.columns] + ["y"] + feature_cols

    out = df[final_cols].copy()
    out["y"] = out["y"].astype("int8")
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    processed.mkdir(parents=True, exist_ok=True)
    out.to_parquet(processed / "features.parquet", index=False)
    out.to_csv(processed / "features.csv", index=False)
    logger.info("features guardadas en %s (parquet + csv)", processed)
    return out


def main() -> int:
    """CLI: `python features.py --data-dir /ruta/a/datos`."""
    ap = argparse.ArgumentParser(
        description="Feature engineering (F3) — genera processed/features.{parquet,csv}",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(get_data_dir()),
                    help="raíz de datos (contiene interim/ y processed/)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    df = build_features(args.data_dir)
    print(f"\nfeatures listas: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("target:", df["y"].value_counts().to_dict())
    print("features:", [c for c in df.columns if c.startswith("f_")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
