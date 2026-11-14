"""Utilidades de EDA (Fase F2) — carga de tablas y helpers de visualización.

Módulo reutilizable que importa el notebook ``notebooks/03_eda.ipynb``.
Proporciona:

- Carga robusta de las tablas de ``interim/`` y ``processed/`` (Parquet o CSV),
  detectando con ``Path.exists()`` si las tablas de equipo opcionales
  (``teams``, ``team_stats``, ``team_profiles``, ``team_rosters``) ya existen.
- Transformaciones de limpieza/agregación para el EDA (parseo de fechas,
  win rate por equipo, matriz head-to-head).
- Helpers de matplotlib/seaborn para unificar estilo, tamaño y títulos.

Solo depende de pandas, numpy, matplotlib y seaborn (sin scikit-learn), por lo
que funciona igual en local (``uv run``) y en Google Colab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Constantes / estilo
# ---------------------------------------------------------------------------

# Paleta coherente con la identidad del proyecto (verde terminal / retro).
COLOR_TEAM1 = "#2ecc71"   # verde: gana team1
COLOR_TEAM2 = "#e74c3c"   # rojo:  gana team2
COLOR_NEUTRAL = "#5f6f7f"

# Equipos que NO son equipos reales (forfeits / pendientes). Se usan solo para
# el análisis descriptivo de calidad, no para reconstruir features.
PLACEHOLDER_TEAMS = {
    "", "forfeit", "bye", "tbd", "n/a", "null", "none", "nan",
    "tba", "to be determined", "to be announced", "unknown",
}

MIN_YEAR = 2020  # Valorant arrancó en 2020; fechas anteriores son artefactos.


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def read_table(path: str | Path) -> pd.DataFrame | None:
    """Lee ``<path>.parquet`` (o ``.csv``) si existe; devuelve ``None`` si no.

    Útil para las tablas de equipo opcionales (``team_stats``, ``team_profiles``,
    ...) que aún se están scrapeando: el EDA no debe romperse si faltan.
    """
    p = Path(path)
    for candidate in (p.with_suffix(".parquet"), p.with_suffix(".csv")):
        if candidate.exists():
            return (pd.read_parquet(candidate) if candidate.suffix == ".parquet"
                    else pd.read_csv(candidate))
    return None


def load_tables(data_dir: str | Path) -> dict[str, pd.DataFrame | None]:
    """Carga todas las tablas disponibles bajo ``data_dir``.

    Devuelve un diccionario con claves ``matches``, ``rankings``, ``features``
    (obligatorias) y ``teams``, ``team_stats``, ``team_profiles``,
    ``team_rosters`` (opcionales, ``None`` si no existen).
    """
    data_dir = Path(data_dir)
    interim = data_dir / "interim"
    processed = data_dir / "processed"

    tables: dict[str, pd.DataFrame | None] = {
        "matches": read_table(interim / "matches"),
        "rankings": read_table(interim / "rankings"),
        "features": read_table(processed / "features"),
        # tablas de equipo opcionales (aún en scraping)
        "teams": read_table(interim / "teams"),
        "team_stats": read_table(interim / "team_stats"),
        "team_profiles": read_table(interim / "team_profiles"),
        "team_rosters": read_table(interim / "team_rosters"),
    }

    missing = [k for k in ("matches", "rankings", "features") if tables[k] is None]
    if missing:
        raise FileNotFoundError(
            f"Faltan tablas obligatorias en {data_dir}: {', '.join(missing)}")

    return tables


# ---------------------------------------------------------------------------
# Limpieza / transformaciones
# ---------------------------------------------------------------------------

def parse_dates(df: pd.DataFrame, date_col: str = "date",
                backup_col: str | None = "unix_timestamp") -> pd.Series:
    """Convierte la columna de fecha a ``datetime64[ns]``, con respaldo.

    El formato de vlr.gg es humano (``"Thu, October 22, 2026"``); si no parsea,
    se intenta ``backup_col`` (que en la práctica es un datetime string, no epoch).
    """
    d = pd.to_datetime(df[date_col], errors="coerce")
    if backup_col and backup_col in df.columns:
        b = pd.to_datetime(df[backup_col], errors="coerce")
        d = d.where(d.notna(), b)
    return d.astype("datetime64[ns]")


def valid_dates_mask(dates: pd.Series, min_year: int = MIN_YEAR) -> pd.Series:
    """Máscara booleana de fechas válidas (no-NaT y posteriores a ``min_year``)."""
    return dates.notna() & (dates.dt.year >= min_year)


def is_placeholder(name: Any) -> bool:
    """True si ``name`` es un placeholder de equipo (forfeit/bye/tbd/...)."""
    if name is None:
        return True
    if isinstance(name, float) and np.isnan(name):
        return True
    return str(name).strip().lower() in PLACEHOLDER_TEAMS


def win_rate_by_team(df: pd.DataFrame, min_matches: int = 1) -> pd.DataFrame:
    """Win rate global por equipo a partir de la tabla de features.

    ``df`` debe tener ``team1``, ``team2`` y ``y`` (1 = gana team1). Apila ambas
    posiciones (team1 y team2) en formato largo y agrega ``n``, ``wins`` y
    ``win_rate`` por equipo. Ordena por nº de partidos descendente.
    """
    long = pd.concat([
        df[["team1", "y"]].rename(columns={"team1": "team"})
            .assign(won=lambda x: x["y"].eq(1).astype(int)),
        df[["team2", "y"]].rename(columns={"team2": "team"})
            .assign(won=lambda x: x["y"].eq(0).astype(int)),
    ], ignore_index=True)

    agg = (
        long.groupby("team")
        .agg(n=("won", "size"), wins=("won", "sum"))
        .reset_index()
    )
    agg["win_rate"] = agg["wins"] / agg["n"]
    agg = agg[agg["n"] >= min_matches].sort_values("n", ascending=False)
    return agg.reset_index(drop=True)


def head_to_head_matrix(df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Matriz head-to-head (win rate del equipo de la fila vs el de la columna).

    Solo considera partidos entre equipos de ``teams``. La celda (i, j) es la
    proporción de partidos en los que ganó ``teams[i]`` (jugando como team1 o
    team2) contra ``teams[j]``. Sin enfrentamientos → ``NaN``.
    """
    mask = df["team1"].isin(teams) & df["team2"].isin(teams)
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame(index=teams, columns=teams, dtype=float)

    # aplanar cada partido en dos filas dirigidas (ganador vs perdedor)
    rows = []
    for _, r in sub.iterrows():
        winner = r["team1"] if r["y"] == 1 else r["team2"]
        loser = r["team2"] if r["y"] == 1 else r["team1"]
        rows.append((winner, loser))
    directed = pd.DataFrame(rows, columns=["winner", "loser"])
    counts = directed.groupby(["winner", "loser"]).size().reset_index(name="n")
    total = directed.groupby(["winner"]).size().rename("total")

    piv_n = counts.pivot(index="winner", columns="loser", values="n")
    piv_t = piv_n.reindex(index=teams, columns=teams)
    denom = total.reindex(teams)
    mat = piv_t.div(denom, axis=0)
    # enfrentarse a sí mismo no tiene sentido
    mat = mat.copy()
    arr = mat.to_numpy(copy=True)
    np.fill_diagonal(arr, np.nan)
    return pd.DataFrame(arr, index=mat.index, columns=mat.columns)


# ---------------------------------------------------------------------------
# Helpers de visualización
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Aplica un estilo consistente a todas las figuras del notebook."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_target_balance(y: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Barra horizontal del balance del target ``y`` (1 = gana team1)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    vc = y.value_counts(normalize=True).sort_index()
    labels = {1: "Gana team1", 0: "Gana team2"}
    colors = [COLOR_TEAM1 if k == 1 else COLOR_TEAM2 for k in vc.index]
    bars = ax.barh(
        [labels.get(k, str(k)) for k in vc.index],
        vc.values, color=colors, alpha=0.85)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proporción de partidos")
    ax.set_title("Balance del target y (1 = gana team1)")
    return ax


def plot_corr_heatmap(corr: pd.DataFrame, title: str,
                      figsize: tuple[int, int] = (9, 7),
                      annot: bool = True) -> plt.Axes:
    """Heatmap de una matriz de correlación (con máscara del triángulo superior)."""
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=annot, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return ax


def plot_feature_vs_target(df: pd.DataFrame, features: list[str],
                           target: str = "y",
                           kind: str = "violin",
                           figsize: tuple[int, int] = (16, 10)) -> list[plt.Axes]:
    """Grid de boxplots/violin de varias features segmentadas por el target.

    Devuelve la lista de ejes (uno por feature). ``kind`` ∈ {"box", "violin"}.
    """
    n = len(features)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, feat in enumerate(features):
        ax = axes[i]
        draw = sns.boxplot if kind == "box" else sns.violinplot
        draw(data=df, x=target, y=feat, hue=target, legend=False, ax=ax,
             palette={0: COLOR_TEAM2, 1: COLOR_TEAM1},
             inner="quart" if kind == "box" else None)
        ax.set_title(feat)
        ax.set_xlabel("y (0 = team2, 1 = team1)")
        ax.set_ylabel("")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distribución de features por target ({kind})", fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return list(axes[:n])
