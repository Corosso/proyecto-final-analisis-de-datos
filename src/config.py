"""Resolución de rutas de datos según el entorno (Colab / WSL / local).

Una sola fuente de verdad para saber dónde están los datos, de modo que el
mismo código corra sin editar nada en:

- **Google Colab**: lee/escribe en Google Drive montado.
- **WSL** (desarrollo local del scraping): en ``/home/coro/valorant-data``
  (OneDrive bloquea el ``os.replace`` atómico, por eso el scraping escribe ahí).
- **Cualquier otro entorno**: en ``data/`` relativo a la raíz del repo.

Override manual con la variable de entorno ``VALORANT_DATA_DIR``.
"""
from __future__ import annotations

import os
from pathlib import Path

# Carpeta del proyecto dentro de Google Drive (relativa a "MyDrive").
_DRIVE_PROJECT = Path("Analisis de datos") / "Proyecto final"

# Ruta WSL donde escribe el scraping (ver docstring).
_WSL_DATA_DIR = Path("/home/coro/valorant-data")


def get_repo_root() -> Path:
    """Raíz del repositorio (la carpeta que contiene ``src/``)."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Directorio de datos (contiene ``interim/``, ``processed/`` y ``bi/``)."""
    # 1) Override manual
    env = os.environ.get("VALORANT_DATA_DIR")
    if env:
        return Path(env)

    # 2) Google Colab (Drive montado)
    try:
        import google.colab  # noqa: F401
        return Path("/content/drive/MyDrive") / _DRIVE_PROJECT / "data"
    except ImportError:
        pass

    # 3) WSL: ruta del scraping
    if _WSL_DATA_DIR.exists():
        return _WSL_DATA_DIR

    # 4) Local por defecto: data/ dentro del repo
    return get_repo_root() / "data"
