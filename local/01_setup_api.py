#!/usr/bin/env python3
"""01 · Configura y arranca la API local (vlrggapi).

Hace falta una sola vez. Clona `vlrggapi` (si no está), instala sus dependencias
con ``uv sync`` y arranca el servidor en ``http://127.0.0.1:3001``.

Requisitos: ``git``, ``uv`` y ``python`` en el PATH.

Uso:
    python local/01_setup_api.py
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent.parent
API_DIR = REPO / "vlrggapi"
BASE_URL = "http://127.0.0.1:3001"


def _api_ready() -> bool:
    try:
        return requests.get(f"{BASE_URL}/version", timeout=3).status_code == 200
    except requests.RequestException:
        return False


def main() -> int:
    # 1) Clonar vlrggapi si no existe
    if not (API_DIR / "main.py").exists():
        print(f"Clonando vlrggapi en {API_DIR} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/axsddlr/vlrggapi.git", str(API_DIR)],
            check=True,
        )

    # 2) Instalar dependencias (uv sync) si no hay venv
    if not (API_DIR / ".venv").exists():
        print("Instalando dependencias de vlrggapi (uv sync) ...")
        subprocess.run(["uv", "sync"], cwd=API_DIR, check=True)

    # 3) Arrancar el servidor si no responde
    if _api_ready():
        print(f"La API ya está corriendo en {BASE_URL}")
        return 0

    print(f"Arrancando la API en {BASE_URL} ...")
    subprocess.Popen(
        ["uv", "run", "python", "main.py"],
        cwd=API_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(60):  # hasta ~120 s
        time.sleep(2)
        if _api_ready():
            print(f"API lista en {BASE_URL}")
            return 0

    print("ERROR: la API no respondió en 120 s.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
