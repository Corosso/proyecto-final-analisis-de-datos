#!/usr/bin/env python3
"""03 · (OPCIONAL) Scrapea la Capa 2: stats por jugador/mapa.

Advertencia  NO es necesario para el modelo actual. Todo lo entregado usa solo la Capa 1.

Descargar la Capa 2 (detalles por partido: kills/ACS/rating por jugador) solo
tiene sentido si quieres añadir features por jugador (ver ``docs/MEJORAS.md``).
Es muy lento: ~108k partidos × 20 req/min ≈ días de scraping.

Uso:
    python local/03_etl_match_details.py --limit 500   # prueba con 500 partidos
    python local/03_etl_match_details.py               # TODO el histórico
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import get_data_dir  # noqa: E402
from scraper import VLRClient, wait_for_api  # noqa: E402
from etl import run_layer2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(get_data_dir()),
                    help="raíz de datos (contiene interim/ y processed/)")
    ap.add_argument("--base-url", default="http://127.0.0.1:3001")
    ap.add_argument("--limit", type=int, default=None,
                    help="acotar a los primeros N partidos (para pruebas)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("etl")

    client = VLRClient(base_url=args.base_url, data_dir=args.data_dir)
    if not wait_for_api(client.base_url, timeout=60):
        log.error("API no disponible en %s. Corre antes 01_setup_api.py", args.base_url)
        return 1

    run_layer2(client, limit=args.limit, progress=log.info)
    log.info("Capa 2 completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
