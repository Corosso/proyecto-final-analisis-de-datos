#!/usr/bin/env python3
"""02 · Scrapea la Capa 1: partidos + rankings + datos de equipo.

Requiere la API corriendo (ver ``01_setup_api.py``). Es idempotente: usa un
checkpoint y retoma donde quedó si se interrumpe.

Por defecto solo enriquece los equipos rankeados (ver ``--all-teams``).

Uso:
    python local/02_etl_matches.py                 # scrape completo (Capa 1)
    python local/02_etl_matches.py --smoke         # prueba rápida
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
from etl import run_layer1  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(get_data_dir()),
                    help="raíz de datos (contiene interim/ y processed/)")
    ap.add_argument("--base-url", default="http://127.0.0.1:3001")
    ap.add_argument("--all-teams", action="store_true",
                    help="enriquecer TODOS los equipos (no solo los rankeados)")
    ap.add_argument("--smoke", action="store_true",
                    help="límites reducidos para validar el pipeline")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("etl")

    client = VLRClient(base_url=args.base_url, data_dir=args.data_dir)
    if not wait_for_api(client.base_url, timeout=60):
        log.error("API no disponible en %s. Corre antes 01_setup_api.py", args.base_url)
        return 1

    kw: dict = {"rankings": True, "team_data": not args.smoke,
                "ranked_only": not args.all_teams, "progress": log.info}
    if args.smoke:
        kw.update(max_results_pages=2, max_events_pages=1, max_team_match_pages=1)

    run_layer1(client, **kw)
    log.info("Capa 1 completada. Datos en %s", client.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
