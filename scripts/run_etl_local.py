#!/usr/bin/env python3
"""Runner local del ETL (F1) — para ejecutar fuera de Colab.

Ejemplos::

    # Smoke test rápido de la Capa 1 (resultados + eventos + rankings)
    python scripts/run_etl_local.py --layer 1 --smoke

    # Smoke test de la Capa 2 (solo 5 partidos)
    python scripts/run_etl_local.py --layer 2 --limit 5

    # Ejecución completa (todo el histórico, checkpoint/resume)
    python scripts/run_etl_local.py --layer both

Requisito: el servidor vlrggapi corriendo en ``--base-url`` (por defecto
``http://127.0.0.1:3001``). Si no está, el runner falla con un mensaje claro.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scraper import VLRClient, wait_for_api  # noqa: E402
from etl import run_layer1, run_layer2  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("run_etl")


def _progress(msg: str) -> None:
    log.info(msg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:3001")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--layer", choices=["1", "2", "both"], default="both")
    ap.add_argument("--smoke", action="store_true",
                    help="límites chicos para validar el pipeline (sin datos de equipo)")
    ap.add_argument("--limit", type=int, default=None,
                    help="acotar el número de partidos en la Capa 2")
    ap.add_argument("--max-results-pages", type=int, default=None)
    ap.add_argument("--max-events-pages", type=int, default=None)
    ap.add_argument("--max-team-match-pages", type=int, default=None)
    ap.add_argument("--no-rankings", action="store_true")
    ap.add_argument("--no-team-data", action="store_true")
    ap.add_argument("--all-teams", action="store_true",
                    help="enriquecer TODOS los equipos (no solo los rankeados)")
    args = ap.parse_args()

    client = VLRClient(base_url=args.base_url, data_dir=args.data_dir)
    if not wait_for_api(client.base_url, timeout=60.0):
        log.error("API no disponible en %s. Arranca vlrggapi primero.", args.base_url)
        return 1

    if args.layer in ("1", "both"):
        rankings = not args.no_rankings
        team_data = not args.no_team_data
        max_results_pages = args.max_results_pages
        max_events_pages = args.max_events_pages
        max_team_match_pages = args.max_team_match_pages
        if args.smoke:
            max_results_pages = 2
            max_events_pages = 1
            max_team_match_pages = 1
            team_data = False  # smoke: sin resolver equipos (muy lento)
            log.info("MODO SMOKE: Capa 1 con límites reducidos y sin datos de equipo.")

        kw = {"rankings": rankings, "team_data": team_data,
              "ranked_only": not args.all_teams, "progress": _progress}
        for name in ("max_results_pages", "max_events_pages", "max_team_match_pages"):
            val = locals()[name]
            if val is not None:
                kw[name] = val
        run_layer1(client, **kw)

    if args.layer in ("2", "both"):
        if args.smoke and args.limit is None:
            args.limit = 5
            log.info("MODO SMOKE: Capa 2 acotada a %d partidos.", args.limit)
        run_layer2(client, limit=args.limit, progress=_progress)

    log.info("ETL completado. Datos en %s", client.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
