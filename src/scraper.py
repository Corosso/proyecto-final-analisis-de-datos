"""Cliente robusto para vlrggapi (v2.0.0).

Módulo de scraping (Fases F0/F1). Encapsula:

* Resolución del *rate-limit tier* por ruta (ver ``docs/api_endpoints.md``).
* Throttling por tier (intervalo mínimo entre llamadas) + reintentos con
  *backoff* exponencial en ``429``/``5xx``, respetando ``Retry-After`` y
  registrando ``X-RateLimit-Remaining``.
* Wrappers por endpoint que devuelven el ``data`` ya desenvuelto del envelope
  ``{"status": "success", "data": {...}}``.
* Checkpoint/resume: manifiesto JSON de ids/páginas ya descargados para retomar
  tras una desconexión de Colab.

Diseñado para correr en Google Colab (rutas configurables) y también local.
Compatibilidad: Python 3.11+. Solo depende de ``requests`` (ya en requirements).

Uso típico::

    from scraper import VLRClient, wait_for_api

    client = VLRClient(base_url="http://127.0.0.1:3001",
                       data_dir="/content/drive/MyDrive/.../data")
    wait_for_api(client.base_url)
    data = client.match_results(num_pages=1)   # -> dict con 'segments' y 'meta'
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger("vlr.scraper")

# ---------------------------------------------------------------------------
# Rate limit tiers (ver docs/api_endpoints.md). Intervalo mínimo por llamada,
# en segundos, con un margen de seguridad sobre el límite (60/limit * 1.05).
# ---------------------------------------------------------------------------
TIER_INTERVALS: dict[str, float] = {
    "cheap": 0.5,       # 200/min  -> ~120/min (amplio margen)
    "moderate": 1.6,    # 60/min   -> ~37/min  (el bucket es compartido entre fases)
    "expensive": 4.0,   # 20/min   -> 15/min   (margen)
}

# Query values de /v2/match que caen en el tier "moderate" (el resto "expensive").
_MODERATE_MATCH_Q = frozenset({"upcoming", "live_score"})

# Regiones válidas para /v2/rankings.
RANKING_REGIONS = [
    "na", "eu", "ap", "la", "la-s", "la-n", "oce", "kr", "mn", "gc",
    "br", "cn", "jp", "col",
]

# Regiones válidas para /v2/stats.
STATS_REGIONS = ["all", "americas", "emea", "pacific", "china", "intl"]

# Campos válidos para include= en match details (usamos la forma path).
MATCH_DETAIL_INCLUDE_FIELDS = [
    "event", "date", "map_vetos", "status", "teams", "streams", "vods",
    "maps", "head_to_head", "performance", "economy", "economy_by_map",
]

# Página máxima de resultados/equipos/jugadores aceptada por la API.
MAX_PAGE = 100
# Ventana máxima de páginas por request en /v2/match.
MAX_PAGE_WINDOW = 20


class VLRError(RuntimeError):
    """Error de la API vlrggapi (4xx/5xx no recuperable)."""

    def __init__(self, message: str, status_code: int | None = None,
                 response: requests.Response | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RateLimitError(VLRError):
    """Se agotaron los reintentos ante 429."""


def _log(level: int, msg: str, *args: Any) -> None:
    if logger.isEnabledFor(level):
        logger.log(level, msg, *args)


def _retry_after_seconds(response: requests.Response, default: float = 3.0) -> float:
    """Extrae el tiempo de espera sugerido ante un 429 (header o body)."""
    header = response.headers.get("Retry-After")
    if header is not None:
        try:
            return max(0.0, float(header))
        except ValueError:
            # Podría ser una fecha HTTP; no la parseamos, usamos default.
            pass
    try:
        detail = response.json().get("detail")
        if isinstance(detail, dict):
            ra = detail.get("retry_after")
            if ra is not None:
                return max(0.0, float(ra))
    except Exception:  # noqa: BLE001 - body no JSON
        pass
    return default


class TierThrottle:
    """Throttling simple por tier: asegura un intervalo mínimo entre llamadas.

    No es un token bucket estricto, pero es suficiente y determinista para un
    cliente secuencial en Colab. Es *thread-safe* vía lock.
    """

    def __init__(self, intervals: dict[str, float] | None = None,
                 jitter: float = 0.05):
        self._intervals = dict(intervals or TIER_INTERVALS)
        self._jitter = jitter
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def _interval(self, tier: str) -> float:
        return self._intervals.get(tier, self._intervals.get("expensive", 3.0))

    def wait(self, tier: str) -> None:
        """Duerme lo necesario para respetar el intervalo del tier.

        Guarda la marca de tiempo *después* de dormir (cuando la petición va a
        salir de verdad), de modo que el intervalo se mida entre salidas
        consecutivas y no entre comienzos.
        """
        interval = self._interval(tier)
        with self._lock:
            last = self._last.get(tier)
        if last is not None:
            elapsed = time.monotonic() - last
            sleep_for = interval - elapsed
            if sleep_for > 0:
                sleep_for += random.uniform(0, self._jitter)  # jitter anti-sincronía
                _log(logging.DEBUG, "throttle[%s] durmiendo %.2fs", tier, sleep_for)
                time.sleep(sleep_for)
        with self._lock:
            self._last[tier] = time.monotonic()


class Checkpoint:
    """Manifiesto de progreso en JSON (ids/páginas ya descargados).

    Cada *sección* guarda una estructura serializable (listas o dicts). El
    método ``save`` escribe de forma atómica (tmp + rename) para no corromper el
    archivo si Colab se corta a mitad de escritura.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    self._data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                logger.warning("Manifiesto corrupto en %s; se reinicia.", self.path)
                self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with self._lock:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._data, fh, ensure_ascii=False, indent=1)
            # OneDrive/9p puede bloquear el destino durante la sincronización:
            # reintentamos el rename y, si no, escribimos directo.
            for _ in range(6):
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:
                    time.sleep(0.5)
            try:
                with self.path.open("w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, ensure_ascii=False, indent=1)
            except OSError as exc:
                logger.warning("No se pudo guardar el manifiesto %s: %s",
                               self.path, exc)

    # -- acceso genérico ----------------------------------------------------
    def section(self, name: str, default: Any) -> Any:
        return self._data.setdefault(name, default)

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def set(self, name: str, value: Any) -> None:
        self._data[name] = value

    # -- helpers de sets/listas --------------------------------------------
    def is_done(self, section: str, key: str) -> bool:
        store = self._data.setdefault(section, [])
        return key in store

    def mark_done(self, section: str, key: str) -> None:
        store = self._data.setdefault(section, [])
        if key not in store:
            store.append(key)

    def done_list(self, section: str) -> list:
        return list(self._data.setdefault(section, []))


# ---------------------------------------------------------------------------
# Utilidades de archivos
# ---------------------------------------------------------------------------

def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json_atomic(path: str | os.PathLike, payload: Any) -> None:
    """Escribe JSON de forma atómica (tmp + rename), con reintento ante locks."""
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    for _ in range(6):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            time.sleep(0.5)
    with p.open("w", encoding="utf-8") as fh:  # fallback: escritura directa
        json.dump(payload, fh, ensure_ascii=False)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------

class VLRClient:
    """Cliente HTTP con throttling, reintentos y checkpoint para vlrggapi."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:3001",
        data_dir: str | os.PathLike = "data",
        intervals: dict[str, float] | None = None,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        timeout: float = 90.0,
        session: requests.Session | None = None,
        throttle: TierThrottle | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.data_dir = Path(data_dir)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", "frf98-valorant-etl/0.1")
        self.throttle = throttle or TierThrottle(intervals)

        # Checkpoint persistido en data_dir/_checkpoint/manifest.json
        self.checkpoint = Checkpoint(self.data_dir / "_checkpoint" / "manifest.json")

    # ------------------------------------------------------------------
    # Resolución de tier
    # ------------------------------------------------------------------
    @staticmethod
    def resolve_tier(path: str, params: dict[str, Any] | None = None) -> str:
        """Devuelve el tier para una ruta ``/v2/...`` y sus query params."""
        params = params or {}
        p = path.rstrip("/").lower()
        if not p.startswith("/v2"):
            # Rutas legacy: mapeo básico.
            if p in ("/version", "/health"):
                return "cheap"
            return "expensive"

        # Orden importante: prefijos más específicos primero.
        tiers: list[tuple[str, str | None]] = [
            ("/v2/match/details", "expensive"),
            ("/v2/events/matches", "moderate"),
            ("/v2/health", "cheap"),
            ("/v2/version", "cheap"),
            ("/v2/rankings", "moderate"),
            ("/v2/news", "moderate"),
            ("/v2/events", "moderate"),
            ("/v2/event/", "moderate"),
            ("/v2/search", "moderate"),
            ("/v2/stats", "expensive"),
            ("/v2/match/", "expensive"),   # /match/{id}/details, /match/{id}/logs
            ("/v2/player", "expensive"),
            ("/v2/team", "expensive"),
            ("/v2/match", None),           # depende de q
        ]
        for prefix, tier in tiers:
            # Coincidencia exacta: el prefijo conserva su slash final (p. ej.
            # "/v2/match/"), de modo que "/v2/match" NO cae en ese caso y llega
            # a la entrada que depende de `q`.
            if p == prefix:
                if tier is not None:
                    return tier
                q = str(params.get("q", "")).lower()
                return "moderate" if q in _MODERATE_MATCH_Q else "expensive"
            base = prefix.rstrip("/")
            if p.startswith(base + "/") and tier is not None:
                return tier
        return "expensive"

    # ------------------------------------------------------------------
    # Núcleo HTTP
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str,
                 params: dict[str, Any] | None = None,
                 tier: str | None = None) -> dict:
        """Ejecuta un request con throttle + reintentos y devuelve JSON."""
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if tier is None:
            tier = self.resolve_tier(path, params)

        url = f"{self.base_url}{path}"
        attempt = 0

        while True:
            self.throttle.wait(tier)
            try:
                resp = self.session.request(method, url, params=params,
                                            timeout=self.timeout)
            except requests.RequestException as exc:  # red / timeout
                attempt += 1
                if attempt > self.max_retries:
                    raise VLRError(f"Fallo de red tras {attempt} intentos: {exc}") from exc
                _log(logging.WARNING, "Red/error (%s). Reintento %d/%d",
                     exc, attempt, self.max_retries)
                time.sleep(self._backoff(attempt))
                continue

            remaining = resp.headers.get("X-RateLimit-Remaining")
            limit = resp.headers.get("X-RateLimit-Limit")
            if remaining is not None or limit is not None:
                _log(logging.DEBUG, "%s %s -> %d (tier=%s, %s/%s)",
                     method, path, resp.status_code, tier,
                     remaining, limit)

            # 429: respetar Retry-After, con un backoff mínimo para no caer en
            # un bucle cerrado cuando el servidor devuelve retry_after=0
            # (ventana deslizante "reintenta ya").
            if resp.status_code == 429:
                attempt += 1
                if attempt > self.max_retries:
                    raise RateLimitError(
                        f"Rate limit persistente en {path} (tier={tier})",
                        status_code=429, response=resp)
                wait = max(_retry_after_seconds(resp), self._backoff(attempt))
                _log(logging.WARNING, "429 en %s. Esperando %.1fs (%d/%d)",
                     path, wait, attempt, self.max_retries)
                time.sleep(wait)
                continue

            # 5xx (502/503/504...): backoff
            if resp.status_code >= 500:
                attempt += 1
                if attempt > self.max_retries:
                    raise VLRError(
                        f"Error {resp.status_code} persistente en {path}",
                        status_code=resp.status_code, response=resp)
                wait = self._backoff(attempt)
                _log(logging.WARNING, "%d en %s. Esperando %.1fs (%d/%d)",
                     resp.status_code, path, wait, attempt, self.max_retries)
                time.sleep(wait)
                continue

            # 4xx (no 429): error de validación o recurso inexistente
            if resp.status_code >= 400:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:  # noqa: BLE001
                    detail = resp.text[:300]
                raise VLRError(
                    f"{resp.status_code} en {path}: {detail}",
                    status_code=resp.status_code, response=resp)

            try:
                return resp.json()
            except ValueError as exc:
                raise VLRError(f"Respuesta no JSON en {path}") from exc

    def _backoff(self, attempt: int) -> float:
        """Backoff exponencial con jitter."""
        return self.backoff_base ** attempt + random.uniform(0, 0.3)

    # ------------------------------------------------------------------
    # Helpers de desenvuelto
    # ------------------------------------------------------------------
    @staticmethod
    def _unwrap(payload: dict) -> dict:
        """Devuelve ``data`` si el envelope V2 lo contiene; si no, el payload."""
        if isinstance(payload, dict) and payload.get("status") == "success" \
                and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload

    def _get(self, path: str, params: dict[str, Any] | None = None,
             tier: str | None = None, unwrap: bool = True) -> dict:
        payload = self._request("GET", path, params=params, tier=tier)
        return self._unwrap(payload) if unwrap else payload

    # ------------------------------------------------------------------
    # Wrappers por endpoint (devuelven `data` desenvuelto)
    # ------------------------------------------------------------------
    def version(self) -> dict:
        """``GET /version`` → ``{"version": ..., "default_api": ...}``."""
        return self._request("GET", "/version", tier="cheap")

    def health(self) -> dict:
        """``GET /v2/health`` → ``{"service": {...}, "http_client": {...}}``."""
        return self._get("/v2/health", tier="cheap")

    def match_results(self, num_pages: int = 1,
                      from_page: int | None = None,
                      to_page: int | None = None,
                      max_retries: int = 3,
                      request_delay: float = 1.0,
                      timeout: int = 30) -> dict:
        """``GET /v2/match?q=results`` (paginado). Devuelve ``data`` (segments+meta)."""
        params = {
            "q": "results",
            "num_pages": num_pages,
            "from_page": from_page,
            "to_page": to_page,
            "max_retries": max_retries,
            "request_delay": request_delay,
            "timeout": timeout,
        }
        return self._get("/v2/match", params=params)

    def match_upcoming(self) -> dict:
        """``GET /v2/match?q=upcoming``."""
        return self._get("/v2/match", params={"q": "upcoming"})

    def match_live_score(self) -> dict:
        """``GET /v2/match?q=live_score``."""
        return self._get("/v2/match", params={"q": "live_score"})

    def match_details(self, match_id: str | int,
                      include: list[str] | str | None = None,
                      light: bool = False) -> dict:
        """``GET /v2/match/{id}/details``.

        ``include``: lista o csv de campos top-level. ``light=True`` salta los
        tabs de performance/economía (más rápido).
        """
        params: dict[str, Any] = {}
        if include is not None:
            params["include"] = include if isinstance(include, str) else ",".join(include)
        if light:
            params["light"] = "true"
        return self._get(f"/v2/match/{match_id}/details", params=params)

    def events(self, q: str | None = None, page: int = 1) -> dict:
        """``GET /v2/events``. ``q`` ∈ {upcoming, completed, live}."""
        params: dict[str, Any] = {"page": page}
        if q:
            params["q"] = q
        return self._get("/v2/events", params=params)

    def event_matches(self, event_id: str | int) -> dict:
        """``GET /v2/events/matches``."""
        return self._get("/v2/events/matches", params={"event_id": str(event_id)})

    def rankings(self, region: str) -> dict:
        """``GET /v2/rankings?region=...``."""
        return self._get("/v2/rankings", params={"region": region})

    def stats(self, region: str, timespan: str = "all",
              event_id: str | None = None, map_name: str = "all",
              min_rounds: int = 200, min_rating: float = 1.55) -> dict:
        """``GET /v2/stats`` (agregados por jugador)."""
        params: dict[str, Any] = {
            "region": region,
            "timespan": timespan,
            "map": map_name,
            "min_rounds": min_rounds,
            "min_rating": min_rating,
        }
        if event_id:
            params["event_id"] = event_id
        return self._get("/v2/stats", params=params)

    def team(self, team_id: str | int) -> dict:
        """``GET /v2/team/{id}`` → ``data`` con ``segments[0]``."""
        return self._get(f"/v2/team/{team_id}")

    def team_stats(self, team_id: str | int, date_start: str = "") -> dict:
        """``GET /v2/team/{id}/stats`` (win rates por mapa)."""
        params: dict[str, Any] = {}
        if date_start:
            params["date_start"] = date_start
        return self._get(f"/v2/team/{team_id}/stats", params=params)

    def team_matches(self, team_id: str | int, page: int = 1) -> dict:
        """``GET /v2/team/{id}/matches`` (paginado, 1–100)."""
        return self._get(f"/v2/team/{team_id}/matches", params={"page": page})

    def team_roster(self, team_id: str | int) -> dict:
        """``GET /v2/team/{id}/roster``."""
        return self._get(f"/v2/team/{team_id}/roster")

    def search(self, query: str) -> dict:
        """``GET /v2/search?q=...`` → ``data.segments.results``."""
        return self._get("/v2/search", params={"q": query})


# ---------------------------------------------------------------------------
# Arranque del servidor (para Colab / local)
# ---------------------------------------------------------------------------

def start_server(server_dir: str | os.PathLike, port: int = 3001,
                 log_file: str | os.PathLike | None = None) -> subprocess.Popen:
    """Arranca vlrggapi en segundo plano con ``python main.py``.

    Devuelve el objeto ``Popen``. En Colab se ejecuta dentro del runtime, así que
    el cliente apunta a ``http://127.0.0.1:<port>``.
    """
    server_dir = Path(server_dir)
    env = dict(os.environ)
    env["PORT"] = str(port)

    kwargs: dict[str, Any] = dict(cwd=str(server_dir), env=env)
    if log_file is not None:
        fh = open(log_file, "a", encoding="utf-8")  # noqa: SIM115 - se cierra con el proc
        kwargs["stdout"] = fh
        kwargs["stderr"] = subprocess.STDOUT

    proc = subprocess.Popen(["python", "main.py"], **kwargs)
    logger.info("vlrggapi arrancado (pid=%s) en http://127.0.0.1:%s", proc.pid, port)
    return proc


def wait_for_api(base_url: str, timeout: float = 240.0,
                 interval: float = 2.0, session: requests.Session | None = None) -> bool:
    """Sondea ``GET /version`` hasta que el servidor responda.

    Devuelve ``True`` si quedó listo, ``False`` si expiró el timeout.
    """
    session = session or requests.Session()
    url = base_url.rstrip("/") + "/version"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = session.get(url, timeout=5)
            if resp.status_code == 200:
                logger.info("vlrggapi listo en %s (versión %s)",
                            base_url, resp.json().get("version"))
                return True
        except requests.RequestException:
            pass
        time.sleep(interval)
    logger.error("vlrggapi no respondió en %.0fs", timeout)
    return False
