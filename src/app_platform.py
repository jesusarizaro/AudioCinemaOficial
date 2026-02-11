from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = APP_DIR / "src"
CFG_DIR = APP_DIR / "config"
DATA_DIR = APP_DIR / "data"
REP_DIR = DATA_DIR / "reports"
ASSETS_DIR = APP_DIR / "assets"


def ensure_dirs() -> None:
    """Ensure runtime directories exist."""
    for path in (CFG_DIR, DATA_DIR, REP_DIR, ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)
