from __future__ import annotations

import argparse
import sys

from app_platform import ASSETS_DIR, CFG_DIR, DATA_DIR, REP_DIR, ensure_dirs
from configio import load_config, save_config
from doctor import run_doctor


def run_setup() -> int:
    ensure_dirs()
    cfg_path = CFG_DIR / "config.yaml"
    cfg = load_config(cfg_path)
    save_config(cfg, cfg_path)

    placeholders = {
        ASSETS_DIR / "audiocinema.png": "# Coloca aquí tu ícono PNG real (256x256 recomendado).\n",
        ASSETS_DIR / "reference_master.wav": "",
    }
    for path, text in placeholders.items():
        if not path.exists():
            path.write_bytes(text.encode("utf-8") if text else b"")

    print("Setup completo.")
    print(f"- Config: {cfg_path}")
    print(f"- Data: {DATA_DIR}")
    print(f"- Reports: {REP_DIR}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AudioCinema CLI")
    parser.add_argument("--setup", action="store_true", help="Inicializa carpetas/config por defecto")
    parser.add_argument("--doctor", action="store_true", help="Ejecuta verificaciones de entorno")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.setup:
        return run_setup()

    if args.doctor:
        return run_doctor()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
