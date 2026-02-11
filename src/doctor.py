from __future__ import annotations

import importlib
import sys
from pathlib import Path

from app_platform import ASSETS_DIR, CFG_DIR, DATA_DIR, REP_DIR, ensure_dirs
from configio import load_config

REQUIRED_MODULES = [
    "numpy",
    "sounddevice",
    "matplotlib",
    "paho.mqtt.client",
    "yaml",
]

OPTIONAL_MODULES = [
    "scipy.signal",
]


def check_python_modules() -> list[str]:
    issues: list[str] = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            issues.append(f"Falta módulo requerido '{mod}': {exc}")

    for mod in OPTIONAL_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as exc:
            issues.append(f"Módulo opcional no disponible '{mod}' (OK, sin filtro avanzado): {exc}")
    return issues


def check_audio_devices() -> list[str]:
    issues: list[str] = []
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        if len(devices) == 0:
            issues.append("No se detectaron dispositivos de audio.")
    except Exception as exc:
        issues.append(f"No se pudo consultar audio devices (PortAudio): {exc}")
    return issues


def check_paths() -> list[str]:
    issues: list[str] = []
    ensure_dirs()

    for path in (CFG_DIR, DATA_DIR, REP_DIR, ASSETS_DIR):
        if not path.exists():
            issues.append(f"No existe directorio requerido: {path}")

    ref = ASSETS_DIR / "reference_master.wav"
    if not ref.exists():
        issues.append(
            "No existe assets/reference_master.wav (recomendado para referencia maestra)."
        )

    icon = ASSETS_DIR / "audiocinema.png"
    if not icon.exists():
        issues.append("No existe assets/audiocinema.png (icono de la app).")

    cfg = CFG_DIR / "config.yaml"
    try:
        _ = load_config(cfg)
    except Exception as exc:
        issues.append(f"Error cargando config/config.yaml: {exc}")

    return issues


def run_doctor() -> int:
    print("== AudioCinema Doctor ==")
    issues = []
    issues.extend(check_python_modules())
    issues.extend(check_audio_devices())
    issues.extend(check_paths())

    if not issues:
        print("OK: verificación completada sin problemas.")
        return 0

    has_required_fail = any("Falta módulo requerido" in item for item in issues)
    for item in issues:
        if item.startswith("Módulo opcional") or "recomendado" in item:
            print(f"WARN: {item}")
        else:
            print(f"ERR: {item}")

    return 2 if has_required_fail else 1


if __name__ == "__main__":
    sys.exit(run_doctor())
