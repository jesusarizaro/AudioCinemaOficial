#!/usr/bin/env bash
set -euo pipefail

APP_NAME="AudioCinema"
APP_SLUG="audiocinema"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_FILE="${HOME}/.local/share/applications/${APP_NAME}.desktop"
ICON_USER="${HOME}/.local/share/icons/hicolor/256x256/apps/${APP_SLUG}.png"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

if systemctl --user list-unit-files | grep -q '^audiocinema\.timer'; then
  systemctl --user disable --now audiocinema.timer || true
fi
rm -f "${SYSTEMD_USER_DIR}/audiocinema.service" "${SYSTEMD_USER_DIR}/audiocinema.timer"
systemctl --user daemon-reload || true

rm -f "$DESKTOP_FILE" "$ICON_USER"

if [[ -d "$ROOT_DIR/venv" ]]; then
  rm -rf "$ROOT_DIR/venv"
fi

echo "Desinstalacion completada (archivos de proyecto no removidos)."
