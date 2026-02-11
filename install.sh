#!/usr/bin/env bash
set -euo pipefail

APP_NAME="AudioCinema"
APP_SLUG="audiocinema"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_DIR_USER="${HOME}/.local/share/icons/hicolor/256x256/apps"
ICON_DST_USER="${ICON_DIR_USER}/${APP_SLUG}.png"
DESKTOP_FILE="${DESKTOP_DIR}/${APP_NAME}.desktop"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

APT_PKGS=(
  libportaudio2
  libsndfile1
  ffmpeg
  python3-tk
  tk
  fonts-dejavu-core
  imagemagick
)

echo "[1/8] Instalando dependencias de sistema"
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y "${APT_PKGS[@]}"
else
  apt-get update
  apt-get install -y "${APT_PKGS[@]}"
fi

echo "[2/8] Creando entorno virtual"
"${PYTHON_BIN}" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[3/8] Instalando dependencias Python"
pip install -r "$ROOT_DIR/requirements.txt"

echo "[4/8] Preparando assets"
mkdir -p "$ROOT_DIR/assets" "$ROOT_DIR/config" "$ROOT_DIR/data/reports"
if [[ ! -f "$ROOT_DIR/assets/audiocinema.png" ]]; then
  cat > "$ROOT_DIR/assets/audiocinema.png" <<'EOF'
PLACEHOLDER: reemplaza este archivo por un PNG real (ideal 256x256) para icono.
EOF
fi

if command -v convert >/dev/null 2>&1; then
  if file "$ROOT_DIR/assets/audiocinema.png" | grep -qi 'PNG image data'; then
    convert "$ROOT_DIR/assets/audiocinema.png" "$ROOT_DIR/assets/audiocinema.ico" || true
  fi
fi

echo "[5/8] Instalando icono y desktop entry"
mkdir -p "$DESKTOP_DIR" "$ICON_DIR_USER"
cp "$ROOT_DIR/assets/audiocinema.png" "$ICON_DST_USER" || true
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=${APP_NAME}
Comment=Audio analysis and channel verification
Exec=${VENV_DIR}/bin/python ${ROOT_DIR}/src/gui_app.py
Path=${ROOT_DIR}
Icon=${ICON_DST_USER}
Terminal=false
Categories=AudioVideo;Audio;
StartupNotify=true
EOF
chmod 644 "$DESKTOP_FILE"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$DESKTOP_DIR" || true
fi

echo "[6/8] Setup de la app"
python "$ROOT_DIR/src/main.py" --setup

echo "[7/8] Verificacion doctor"
python "$ROOT_DIR/src/doctor.py"

echo "[8/8] Instalando systemd user units (si existen)"
if [[ -f "$ROOT_DIR/systemd/audiocinema.service" && -f "$ROOT_DIR/systemd/audiocinema.timer" ]]; then
  mkdir -p "$SYSTEMD_USER_DIR"
  cp "$ROOT_DIR/systemd/audiocinema.service" "$SYSTEMD_USER_DIR/"
  cp "$ROOT_DIR/systemd/audiocinema.timer" "$SYSTEMD_USER_DIR/"
  systemctl --user daemon-reload
  systemctl --user enable --now audiocinema.timer
fi

echo "Instalacion completada. Ejecuta: ${VENV_DIR}/bin/python ${ROOT_DIR}/src/gui_app.py"
