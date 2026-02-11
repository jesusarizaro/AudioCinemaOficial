# AudioCinema (Linux Debian/Ubuntu)

Reempaque de `freq24.py` como aplicación instalable Linux, manteniendo la lógica principal en `src/gui_app.py`.

## Estructura

- `src/gui_app.py`: GUI principal (monolito original adaptado a rutas de proyecto).
- `src/main.py`: CLI con `--setup` y `--doctor`.
- `src/doctor.py`: verificación de entorno.
- `src/configio.py`: lectura/escritura YAML (`config/config.yaml`).
- `src/app_platform.py`: rutas de plataforma y creación de directorios.
- `install.sh`: instalador completo para Debian/Ubuntu.
- `uninstall.sh`: limpieza de instalación en usuario.
- `systemd/*.service|*.timer`: ejecución programada (doctor).

## Instalación

```bash
chmod +x install.sh uninstall.sh
./install.sh
```

El instalador:
1. Instala dependencias APT:
   - `libportaudio2 libsndfile1 ffmpeg python3-tk tk fonts-dejavu-core imagemagick`
2. Crea `venv/`.
3. Instala dependencias pip de `requirements.txt`.
4. Crea `.desktop` en `~/.local/share/applications/AudioCinema.desktop`.
5. Copia icono a `~/.local/share/icons/hicolor/256x256/apps/audiocinema.png`.
6. Ejecuta `python src/main.py --setup`.
7. Ejecuta `python src/doctor.py`.
8. Instala y habilita `systemd --user` timer/service (si existen).

## Ejecutar la GUI

```bash
./venv/bin/python src/gui_app.py
```

## Configuración

Archivo principal:

- `config/config.yaml`

También se usa:

- `config/tb_config.json` para ThingsBoard (creado/gestionado por la GUI).

## Assets importantes

- `assets/audiocinema.png` (icono real recomendado, 256x256).
- `assets/reference_master.wav` (referencia recomendada para validaciones/doctor).

Si no colocas estos archivos, el doctor reportará advertencias o errores claros.

## Troubleshooting

### Error con PortAudio / sounddevice

- Verifica instalación de `libportaudio2`.
- Revisa dispositivos con `python -c "import sounddevice as sd; print(sd.query_devices())"`.
- En servidores/headless puede no haber dispositivos de captura.

### Tkinter no abre GUI

- Verifica `python3-tk` instalado.
- Si usas sesión remota sin display, exporta `DISPLAY` o usa entorno gráfico local.

### FFT/audio fallan por codec o utilidades

- Verifica `ffmpeg` instalado.

### ThingsBoard no envía

- Revisa `server/port/token` en GUI.
- Verifica conectividad de red y credenciales.

## Desinstalación

```bash
./uninstall.sh
```

Esto elimina desktop entry, icono de usuario, unidades systemd de usuario y `venv/`.
