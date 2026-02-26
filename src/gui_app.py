# RECORTA POR CANAL
# ANALIZA TODOS LOS CANALES
# 10oo10
#------------------------------------------------------------Cambio añadido            
# Ajuste especial para subwoofer en Ch6)                 
#------------------------------------------------------------Cambio añadido            
# Está recordato para tener el código más limpio         
#------------------------------------------------------------Cambio añadido           
# Ajuste para agregar bandera inicial 3khz
#------------------------------------------------------------Cambio añadido
# ANALIZA BANDAS POR CANAL
#------------------------------------------------------------Cambio añadido
# SISTEMA 5.1 y 7,1
# Ajuste especial para subwoofer en LFE (último canal)   
#------------------------------------------------------------Cambio añadido
# Filtro y grabacion de ruido
#------------------------------------------------------------Cambio añadido
# Criterios de evaluación (High, Medium, Low)
#------------------------------------------------------------Cambio añadido
# Slidebar en Bands y modificación de tamaño de ventana
# Criterio añadido a Evaluation Criteria: Best Corr
#------------------------------------------------------------Cambio añadido
# Cambio de lógica para hacer PASSED en SCORE de EVALUATE con todo OK en BANDS
#------------------------------------------------------------Cambio añadido
# Ajuste de la evaluación para LFE en 7.1
#------------------------------------------------------------Cambio añadido
# Guarda los archivos WAV
#------------------------------------------------------------Cambio añadido
# Eliminación del cambio (# Ajuste de la evaluación para LFE en 7.1)
#------------------------------------------------------------Cambio añadido
# Envío a TB




import json
import os
import subprocess
import threading
import queue
import wave
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import sounddevice as sd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import importlib.util


import paho.mqtt.client as mqtt

from app_platform import APP_DIR, CFG_DIR, DATA_DIR, ASSETS_DIR, ensure_dirs
from configio import load_config, save_config










BASE_DIR = APP_DIR
TB_CONFIG_PATH = os.path.join(CFG_DIR, "tb_config.json")
AUTORUN_CONFIG_PATH = os.path.join(CFG_DIR, "autorun_config.json")
DEFAULT_WAV_PATHS = {
    "noise": os.path.join(DATA_DIR, "noise.wav"),
    "ref": os.path.join(DATA_DIR, "ref.wav"),
    "tst": os.path.join(DATA_DIR, "test.wav"),
}





# ============================================================
# Recording filter configuration
# ============================================================
FILTER_ENABLED = True
FILTER_LOW_HZ = 20.0
FILTER_HIGH_HZ = 20000.0
FILTER_ORDER = 4
FILTER_FADE_MS = 10.0
FILTER_DEBUG_SPECTRAL = False

_SCIPY_SIGNAL_SPEC = importlib.util.find_spec("scipy.signal")
if _SCIPY_SIGNAL_SPEC is not None:
    import scipy.signal as sp_signal
else:
    sp_signal = None




# ============================================================
# Band definitions (Hz)
# ============================================================
BAND_RANGES_HZ = {
    "LFE": (20.0, 149.0),
    "LF": (20.0, 149.0),
    "MF": (150.0, 1999.0),
    "HF": (2000.0, 10000.0),
}
BAND_TOL_DB = 4.0


#-----------------------------------------------------------Definición de Evaluation Criteria            
# Criterios de evaluación (High, Medium, Low)
DEFAULT_BAND_CRITERIA = "Medium (Δ<= 6dB) & (Best Corr >= 0.50)"  # NEW
CRITERIA_THRESHOLDS = {  # NEW
    "High (Δ<= 2.5dB) & (Best Corr >= 0.75)": {
        "delta_db": 2.5,
        "best_corr": 0.75,
    },
    "Medium (Δ<= 6dB) & (Best Corr >= 0.50)": {
        "delta_db": 6.0,
        "best_corr": 0.50,
    },
    "Low (Δ<=10dB) & (Best Corr >= 0.32)": {
        "delta_db": 10.0,
        "best_corr": 0.32,
    },
}

def get_delta_threshold_db_from_criteria(criteria: str) -> float:  # NEW
    criteria = (criteria or "").strip()
    if not criteria:
        return float(CRITERIA_THRESHOLDS[DEFAULT_BAND_CRITERIA]["delta_db"])
    entry = CRITERIA_THRESHOLDS.get(criteria, CRITERIA_THRESHOLDS[DEFAULT_BAND_CRITERIA])
    return float(entry["delta_db"])


def get_best_corr_threshold_from_criteria(criteria: str) -> float:  # NEW
    criteria = (criteria or "").strip()
    if not criteria:
        return float(CRITERIA_THRESHOLDS[DEFAULT_BAND_CRITERIA]["best_corr"])
    entry = CRITERIA_THRESHOLDS.get(criteria, CRITERIA_THRESHOLDS[DEFAULT_BAND_CRITERIA])
    return float(entry["best_corr"])



def best_corr_to_db(best_corr: float) -> float:  # NEW
    """Convert BEST_CORR to dB using 20 * log10(|BEST_CORR|)."""
    return float(20.0 * np.log10(max(abs(best_corr), 1e-12)))




def send_to_thingsboard(server: str, port: int, token: str, telemetry: dict) -> None:
    # 1) Client compatible (evita problemas si la versión de paho cambia)
    try:
        client = mqtt.Client(
            client_id=f"ac_{os.getpid()}_{int(threading.get_ident())}",
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
    except TypeError:
        # fallback para paho antiguo (sin callback_api_version)
        client = mqtt.Client(
            client_id=f"ac_{os.getpid()}_{int(threading.get_ident())}",
            protocol=mqtt.MQTTv311
        )

    client.username_pw_set(token)

    # 2) Tiempo de espera razonable
    client.connect(server, int(port), keepalive=30)

    # 3) Arranca loop de red ANTES de publicar
    client.loop_start()

    payload = json.dumps(telemetry, ensure_ascii=False)

    # 4) QoS 1 + esperar a que realmente se publique
    info = client.publish("v1/devices/me/telemetry", payload=payload, qos=1)
    info.wait_for_publish(timeout=5)

    # 5) Validación de estado
    if not info.is_published():
        client.loop_stop()
        client.disconnect()
        raise RuntimeError("MQTT publish timeout (not published).")

    # 6) Cierra limpio
    client.loop_stop()
    client.disconnect()






#-----------------------------------------------------------Definición de Evaluation Criteria            


SYSTEM_LAYOUTS = {
    "5.1": {
        "channels": 6,
        "beeps": 8,
        "order": ["FL", "FR", "C", "SL", "SR", "LFE"],
    },
    "7.1": {
        "channels": 8,
        "beeps": 10,
        "order": ["FL", "FR", "C", "SL", "SR", "SBL", "SBR", "LFE"],
    },
}





#============================================================PARTE 1: Utilidades 

# CONVIERTE A FLOAT32 Y NORMALIZA INT
def to_float32(x: np.ndarray) -> np.ndarray:
    """Convert to float32, scaling integer PCM to [-1, 1]."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        max_abs = float(max(abs(info.min), info.max))
        return (x.astype(np.float32) / max_abs).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)






# CONVIERTE A MONO Y NORMALIZA
def normalize_mono(x: np.ndarray) -> np.ndarray:
    """Convert to mono and normalize if peak > 1.0."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / (peak + 1e-12)
    return x

# GRABA AUDIO MONO
def record_audio(duration_s: float, fs: int, device: int | None) -> np.ndarray:
    """Record mono audio from a chosen input device."""
    n = int(round(duration_s * fs))
    kwargs = dict(samplerate=fs, channels=1, dtype="float32")
    if device is not None:
        kwargs["device"] = device
    x = sd.rec(n, **kwargs)
    sd.wait()
    return normalize_mono(x.squeeze())


# APLICA FADE-IN/OUT CORTO PARA REDUCIR ARTEFACTOS
def apply_fade(x: np.ndarray, fs: int, fade_ms: float = 10.0) -> np.ndarray:
    if fade_ms <= 0 or fs <= 0:
        return x
    x = np.asarray(x)
    n = x.shape[0]
    fade_len = int(round(float(fade_ms) * 1e-3 * fs))
    if fade_len <= 0 or fade_len * 2 >= n:
        return x
    fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    if x.ndim == 1:
        x[:fade_len] *= fade
        x[-fade_len:] *= fade[::-1]
    else:
        x[:fade_len, :] *= fade[:, None]
        x[-fade_len:, :] *= fade[::-1, None]
    return x



def save_wav_mono(path: str, fs: int, x: np.ndarray) -> None:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError("Audio must be mono for saving.")
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(pcm.tobytes())


def load_wav_mono(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        if channels != 1:
            raise ValueError("Only mono WAV files are supported.")
        fs = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return fs, normalize_mono(x)



# RMS en dB
def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x * x) + 1e-20)
    return float(20.0 * np.log10(rms + 1e-20))


# RMS dB por canal
def rms_db_per_channel(x: np.ndarray) -> list[float]:
    x = np.asarray(x)
    if x.ndim == 1:
        return [rms_db(x)]
    return [rms_db(x[:, ch]) for ch in range(x.shape[1])]



def subtract_noise(x: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Subtract recorded noise from signal, matching length by truncation/tiling."""
    x = to_float32(x)
    noise = to_float32(noise)
    if noise.size == 0:
        return x
    if noise.ndim != x.ndim:
        if noise.ndim == 1 and x.ndim == 2:
            noise = noise[:, None]
        elif noise.ndim == 2 and x.ndim == 1:
            noise = noise[:, 0]
        else:
            return x
    n = x.shape[0]
    if noise.shape[0] < n:
        reps = int(np.ceil(n / noise.shape[0]))
        noise = np.tile(noise, (reps, 1)) if noise.ndim == 2 else np.tile(noise, reps)
    noise = noise[:n]
    return (x - noise).astype(np.float32, copy=False)








# FILTRO PASA-BANDA CON FILTFILT
def bandpass_filter(
    x: np.ndarray,
    fs: int,
    low_hz: float = 20.0,
    high_hz: float = 20000.0,
    order: int = 4,
    fade_ms: float = 10.0,
) -> np.ndarray:
    if fs <= 0:
        return x

    x = to_float32(x)
    nyquist = fs / 2.0

    low = max(float(low_hz), 1e-3)
    high = float(high_hz) if high_hz is not None else nyquist
    if high >= nyquist:
        high = 0.45 * fs
    high = min(high, nyquist * 0.99)

    if high <= low:
        print(f"[WARN] bandpass skipped: low={low:.2f}Hz >= high={high:.2f}Hz")
        return x

    low_norm = low / nyquist
    high_norm = high / nyquist

    if sp_signal is not None:
        sos = sp_signal.butter(int(order), [low_norm, high_norm], btype="band", output="sos")
        y = sp_signal.sosfiltfilt(sos, x, axis=0)
    else:
        n = x.shape[0]
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= low) & (freqs <= high)
        X = np.fft.rfft(x, axis=0)
        X[~mask, ...] = 0.0
        y = np.fft.irfft(X, n=n, axis=0).astype(np.float32, copy=False)
        print("[WARN] scipy not available: using FFT bandpass (may ring).")

    y = apply_fade(y, fs, fade_ms=fade_ms)
    return y.astype(np.float32, copy=False)

def _format_rms_list(rms_list: list[float]) -> str:
    return ", ".join(f"{v:.1f} dB" for v in rms_list)

def spectral_band_ratio_db(x: np.ndarray, fs: int, low: float, high: float) -> float | None:
    if x.size == 0:
        return None
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    X = np.fft.rfft(x, axis=0)
    mag2 = (X.real * X.real + X.imag * X.imag)
    in_band = (freqs >= low) & (freqs <= high)
    if not np.any(in_band):
        return None
    in_energy = np.sum(mag2[in_band, ...])
    out_energy = np.sum(mag2[~in_band, ...])
    ratio = (in_energy + 1e-20) / (out_energy + 1e-20)
    return float(10.0 * np.log10(ratio))




# PLOTEA WAVEFORM CON LINEAS DE MARCADORES Y VENTANAS
def plot_wave_with_lines(
    x: np.ndarray,
    fs: int,
    title: str,
    marker_lines_s: list[float] | None = None,
    window_lines_s: list[float] | None = None,
    shaded_ranges_s: list[tuple[float, float]] | None = None,
):
    """Orange dashed: markers. Green dashed: channel window edges. Optional shading: analysis windows."""
    t = np.arange(len(x)) / fs
    plt.figure(figsize=(12, 4.0))
    plt.plot(t, x, linewidth=0.6)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True, axis="x", linestyle=":")

    if shaded_ranges_s:
        ymin, ymax = plt.ylim()
        for a, b in shaded_ranges_s:
            plt.fill_between([a, b], [ymin, ymin], [ymax, ymax], alpha=0.12)

    if marker_lines_s:
        for s in marker_lines_s:
            plt.axvline(float(s), linestyle="--", linewidth=1.3, color="orange")

    if window_lines_s:
        for s in window_lines_s:
            plt.axvline(float(s), linestyle="--", linewidth=1.3, color="green")

    plt.tight_layout()
    plt.show()

# Calcula siguiente potencia de 2 >= n
def _next_pow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()

#============================================================PARTE 2: FRECUENCIA DOMINANTE POR VENTANA 

# ESTIMA FRECUENCIA DOMINANTE POR VENTANA (FFT PEAK TRACKING)
def sweep_freq_track(
    x: np.ndarray,
    fs: int,
    fmin: float = 20.0,
    fmax: float = 20000.0,
    win_s: float = 0.06,     # 60 ms (buena resolución para barridos)
    hop_s: float = 0.02,     # 20 ms
    gate_db_below_global: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estima frecuencia dominante por ventana (FFT peak tracking).
    Retorna:
      t_centers (frames,)
      f_est (frames,)  (NaN si está muy quieto)
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        z = np.array([], dtype=np.float32)
        return z, z

    win = max(128, int(round(win_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    if n < win:
        win = n
    nfft = _next_pow2(win)
    w = np.hanning(win).astype(np.float32)

    # Gate usando RMS global del segmento (para no leer ruido como frecuencia)
    g_db = rms_db(x)
    gate_db = g_db - float(gate_db_below_global)
    gate_lin = 10 ** (gate_db / 20.0)

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs).astype(np.float32)
    band = (freqs >= float(fmin)) & (freqs <= float(fmax))
    if not np.any(band):
        band = freqs > 0.0

    frames = 1 + max(0, (n - win) // hop)
    t = np.empty(frames, dtype=np.float32)
    f = np.full(frames, np.nan, dtype=np.float32)

    eps = 1e-18

    for i in range(frames):
        s = i * hop
        seg = x[s:s + win]
        t[i] = (s + 0.5 * win) / fs

        seg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-20))
        if seg_rms < gate_lin:
            continue

        X = np.fft.rfft(seg * w, n=nfft)
        mag2 = (X.real * X.real + X.imag * X.imag).astype(np.float32)

        m = mag2.copy()
        m[~band] = 0.0

        k = int(np.argmax(m))
        if m[k] <= eps or k <= 0 or k >= len(m) - 1:
            f[i] = float(freqs[k])
            continue

        # Interpolación parabólica para afinar
        a = float(m[k - 1])
        b = float(m[k])
        c = float(m[k + 1])
        denom = (a - 2.0 * b + c)
        if abs(denom) < 1e-20:
            delta = 0.0
        else:
            delta = 0.5 * (a - c) / denom
            delta = float(np.clip(delta, -1.0, 1.0))

        k_ref = float(k) + delta
        f[i] = float(k_ref * fs / nfft)

    return t, f

def compare_sweep_freq_curves(
    t_ref: np.ndarray,
    f_ref: np.ndarray,
    t_tst: np.ndarray,
    f_tst: np.ndarray,
    tol_hz: float = 500.0,
    min_valid_frac: float = 0.75,     # mínimo % de frames con frecuencia detectada
    pctl_err: float = 90.0,           # usamos percentil para no fallar por 1-2 outliers
    range_tol_hz: float = 200.0,      # tolerancia en min/max del barrido
) -> tuple[bool, dict]:
    """
    Compara barridos por la curva freq(t), no por waveform.
    Retorna (passed, metrics)
    """
    t_ref = np.asarray(t_ref, dtype=np.float64)
    f_ref = np.asarray(f_ref, dtype=np.float64)
    t_tst = np.asarray(t_tst, dtype=np.float64)
    f_tst = np.asarray(f_tst, dtype=np.float64)

    if t_ref.size < 5 or t_tst.size < 5:
        return False, {"reason": "too_few_frames"}

    # Intersección de tiempo común
    t0 = max(float(np.nanmin(t_ref)), float(np.nanmin(t_tst)))
    t1 = min(float(np.nanmax(t_ref)), float(np.nanmax(t_tst)))
    if t1 <= t0:
        return False, {"reason": "no_overlap_time"}

    # malla común
    dt_ref = float(np.nanmedian(np.diff(t_ref))) if t_ref.size > 2 else 0.02
    dt_tst = float(np.nanmedian(np.diff(t_tst))) if t_tst.size > 2 else 0.02
    dt = max(1e-3, min(dt_ref, dt_tst))
    tg = np.arange(t0, t1, dt, dtype=np.float64)
    if tg.size < 10:
        return False, {"reason": "overlap_too_small"}

    # Interpolar donde hay datos válidos
    def interp_nan_safe(t, f):
        m = np.isfinite(f)
        if np.sum(m) < 5:
            return np.full_like(tg, np.nan, dtype=np.float64)
        return np.interp(tg, t[m], f[m], left=np.nan, right=np.nan)

    fr = interp_nan_safe(t_ref, f_ref)
    ft = interp_nan_safe(t_tst, f_tst)

    vr = np.isfinite(fr)
    vt = np.isfinite(ft)
    both = vr & vt

    valid_frac_ref = float(np.mean(vr))
    valid_frac_tst = float(np.mean(vt))
    valid_frac_both = float(np.mean(both))

    # Si una tiene huecos donde la otra sí ve barrido, eso es “falta de información”
    missing_mismatch = float(np.mean(vr ^ vt))  # XOR

    # Si no hay suficientes puntos comparables, falla
    if valid_frac_ref < min_valid_frac or valid_frac_tst < min_valid_frac or valid_frac_both < min_valid_frac:
        return False, {
            "reason": "insufficient_valid_frames",
            "valid_frac_ref": valid_frac_ref,
            "valid_frac_tst": valid_frac_tst,
            "valid_frac_both": valid_frac_both,
            "missing_mismatch": missing_mismatch,
        }

    # Error por tiempo
    err = np.abs(fr[both] - ft[both])
    p_err = float(np.percentile(err, pctl_err))
    mean_err = float(np.mean(err))

    # Rango (min/max) del sweep
    fr_min, fr_max = float(np.nanmin(fr)), float(np.nanmax(fr))
    ft_min, ft_max = float(np.nanmin(ft)), float(np.nanmax(ft))
    range_ok = (abs(fr_min - ft_min) <= range_tol_hz) and (abs(fr_max - ft_max) <= range_tol_hz)

    passed = (p_err <= tol_hz) and range_ok and (missing_mismatch <= 0.10)

    metrics = {
        "valid_frac_ref": valid_frac_ref,
        "valid_frac_tst": valid_frac_tst,
        "valid_frac_both": valid_frac_both,
        "missing_mismatch": missing_mismatch,
        "pctl_err_hz": p_err,
        "mean_err_hz": mean_err,
        "ref_min_hz": fr_min,
        "ref_max_hz": fr_max,
        "tst_min_hz": ft_min,
        "tst_max_hz": ft_max,
        "tol_hz": float(tol_hz),
        "range_tol_hz": float(range_tol_hz),
        "pctl": float(pctl_err),
        "min_valid_frac": float(min_valid_frac),
    }
    return passed, metrics


def _interp_to_grid(t: np.ndarray, f: np.ndarray, tg: np.ndarray) -> np.ndarray:
    """Interpola f(t) a tg usando solo puntos finitos; devuelve NaN fuera de soporte."""
    t = np.asarray(t, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    tg = np.asarray(tg, dtype=np.float64)

    m = np.isfinite(t) & np.isfinite(f)
    if np.sum(m) < 5:
        return np.full_like(tg, np.nan, dtype=np.float64)

    # np.interp no maneja NaN, así que forzamos NaN fuera del rango
    out = np.interp(tg, t[m], f[m], left=np.nan, right=np.nan)
    # pero np.interp no pone NaN en left/right realmente; lo hace numérico.
    # entonces lo arreglamos:
    tmin = float(np.min(t[m])); tmax = float(np.max(t[m]))
    out[(tg < tmin) | (tg > tmax)] = np.nan
    return out


def _zscore_nan(x: np.ndarray) -> np.ndarray:
    """Z-score ignorando NaN; deja NaN donde no hay datos."""
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)
    if np.sum(m) < 5:
        return x.copy()
    mu = float(np.mean(x[m]))
    sig = float(np.std(x[m])) + 1e-12
    y = x.copy()
    y[m] = (y[m] - mu) / sig
    return y


def best_xcorr_shift_on_freq_curve(
    t_ref: np.ndarray,
    f_ref: np.ndarray,
    t_tst: np.ndarray,
    f_tst: np.ndarray,
    dt: float | None = None,
    max_lag_s: float = 1.5,
    min_valid_frac: float = 0.70,
) -> tuple[float, int, dict]:
    """
    Encuentra el desfase temporal que maximiza la correlación entre curvas f(t),
    ignorando NaNs.

    Retorna:
      best_corr (float)
      best_lag_frames (int)   # positivo => TEST va "atrasado" y hay que moverlo a la derecha en tiempo
      dbg (dict) con dt, lag_s, fr_grid, ft_grid (opcional)
    """
    t_ref = np.asarray(t_ref, dtype=np.float64)
    f_ref = np.asarray(f_ref, dtype=np.float64)
    t_tst = np.asarray(t_tst, dtype=np.float64)
    f_tst = np.asarray(f_tst, dtype=np.float64)

    if t_ref.size < 5 or t_tst.size < 5:
        return 0.0, 0, {"reason": "too_few_frames"}

    # malla común (usar dt más pequeño)
    dt_ref = float(np.nanmedian(np.diff(t_ref))) if t_ref.size > 2 else 0.02
    dt_tst = float(np.nanmedian(np.diff(t_tst))) if t_tst.size > 2 else 0.02
    if dt is None:
        dt = max(1e-3, min(dt_ref, dt_tst))
    dt = float(dt)

    # rango de tiempo conjunto "amplio" (unión)
    t0 = min(float(np.nanmin(t_ref)), float(np.nanmin(t_tst)))
    t1 = max(float(np.nanmax(t_ref)), float(np.nanmax(t_tst)))
    tg = np.arange(t0, t1, dt, dtype=np.float64)
    if tg.size < 20:
        return 0.0, 0, {"reason": "grid_too_small"}

    fr = _interp_to_grid(t_ref, f_ref, tg)
    ft = _interp_to_grid(t_tst, f_tst, tg)

    vr = np.isfinite(fr)
    vt = np.isfinite(ft)
    both = vr & vt

    valid_frac = float(np.mean(both))
    if valid_frac < min_valid_frac:
        return 0.0, 0, {
            "reason": "insufficient_overlap",
            "valid_frac_both": valid_frac,
            "dt": dt
        }

    # z-score (solo donde hay datos)
    frz = _zscore_nan(fr)
    ftz = _zscore_nan(ft)

    # para xcorr necesitamos vectores sin NaN -> usamos máscara BOTH
    a = frz[both]
    b = ftz[both]
    if a.size < 20 or b.size < 20:
        return 0.0, 0, {"reason": "too_few_aligned_points", "dt": dt}

    # correlación cruzada normalizada en ventana de lag
    max_lag_frames = int(round(max_lag_s / dt))
    max_lag_frames = max(1, min(max_lag_frames, a.size - 2))

    # xcorr directa (simple y robusta para tamaños moderados)
    # corr[lag] = corr(a, shift(b, lag))
    best_corr = -1e9
    best_lag = 0
    
    corr_values: list[float] = []




    # normalización base
    an = np.linalg.norm(a) + 1e-12
    bn = np.linalg.norm(b) + 1e-12

    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[:aa.size]
        elif lag > 0:
            bb = b[lag:]
            aa = a[:bb.size]
        else:
            aa = a
            bb = b

        if aa.size < 20:
            continue

        c = float(np.dot(aa, bb) / ((np.linalg.norm(aa) + 1e-12) * (np.linalg.norm(bb) + 1e-12)))
        corr_values.append(c)
        if c > best_corr:
            best_corr = c
            best_lag = lag

    corr_mean = float(np.mean(corr_values)) if corr_values else float("nan")
    dbg = {
        "dt": dt,
        "best_lag_frames": int(best_lag),
        "best_lag_s": float(best_lag * dt),
        "valid_frac_both": valid_frac,
        "corr_mean": corr_mean,
    }
    return float(best_corr), int(best_lag), dbg


def compare_sweep_freq_curves_xcorr(
    t_ref: np.ndarray,
    f_ref: np.ndarray,
    t_tst: np.ndarray,
    f_tst: np.ndarray,
    tol_hz: float = 200.0,
    max_lag_s: float = 1.5,
    pctl_err: float = 90.0,
    range_tol_hz: float = 200.0,
    min_valid_frac: float = 0.70,
) -> tuple[bool, dict]:
    """
    1) Encuentra el mejor desfase usando cross-correlation de la curva f(t)
    2) Aplica ese desfase y calcula errores |Δf|
    3) Decide PASS/FAIL
    """
    # 1) mejor lag
    best_corr, best_lag, dbg_x = best_xcorr_shift_on_freq_curve(
        t_ref, f_ref, t_tst, f_tst,
        dt=None,
        max_lag_s=max_lag_s,
        min_valid_frac=min_valid_frac
    )
    if "reason" in dbg_x:
        return False, {"reason": dbg_x["reason"], **dbg_x}

    dt = float(dbg_x["dt"])

    # 2) Rehacer malla común en intersección (para comparar ya alineado)
    t_ref = np.asarray(t_ref, dtype=np.float64)
    f_ref = np.asarray(f_ref, dtype=np.float64)
    t_tst = np.asarray(t_tst, dtype=np.float64)
    f_tst = np.asarray(f_tst, dtype=np.float64)

    t0 = max(float(np.nanmin(t_ref)), float(np.nanmin(t_tst)))
    t1 = min(float(np.nanmax(t_ref)), float(np.nanmax(t_tst)))
    tg = np.arange(t0, t1, dt, dtype=np.float64)
    if tg.size < 20:
        return False, {"reason": "overlap_too_small_after_grid", "dt": dt}

    fr = _interp_to_grid(t_ref, f_ref, tg)

    # aplicar desfase: si best_lag > 0, b se “corre” hacia adelante en tiempo
    # equivalente a evaluar TEST en (tg - lag*dt)
    shift_s = best_lag * dt
    ft = _interp_to_grid(t_tst, f_tst, tg - shift_s)

    vr = np.isfinite(fr)
    vt = np.isfinite(ft)
    both = vr & vt

    valid_frac_both = float(np.mean(both))
    missing_mismatch = float(np.mean(vr ^ vt))
    if valid_frac_both < min_valid_frac:
        return False, {
            "reason": "insufficient_valid_frames_after_shift",
            "valid_frac_both": valid_frac_both,
            "missing_mismatch": missing_mismatch,
            "best_corr": best_corr,
            "best_lag_s": shift_s
        }

    err = np.abs(fr[both] - ft[both])
    p_err = float(np.percentile(err, pctl_err))
    mean_err = float(np.mean(err))

    fr_min, fr_max = float(np.nanmin(fr)), float(np.nanmax(fr))
    ft_min, ft_max = float(np.nanmin(ft)), float(np.nanmax(ft))
    range_ok = (abs(fr_min - ft_min) <= range_tol_hz) and (abs(fr_max - ft_max) <= range_tol_hz)

    passed = (p_err <= tol_hz) and range_ok and (missing_mismatch <= 0.10)

    metrics = {
        "best_corr": float(best_corr),
        "best_lag_s": float(shift_s),
        "best_corr_mean": float(dbg_x.get("corr_mean", np.nan)),
        "pctl_err_hz": float(p_err),
        "mean_err_hz": float(mean_err),
        "missing_mismatch": float(missing_mismatch),
        "ref_min_hz": fr_min, "ref_max_hz": fr_max,
        "tst_min_hz": ft_min, "tst_max_hz": ft_max,
        "tol_hz": float(tol_hz),
        "range_tol_hz": float(range_tol_hz),
        "pctl": float(pctl_err),
        "min_valid_frac": float(min_valid_frac),
    }
    return passed, metrics


# ============================================================
# NEW: Channel 1 stats plot (trimmed waveform + mean ± std + gaussian inset)
# ============================================================
def plot_channel1_mean_std(
    x: np.ndarray,
    fs: int,
    ch1_start_s: float,
    ch1_end_s: float,
    title: str = "Channel 1 (trimmed) + mean ± std",
):
    """
    Plots:
      - trimmed Channel 1 waveform
      - mean (horizontal line)
      - mean ± std band (shaded)
      - histogram + Gaussian fit inset
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if not np.isfinite(ch1_start_s) or not np.isfinite(ch1_end_s) or ch1_end_s <= ch1_start_s:
        raise ValueError("Invalid Channel 1 window (start/end seconds).")

    i0 = int(max(0, round(ch1_start_s * fs)))
    i1 = int(min(len(x), round(ch1_end_s * fs)))
    if i1 - i0 < int(0.05 * fs):
        raise ValueError("Channel 1 window too short to plot.")

    seg = np.asarray(x[i0:i1], dtype=np.float64)
    t = np.arange(seg.size, dtype=np.float64) / fs


    #valores para Histograma y Gaussiana
    mu = float(np.mean(seg))
    sigma = float(np.std(seg, ddof=0)) + 1e-18



    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(t, seg, linewidth=0.7)
    ax.axhline(mu, linestyle="--", linewidth=1.4, color="black", alpha=0.9, label=f"mean = {mu:.6f}")
    ax.fill_between(t, mu - sigma, mu + sigma, alpha=0.18, label=f"±1σ = {sigma:.6f}")

    ax.set_title(title)
    ax.set_xlabel("Time inside Ch1 window [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper right")

    # Inset histogram + gaussian
    inset = ax.inset_axes([0.72, 0.12, 0.26, 0.78])
    bins = 70

    counts, edges = np.histogram(seg, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])

    inset.plot(centers, counts, linewidth=1.0, label="hist")

    gauss = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((centers - mu) / sigma) ** 2)
    gauss_scaled = gauss * (seg.size * bw)
    inset.plot(centers, gauss_scaled, linewidth=1.2, label="gauss")

    inset.set_title("Histogram + Gaussian", fontsize=9)
    inset.grid(True, linestyle=":", alpha=0.8)

    plt.tight_layout()
    plt.show()




# ============================================================
# NEW: mean over time (windowed) + plot
# ============================================================
def mean_std_over_time(
    x: np.ndarray,
    fs: int,
    win_s: float = 0.20,
    hop_s: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_center_s : (frames,)
      mean_vec   : (frames,)
      std_vec    : (frames,)  std of samples inside each window
      abs_std    : (frames,)  abs(std_vec) (same as std, but kept for clarity)
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        z = np.array([], dtype=np.float32)
        return z, z, z, z

    win = max(16, int(round(win_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    if n < win:
        win = n

    frames = 1 + max(0, (n - win) // hop)
    t = np.empty(frames, dtype=np.float32)
    m = np.empty(frames, dtype=np.float32)
    s = np.empty(frames, dtype=np.float32)

    for i in range(frames):
        a = i * hop
        seg = x[a:a + win].astype(np.float64, copy=False)
        t[i] = (a + 0.5 * win) / fs
        m[i] = float(np.mean(seg))
        s[i] = float(np.std(seg, ddof=0))

    abs_s = np.abs(s)
    return t, m, s, abs_s







#--------------------------------------------------------------------------------------------NUEVO



def plot_mean_vs_time_with_std_and_absstd(
    x_seg: np.ndarray,
    fs: int,
    title: str,
    win_s: float = 0.20,
    hop_s: float = 0.05,
    show_band: bool = True,
    use_twin_axis: bool = True,
):
    t, m, s, abs_s = mean_std_over_time(x_seg, fs, win_s=win_s, hop_s=hop_s)

    fig, ax = plt.subplots(figsize=(12, 4))
    if len(t) > 0:
        ax.plot(t, m, linewidth=0.9, label="mean")

        if show_band:
            ax.fill_between(t, m - s, m + s, alpha=0.18, label="±1σ")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Mean amplitude")
        ax.grid(True, linestyle=":")

        if use_twin_axis:
            ax2 = ax.twinx()
            ax2.plot(t, abs_s, linewidth=0.9, linestyle="--", label="abs(std)")
            ax2.set_ylabel("abs(std)")

            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right")
        else:
            ax.plot(t, abs_s, linewidth=0.9, linestyle="--", label="abs(std)")
            ax.legend(loc="upper right")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()





#-----------------------------------------------FUN RECORTE PARA CANALES
def find_sweep_segment_by_silence(
    t: np.ndarray,
    abs_std: np.ndarray,
    sweep_s: float = 10.0,
    silence_s: float = 2.0,
    smooth_frames: int = 7,
) -> tuple[int, int] | None:
    """
    Usa abs(std) para detectar dónde hay energía.
    Busca un tramo "activo" que tenga alrededor de sweep_s, y que esté rodeado por ~silencio_s.

    Retorna (i_start, i_end) índices sobre los vectores (t, abs_std),
    donde i_end es EXCLUSIVO.
    """
    if len(t) < 5:
        return None

    # --- Suavizado simple para evitar falsos picos ---
    x = abs_std.astype(np.float64, copy=False)
    if smooth_frames >= 3:
        k = int(smooth_frames)
        k = k + 1 if (k % 2 == 0) else k  # impar
        kernel = np.ones(k, dtype=np.float64) / k
        xs = np.convolve(x, kernel, mode="same")
    else:
        xs = x

    # --- Umbral robusto: baseline + 3*MAD ---
    med = float(np.median(xs))
    mad = float(np.median(np.abs(xs - med))) + 1e-12
    thr = med + 3.0 * mad

    # Evita thr demasiado pequeño
    thr = max(thr, med * 2.5, 1e-4)

    active = xs > thr

    # --- Encuentra runs activos (segmentos contiguos True) ---
    idx = np.where(active)[0]
    if idx.size == 0:
        return None

    # Construir runs
    runs = []
    start = idx[0]
    prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
        else:
            runs.append((start, prev))
            start = k
            prev = k
    runs.append((start, prev))

    # --- Elegir el run más compatible con sweep_s ---
    # Convertimos duración usando t (centros de ventana)
    def run_duration(a, b):
        return float(t[b] - t[a])

    # Preferimos runs con duración >= sweep_s*0.6 y luego el más largo
    candidates = []
    for a, b in runs:
        dur = run_duration(a, b)
        candidates.append((dur, a, b))
    candidates.sort(reverse=True)  # más largo primero

    best = None
    for dur, a, b in candidates:
        if dur >= sweep_s * 0.6:
            best = (a, b)
            break
    if best is None:
        # si todo es corto, igual tomamos el más largo
        _, a, b = candidates[0]
        best = (a, b)

    a_run, b_run = best

    # --- Ajuste final: recorte EXACTO a sweep_s ---
    # Tomamos el inicio del run y buscamos el índice final para sweep_s
    t_start = float(t[a_run])
    t_end_target = t_start + float(sweep_s)
    i_end = int(np.searchsorted(t, t_end_target, side="left"))
    i_end = max(i_end, a_run + 1)
    i_end = min(i_end, len(t))

    # --- Regla de "2s de silencio antes y después" (si existe) ---
    # Verifica que en la zona anterior y posterior el abs_std esté mayormente bajo el umbral.
    # Si no se cumple, NO fallamos: solo es una validación suave.
    def mostly_silent(i0, i1):
        if i1 <= i0:
            return True
        frac_active = float(np.mean(active[i0:i1]))
        return frac_active <= 0.10  # 90% "silencio"

    # Convertimos silencio_s a cantidad de frames aproximados usando delta t
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.05
    sil_frames = int(round(silence_s / max(dt, 1e-6)))

    i_pre0 = max(0, a_run - sil_frames)
    i_pre1 = a_run
    i_post0 = i_end
    i_post1 = min(len(t), i_end + sil_frames)

    # Si quieres ser estricto, aquí podrías "buscar" un mejor start dentro del run,
    # pero para mantenerlo estable, solo validamos suave.
    _ = mostly_silent(i_pre0, i_pre1)
    _ = mostly_silent(i_post0, i_post1)

    return a_run, i_end


def trim_by_silence_on_absstd(
    x_ch: np.ndarray,
    fs: int,
    win_s: float = 0.20,
    hop_s: float = 0.05,
    silence_s: float = 2.0,
    sweep_s: float = 10.0,
    smooth_frames: int = 9,
    floor_q: float = 30.0,          # usa el 30% más bajo como "piso"
    k_off: float = 4.0,             # histéresis: apagar (más sensible)
    k_on: float = 8.0,              # histéresis: encender (más estricto)
    sustain_s: float = 0.35,        # activo debe sostenerse ~0.35s (evita picos)
) -> tuple[np.ndarray, int, int]:
    """
    Recorta el sweep usando abs(std) vs tiempo (energía por ventana), NO amplitud.
    - Umbral adaptativo por canal (silencio es un RANGO, no un valor fijo).
    - Histéresis + sustain para evitar falsos inicios por clicks/picos.
    Devuelve: (x_trim, i0, i1) sobre x_ch original.
    """
    x = np.asarray(x_ch, dtype=np.float32)
    n = len(x)
    if n == 0:
        return x, 0, 0

    t, m, s, abs_s = mean_std_over_time(x, fs, win_s=win_s, hop_s=hop_s)
    if len(t) < 5:
        i1 = min(n, int(round(sweep_s * fs)))
        return x[:i1], 0, i1

    # --- suavizado de abs(std) ---
    xs = abs_s.astype(np.float64, copy=False)
    if smooth_frames >= 3:
        k = int(smooth_frames)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=np.float64) / k
        xs = np.convolve(xs, kernel, mode="same")

    # --- 1) estimar piso (noise floor) de forma robusta ---
    q = float(np.clip(floor_q, 5.0, 60.0))
    cut = float(np.percentile(xs, q))
    low = xs[xs <= cut]
    if low.size < 10:
        low = xs  # fallback

    med0 = float(np.median(low))
    mad0 = float(np.median(np.abs(low - med0))) + 1e-12

    # --- 2) umbrales adaptativos con histéresis ---
    thr_off = med0 + float(k_off) * mad0
    thr_on  = med0 + float(k_on)  * mad0

    # si mad sale ultra pequeño, usa spread de percentiles como respaldo
    if mad0 < 1e-10:
        p10 = float(np.percentile(xs, 10))
        p25 = float(np.percentile(xs, 25))
        spread = max(p25 - p10, 1e-12)
        thr_off = p10 + 0.8 * spread
        thr_on  = p10 + 2.0 * spread

    # --- 3) estado activo con histéresis ---
    active = np.zeros_like(xs, dtype=bool)
    state = False
    for i, v in enumerate(xs):
        if (not state) and (v > thr_on):
            state = True
        elif state and (v < thr_off):
            state = False
        active[i] = state

    # --- 4) requerir silencio sostenido antes del inicio ---
    dt = float(np.median(np.diff(t))) if len(t) > 2 else float(hop_s)
    sil_frames = max(1, int(round(silence_s / max(dt, 1e-9))))
    sus_frames = max(1, int(round(sustain_s / max(dt, 1e-9))))

    # buscar el primer inicio "real":
    # - viene precedido por ~silence_s de silencio (95% off)
    # - y el activo se sostiene sus_frames
    start_frame = None
    for i in range(len(active) - sus_frames):
        if active[i] and np.all(active[i:i + sus_frames]):
            pre0 = max(0, i - sil_frames)
            if float(np.mean(active[pre0:i])) <= 0.05:
                start_frame = i
                break

    if start_frame is None:
        # fallback: primer run sostenido
        for i in range(len(active) - sus_frames):
            if active[i] and np.all(active[i:i + sus_frames]):
                start_frame = i
                break
        if start_frame is None:
            start_frame = 0

    # frame -> samples (t es centro de ventana)
    t0 = float(t[start_frame]) - 0.5 * float(win_s)
    i0 = int(round(max(0.0, t0) * fs))

    sweep_n = int(round(sweep_s * fs))
    i1 = i0 + sweep_n

    # ajustar límites
    if i1 > n:
        i1 = n
        i0 = max(0, i1 - sweep_n)

    return x[i0:i1], i0, i1


def trim_indices_by_absstd_silence(
    x_ch: np.ndarray,
    fs: int,
    sweep_s: float = 10.0,
    silence_s: float = 2.0,
    win_s: float = 0.20,
    hop_s: float = 0.05,
    smooth_frames: int = 9,
    silence_thr: float | None = None,   # <-- aquí ajustas el “0.00025”
) -> tuple[int, int, dict]:
    """
    Recorta usando abs(std) (desviación estándar por ventanas).
    1) Calcula abs(std) vs tiempo.
    2) Define 'silencio' = abs(std) <= silence_thr (o umbral auto).
    3) Busca silencio sostenido al inicio y al final (>= silence_s).
    4) Dentro del tramo activo, elige la ventana de sweep_s con mayor energía (sum(abs_std)).
    Retorna índices (sample_i0, sample_i1) sobre x_ch y un dict debug.
    """
    # 1) stats vs tiempo
    t, m, s, abs_s = mean_std_over_time(x_ch, fs, win_s=win_s, hop_s=hop_s)

    if len(t) < 5:
        return 0, min(len(x_ch), int(round(sweep_s * fs))), {"reason": "too_short"}

    # 2) suaviza abs(std) para evitar clicks
    xs = abs_s.astype(np.float64, copy=False)
    if smooth_frames >= 3:
        k = int(smooth_frames)
        k = k + 1 if (k % 2 == 0) else k
        kernel = np.ones(k, dtype=np.float64) / k
        xs = np.convolve(xs, kernel, mode="same")

    # 3) umbral de silencio
    if silence_thr is None:
        # baseline: percentil bajo (silencio real) + margen
        p10 = float(np.percentile(xs, 10))
        p25 = float(np.percentile(xs, 25))
        # si p10≈0.00025, esto te queda cerca de ese piso pero con margen
        silence_thr = max(p10 * 1.6, p10 + 0.35 * (p25 - p10), 1e-6)

    silent = xs <= float(silence_thr)

    # 4) frames equivalentes a silence_s
    dt = float(np.median(np.diff(t))) if len(t) > 2 else hop_s
    need_frames = max(1, int(round(silence_s / max(dt, 1e-9))))

    # helper: run-length de True
    def find_first_silent_run():
        count = 0
        for i in range(len(silent)):
            count = count + 1 if silent[i] else 0
            if count >= need_frames:
                return i - count + 1, i  # inclusive
        return None

    def find_last_silent_run():
        count = 0
        for i in range(len(silent) - 1, -1, -1):
            count = count + 1 if silent[i] else 0
            if count >= need_frames:
                return i, i + count - 1  # inclusive
        return None

    first_run = find_first_silent_run()
    last_run  = find_last_silent_run()

    # Si no hay silencios claros, igual elegimos la ventana de 10s con máxima energía en todo
    if first_run is None or last_run is None or last_run[0] <= first_run[1]:
        active_i0 = 0
        active_i1 = len(t) - 1
    else:
        # tramo activo entre el final del silencio inicial y el inicio del silencio final
        active_i0 = min(len(t) - 1, first_run[1] + 1)
        active_i1 = max(0, last_run[0] - 1)

    # 5) elegir ventana de 10s con mayor energía dentro del tramo activo
    sweep_frames = max(2, int(round(sweep_s / max(dt, 1e-9))))
    a0 = active_i0
    a1 = active_i1

    if a1 - a0 + 1 < sweep_frames:
        # si el tramo activo es más corto que 10s, anclamos al inicio posible
        best_i0 = a0
    else:
        seg = xs[a0:a1 + 1]
        # suma móvil (energía)
        c = np.cumsum(np.concatenate(([0.0], seg)))
        energy = c[sweep_frames:] - c[:-sweep_frames]
        best_off = int(np.argmax(energy))
        best_i0 = a0 + best_off

    best_i1 = min(len(t) - 1, best_i0 + sweep_frames - 1)

    # 6) pasar a índices de samples (x_ch)
    sample_i0 = int(round(t[best_i0] * fs))
    sample_i1 = sample_i0 + int(round(sweep_s * fs))
    sample_i0 = int(np.clip(sample_i0, 0, max(0, len(x_ch) - 1)))
    sample_i1 = int(np.clip(sample_i1, sample_i0 + 1, len(x_ch)))

    dbg = {
        "silence_thr": float(silence_thr),
        "dt": float(dt),
        "need_frames": int(need_frames),
        "active_i0": int(active_i0),
        "active_i1": int(active_i1),
        "best_i0": int(best_i0),
        "best_i1": int(best_i1),
        "t_start": float(t[best_i0]),
        "t_end": float(t[min(best_i1, len(t)-1)]),
    }
    return sample_i0, sample_i1, dbg


def trim_by_known_structure(
    x_ch: np.ndarray,
    fs: int,
    pre_flag_s: float = 1.0,
    pre_silence_s: float = 1.0,
    sweep_s: float = 10.0,
    post_silence_s: float = 4.0,
) -> tuple[np.ndarray, int, int]:
    """
    Canal (16s):
      [0-2]  bandera
      [2-4]  silencio
      [4-14] sweep (10s)
      [14-16] silencio

    Retorna el sweep exacto: [pre_flag+pre_silence, +sweep]
    """
    x = np.asarray(x_ch, dtype=np.float32)
    n = len(x)

    start_s = pre_flag_s + pre_silence_s      # 4.0s
    end_s   = start_s + sweep_s               # 14.0s

    i0 = int(round(start_s * fs))
    i1 = int(round(end_s * fs))

    i0 = max(0, min(i0, n))
    i1 = max(i0, min(i1, n))

    return x[i0:i1], i0, i1


def trim_sweep_by_absstd(
    x_ch: np.ndarray,
    fs: int,
    sweep_s: float = 10.0,
    win_s: float = 0.20,
    hop_s: float = 0.05,
    silence_hi: float = 0.0005,   # tu definición: 0.0000–0.0005
    smooth_frames: int = 9,
    min_active_s: float = 0.30,   # cuánto debe durar "activo" para aceptarlo
    min_silence_s: float = 0.30,  # cuánto debe durar "silencio" para separarlo
    active_mult: float = 2.0,     # activo si abs(std) > silence_hi*active_mult
) -> tuple[np.ndarray, int, int, dict]:
    """
    Recorta un sweep de sweep_s usando abs(std) vs tiempo.
    Silencio: abs(std) <= silence_hi
    Activo:   abs(std) > silence_hi*active_mult  (con histéresis básica)
    Retorna: (x_trim, i0, i1, dbg)
    """

    x = np.asarray(x_ch, dtype=np.float32)
    n = len(x)
    if n == 0:
        return x, 0, 0, {"reason": "empty"}

    # 1) abs(std) por ventanas
    t, m, s, abs_s = mean_std_over_time(x, fs, win_s=win_s, hop_s=hop_s)
    if len(t) < 5:
        i1 = min(n, int(round(sweep_s * fs)))
        return x[:i1], 0, i1, {"reason": "too_short_stats"}

    xs = abs_s.astype(np.float64, copy=False)

    # 2) suavizar para evitar picos sueltos
    if smooth_frames >= 3:
        k = int(smooth_frames)
        if k % 2 == 0:
            k += 1
        kernel = np.ones(k, dtype=np.float64) / k
        xs = np.convolve(xs, kernel, mode="same")

    dt = float(np.median(np.diff(t))) if len(t) > 2 else float(hop_s)
    sweep_frames = max(2, int(round(sweep_s / max(dt, 1e-9))))
    min_active_frames = max(1, int(round(min_active_s / max(dt, 1e-9))))
    min_sil_frames = max(1, int(round(min_silence_s / max(dt, 1e-9))))

    thr_sil = float(silence_hi)
    thr_on  = float(silence_hi) * float(active_mult)
    thr_off = float(silence_hi) * 1.2  # histéresis: baja cuando vuelve cerca al silencio

    # 3) máscara activa con histéresis (evita parpadeo)
    active = np.zeros_like(xs, dtype=bool)
    state = False
    for i, v in enumerate(xs):
        if (not state) and (v > thr_on):
            state = True
        elif state and (v < thr_off):
            state = False
        active[i] = state

    # 4) encuentra runs activos "válidos" (duración mínima)
    idx = np.where(active)[0]
    if idx.size == 0:
        # fallback: escoger ventana de 10s con mayor energía total
        c = np.cumsum(np.concatenate(([0.0], xs)))
        energy = c[sweep_frames:] - c[:-sweep_frames]
        best = int(np.argmax(energy)) if energy.size else 0
        t0 = float(t[best])
        i0 = int(round(t0 * fs))
        i1 = min(n, i0 + int(round(sweep_s * fs)))
        i0 = max(0, i1 - int(round(sweep_s * fs)))
        return x[i0:i1], i0, i1, {"reason": "no_active_run", "thr_on": thr_on, "thr_off": thr_off}

    runs = []
    a = idx[0]
    p = idx[0]
    for k in idx[1:]:
        if k == p + 1:
            p = k
        else:
            runs.append((a, p))
            a = k
            p = k
    runs.append((a, p))

    # filtra runs por duración mínima
    good_runs = []
    for a, b in runs:
        if (b - a + 1) >= min_active_frames:
            good_runs.append((a, b))

    if not good_runs:
        good_runs = runs  # si todos son cortos, igual usa lo que haya

    # 5) de los runs activos, elige el mejor sweep de 10s:
    #    dentro de cada run, buscamos la ventana de sweep_frames con máxima energía
    best_i0 = None
    best_energy = -1.0
    for a, b in good_runs:
        # expandimos un poquito hacia los lados por si el run está "cortado"
        aa = max(0, a - min_sil_frames)
        bb = min(len(xs) - 1, b + min_sil_frames)

        if (bb - aa + 1) < sweep_frames:
            # no cabe 10s entero, igual lo consideramos
            seg = xs[aa:bb+1]
            e = float(np.sum(seg))
            if e > best_energy:
                best_energy = e
                best_i0 = aa
            continue

        seg = xs[aa:bb+1]
        c = np.cumsum(np.concatenate(([0.0], seg)))
        ewin = c[sweep_frames:] - c[:-sweep_frames]
        off = int(np.argmax(ewin))
        e = float(ewin[off])
        if e > best_energy:
            best_energy = e
            best_i0 = aa + off

    if best_i0 is None:
        best_i0 = int(good_runs[0][0])

    best_i1 = min(len(t) - 1, best_i0 + sweep_frames - 1)

    # 6) frames->samples (t es centro de ventana, ajustamos al inicio real aprox)
    t_start = float(t[best_i0]) - 0.5 * float(win_s)
    i0 = int(round(max(0.0, t_start) * fs))
    i1 = min(n, i0 + int(round(sweep_s * fs)))
    i0 = max(0, i1 - int(round(sweep_s * fs)))

    dbg = {
        "thr_sil": thr_sil,
        "thr_on": thr_on,
        "thr_off": thr_off,
        "dt": dt,
        "best_frame_i0": int(best_i0),
        "best_frame_i1": int(best_i1),
        "t_start_est": float(t_start),
        "t_end_est": float(t_start + sweep_s),
        "best_energy": float(best_energy),
        "runs": good_runs[:10],
    }

    return x[i0:i1], i0, i1, dbg


def extract_channel_segment(x: np.ndarray, fs: int, win_s: tuple[float, float]) -> np.ndarray:
    a, b = win_s
    if (not np.isfinite(a)) or (not np.isfinite(b)) or (b <= a):
        return np.array([], dtype=np.float32)
    i0 = int(max(0, round(float(a) * fs)))
    i1 = int(min(len(x), round(float(b) * fs)))
    if i1 <= i0:
        return np.array([], dtype=np.float32)
    return np.asarray(x[i0:i1], dtype=np.float32)


def get_trimmed_sweep_from_window(
    x: np.ndarray,
    fs: int,
    win_s: tuple[float, float],
    silence_hi: float = 0.0005,
    active_mult: float = 2.0,
    sweep_s: float = 10.0,
    win_stat_s: float = 0.20,
    hop_stat_s: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """
    Extrae el segmento del canal (ventana marker-based) y recorta el sweep (10s)
    usando abs(std) con silencio en rango (<=silence_hi).
    """
    x_ch = extract_channel_segment(x, fs, win_s)
    if x_ch.size == 0:
        return x_ch, {"reason": "empty_channel_window"}

    x_trim, i0, i1, dbg = trim_sweep_by_absstd(
        x_ch, fs,
        sweep_s=sweep_s,
        win_s=win_stat_s,
        hop_s=hop_stat_s,
        silence_hi=silence_hi,
        active_mult=active_mult,
        smooth_frames=9
    )
    dbg = dict(dbg)
    dbg.update({"cut_i0_in_channel": int(i0), "cut_i1_in_channel": int(i1), "channel_len_samples": int(x_ch.size)})
    return x_trim, dbg


def normalize_for_compare(x: np.ndarray) -> np.ndarray:
    """Zero-mean + RMS normalize (evita que el volumen domine)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float64)
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x * x) + 1e-20)
    return (x / (rms + 1e-20)).astype(np.float64)







def extract_channel_segment(x: np.ndarray, fs: int, win_s: tuple[float, float]) -> np.ndarray:
    a, b = win_s
    if (not np.isfinite(a)) or (not np.isfinite(b)) or (b <= a):
        return np.array([], dtype=np.float32)
    i0 = int(max(0, round(float(a) * fs)))
    i1 = int(min(len(x), round(float(b) * fs)))
    if i1 <= i0:
        return np.array([], dtype=np.float32)
    return np.asarray(x[i0:i1], dtype=np.float32)


def get_trimmed_sweep_from_window(
    x: np.ndarray,
    fs: int,
    win_s: tuple[float, float],
    silence_hi: float = 0.0005,
    active_mult: float = 2.0,
    sweep_s: float = 10.0,
    win_stat_s: float = 0.20,
    hop_stat_s: float = 0.05,
) -> tuple[np.ndarray, dict]:
    """
    Extrae el segmento del canal (ventana marker-based) y recorta el sweep (10s)
    usando abs(std) con silencio en rango (<=silence_hi).
    """
    x_ch = extract_channel_segment(x, fs, win_s)
    if x_ch.size == 0:
        return x_ch, {"reason": "empty_channel_window"}

    x_trim, i0, i1, dbg = trim_sweep_by_absstd(
        x_ch, fs,
        sweep_s=sweep_s,
        win_s=win_stat_s,
        hop_s=hop_stat_s,
        silence_hi=silence_hi,
        active_mult=active_mult,
        smooth_frames=9
    )
    dbg = dict(dbg)
    dbg.update({"cut_i0_in_channel": int(i0), "cut_i1_in_channel": int(i1), "channel_len_samples": int(x_ch.size)})
    return x_trim, dbg


def normalize_for_compare(x: np.ndarray) -> np.ndarray:
    """Zero-mean + RMS normalize (evita que el volumen domine)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float64)
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x * x) + 1e-20)
    return (x / (rms + 1e-20)).astype(np.float64)



def auto_silence_thr_from_absstd(xs: np.ndarray,
                                p_floor: float = 10,      # piso (10%)
                                p_hi: float = 25,         # “silencio alto” (25%)
                                k: float = 0.8,           # margen
                                min_thr: float = 1e-6):
    xs = np.asarray(xs, dtype=np.float64)
    p10 = float(np.percentile(xs, p_floor))
    p25 = float(np.percentile(xs, p_hi))
    thr = p10 + k * (p25 - p10)         # umbral dentro del rango de “silencio”
    thr = max(thr, min_thr)
    return thr



def plot_mean_vs_time(
    x_ch: np.ndarray,
    fs: int,
    title: str = "Channel — Mean vs Time",
    win_s: float = 0.20,
    hop_s: float = 0.05,
    auto_trim: bool = True,
    sweep_s: float = 10.0,
    silence_s: float = 2.0,
):
    t, m, s, abs_s = mean_std_over_time(x_ch, fs, win_s=win_s, hop_s=hop_s)

    if len(t) == 0:
        messagebox.showwarning("No data", "Empty signal.")
        return

    # --- Auto recorte usando abs(std) ---
    i0, i1 = 0, len(t)
    if auto_trim:
        seg = find_sweep_segment_by_silence(
            t, abs_s,
            sweep_s=sweep_s,
            silence_s=silence_s,
            smooth_frames=7,
        )
        if seg is not None:
            i0, i1 = seg

    tt = t[i0:i1]
    mm = m[i0:i1]
    ss = s[i0:i1]
    aa = abs_s[i0:i1]

    fig, ax = plt.subplots(figsize=(12, 4))

    # mean (línea azul)
    ax.plot(tt, mm, linewidth=0.9, label="mean")

    # mean ± std (sombra)
    ax.fill_between(tt, mm - ss, mm + ss, alpha=0.18, label="±1σ")

    ax.set_title(title + (" — AUTO-TRIM" if auto_trim else ""))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mean amplitude")
    ax.grid(True, linestyle=":")

    # abs(std) en eje derecho
    ax2 = ax.twinx()
    ax2.plot(tt, aa, linestyle="--", linewidth=1.0, label="abs(std)")
    ax2.set_ylabel("abs(std)")

    # juntar leyendas
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")


    ax.axvline(tt[0], linestyle="--", linewidth=1.0)
    ax.axvline(tt[-1], linestyle="--", linewidth=1.0)


    plt.tight_layout()
    plt.show()



# ============================================================
# Tone detection (frequency tolerance, time window)
# ============================================================
def goertzel_mag(x: np.ndarray, fs: int, f0: float) -> float:
    """Single-bin DFT magnitude using complex exponential."""
    n = len(x)
    if n <= 0:
        return 0.0
    t = np.arange(n, dtype=np.float64) / float(fs)
    w = np.exp(-1j * 2.0 * np.pi * float(f0) * t)
    v = np.dot(x.astype(np.float64), w)
    return float(np.abs(v))


def best_tone_in_band(seg: np.ndarray, fs: int, f_center: float, tol_hz: float, step_hz: float) -> tuple[float, float]:
    """
    Scan frequency band [f_center - tol, f_center + tol] and return (best_f, best_mag).
    This makes marker detection robust if tone is not exactly the requested Hz.
    """
    f0 = float(f_center)
    tol = float(max(0.0, tol_hz))
    step = float(max(1.0, step_hz))

    f_lo = max(1.0, f0 - tol)
    f_hi = f0 + tol
    freqs = np.arange(f_lo, f_hi + 0.5 * step, step, dtype=np.float64)

    best_f = f0
    best_m = -1.0
    for f in freqs:
        m = goertzel_mag(seg, fs, float(f))
        if m > best_m:
            best_m = m
            best_f = float(f)
    return best_f, float(best_m)


def detect_tone_time_in_window(
    x: np.ndarray,
    fs: int,
    f_target: float,
    t_from_s: float,
    t_to_s: float,
    frame_s: float = 0.08,
    hop_s: float = 0.02,
    min_dbfs: float = -45.0,
    freq_tol_hz: float = 100.0,
    freq_step_hz: float = 5.0,
) -> tuple[float | None, float | None]:
    """
    Detect time where a tone near f_target is strongest within [t_from_s, t_to_s].
    Returns (best_time_s, best_freq_hz_found).
    """
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        return None, None

    s0 = int(max(0, round(t_from_s * fs)))
    s1 = int(min(n, round(t_to_s * fs)))
    win = max(64, int(round(frame_s * fs)))
    hop = max(1, int(round(hop_s * fs)))

    if s1 - s0 < win:
        return None, None

    thr_lin = 10 ** (float(min_dbfs) / 20.0)

    best_score = -1.0
    best_center = None
    best_freq = None

    for s in range(s0, s1 - win + 1, hop):
        seg = x[s:s + win]
        seg_rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-20))
        if seg_rms < thr_lin:
            continue

        f_found, mag = best_tone_in_band(seg, fs, f_target, tol_hz=freq_tol_hz, step_hz=freq_step_hz)
        score = mag / (seg_rms + 1e-12)

        if score > best_score:
            best_score = score
            best_center = (s + win * 0.5) / fs
            best_freq = f_found

    if best_center is None:
        return None, None
    return float(best_center), float(best_freq) if best_freq is not None else None


# ============================================================
# Frequency estimation (targeted peak within ±tol)
# ============================================================
def _next_pow2(n: int) -> int:
    n = int(max(1, n))
    return 1 << (n - 1).bit_length()


def targeted_peak_freq_in_segment(
    x: np.ndarray,
    fs: int,
    start_s: float,
    end_s: float,
    target_hz: float,
    tol_hz: float = 100.0,
    win_s: float = 0.20,
    hop_s: float = 0.08,
    gate_db_below_seg: float = 35.0,
) -> float | None:
    """
    Find the frequency inside [target-tol, target+tol] that is strongest,
    and return the estimate (with parabolic interpolation) closest to target.
    """
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return None

    i0 = int(max(0, round(start_s * fs)))
    i1 = int(min(len(x), round(end_s * fs)))
    if i1 - i0 < int(0.10 * fs):
        return None

    segx = x[i0:i1].astype(np.float32, copy=False)

    win = max(64, int(round(win_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    if len(segx) < win:
        win = len(segx)

    nfft = _next_pow2(win)
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs).astype(np.float32)

    f0 = float(target_hz)
    tol = float(abs(tol_hz))
    band = (freqs >= (f0 - tol)) & (freqs <= (f0 + tol))
    if not np.any(band):
        return None

    g_db = rms_db(segx)
    gate_db = g_db - float(gate_db_below_seg)
    gate_lin = 10 ** (gate_db / 20.0)

    candidates = []

    for s in range(0, len(segx) - win + 1, hop):
        fr = segx[s:s + win]
        fr_rms = float(np.sqrt(np.mean(fr.astype(np.float64) ** 2) + 1e-20))
        if fr_rms < gate_lin:
            continue

        X = np.fft.rfft(fr * window, n=nfft)
        mag2 = (X.real * X.real + X.imag * X.imag).astype(np.float32)

        m = mag2.copy()
        m[~band] = 0.0
        k = int(np.argmax(m))
        if m[k] <= 1e-18:
            continue

        # refine peak
        if 1 <= k < len(m) - 1:
            a = float(m[k - 1]); b = float(m[k]); c = float(m[k + 1])
            denom = (a - 2.0 * b + c)
            if abs(denom) < 1e-20:
                delta = 0.0
            else:
                delta = 0.5 * (a - c) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
            k_ref = float(k) + delta
            f_est = float(k_ref * fs / nfft)
        else:
            f_est = float(freqs[k])

        if (f0 - tol) <= f_est <= (f0 + tol):
            candidates.append(f_est)

    if not candidates:
        return None

    candidates = np.asarray(candidates, dtype=np.float64)
    idx = int(np.argmin(np.abs(candidates - f0)))
    return float(candidates[idx])








# ============================================================
# Band analysis helpers (spectrum + band power)
# ============================================================
def compute_spectrum(
    x: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    win = np.hanning(n)
    X = np.fft.rfft(x * win)
    mag = np.abs(X).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs).astype(np.float64)
    return freqs, mag


def band_power_db(
    freqs: np.ndarray,
    mag: np.ndarray,
    f_lo: float,
    f_hi: float,
) -> float | None:
    if freqs.size == 0 or mag.size == 0:
        return None
    mask = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    if not np.any(mask):
        return None
    power = float(np.mean(mag[mask] ** 2))
    return float(10.0 * np.log10(power + 1e-20))


def align_by_lag_s(
    x_ref: np.ndarray,
    x_tst: np.ndarray,
    fs: int,
    lag_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(lag_s):
        return x_ref, x_tst

    shift = int(round(abs(lag_s) * fs))
    if shift <= 0:
        return x_ref, x_tst

    if lag_s > 0:
        x_tst = x_tst[shift:]
    else:
        x_ref = x_ref[shift:]

    n = min(len(x_ref), len(x_tst))
    if n <= 0:
        return np.array([], dtype=x_ref.dtype), np.array([], dtype=x_tst.dtype)
    return x_ref[:n], x_tst[:n]









# ============================================================
# Marker logic for your specific pattern
# ============================================================
def detect_markers_pattern(
    x: np.ndarray,
    fs: int,
    f_start: float,
    f_ch_marker: float,
    f_end: float,
    channel_count: int,
    start_to_ch1_s: float = 4.0,
    spacing_s: float = 16.0,
    time_radius_s: float = 3.0,
    freq_tol_hz: float = 100.0,
) -> dict:
    """
    Pattern:
      - Start marker: 3000 Hz near start
      - Ch1..ChN markers: 4500 Hz at start + start_to_ch1 + (i-1)*spacing
      - End marker: 3000 Hz near Ch1 + N*spacing
    """
    dur_s = len(x) / fs if fs > 0 else 0.0

    t_start, f_start_found = detect_tone_time_in_window(
        x, fs, f_start,
        t_from_s=0.0,
        t_to_s=min(dur_s, max(5.0, spacing_s * 1.5)),
        freq_tol_hz=freq_tol_hz
    )

    ch_times = [None] * channel_count
    ch_freq_found = [None] * channel_count

    if t_start is None:
        return {
            "start_time": None,
            "start_freq_found": None,
            "ch_times": ch_times,
            "ch_freq_found": ch_freq_found,
            "end_time": None,
            "end_freq_found": None,
        }

    t1_exp = t_start + float(start_to_ch1_s)
    t_from = max(0.0, t1_exp - time_radius_s)
    t_to = min(dur_s, t1_exp + time_radius_s)
    t1, f1_found = detect_tone_time_in_window(
        x, fs, f_ch_marker,
        t_from_s=t_from,
        t_to_s=t_to,
        freq_tol_hz=freq_tol_hz
    )
    ch_times[0] = t1
    ch_freq_found[0] = f1_found

    if t1 is None:
        return {
            "start_time": t_start,
            "start_freq_found": f_start_found,
            "ch_times": ch_times,
            "ch_freq_found": ch_freq_found,
            "end_time": None,
            "end_freq_found": None,
        }

    for i in range(2, channel_count + 1):
        t_exp = t1 + (i - 1) * spacing_s
        t_from = max(0.0, t_exp - time_radius_s)
        t_to = min(dur_s, t_exp + time_radius_s)
        ti, fi = detect_tone_time_in_window(
            x, fs, f_ch_marker,
            t_from_s=t_from,
            t_to_s=t_to,
            freq_tol_hz=freq_tol_hz
        )
        ch_times[i - 1] = ti
        ch_freq_found[i - 1] = fi

    t_end_exp = t1 + float(channel_count) * spacing_s
    t_from = max(0.0, t_end_exp - time_radius_s)
    t_to = min(dur_s, t_end_exp + max(time_radius_s, 5.0))
    t_end, f_end_found = detect_tone_time_in_window(
        x, fs, f_end,
        t_from_s=t_from,
        t_to_s=t_to,
        freq_tol_hz=freq_tol_hz
    )

    return {
        "start_time": t_start,
        "start_freq_found": f_start_found,
        "ch_times": ch_times,
        "ch_freq_found": ch_freq_found,
        "end_time": t_end,
        "end_freq_found": f_end_found,
    }


def build_channel_windows(
    markers: dict,
    offset_in_channel_s: float,
    channel_len_s: float,
    guard_s: float,
    channel_count: int,
) -> tuple[list[tuple[float, float]], list[float], list[float], list[tuple[float, float]]]:
    """
    For each channel i:
      start = t_ch_i + offset + guard
      end   = start + channel_len - 2*guard
    Also cap by end_time if present.
    """
    ch_times = markers.get("ch_times", [None] * channel_count)
    start_time = markers.get("start_time", None)
    end_time = markers.get("end_time", None)

    marker_lines = []
    if start_time is not None and np.isfinite(start_time):
        marker_lines.append(float(start_time))
    for ti in ch_times:
        if ti is not None and np.isfinite(ti):
            marker_lines.append(float(ti))
    if end_time is not None and np.isfinite(end_time):
        marker_lines.append(float(end_time))

    windows = []
    window_lines = []
    shaded = []

    for i in range(channel_count):
        ti = ch_times[i]
        if ti is None or not np.isfinite(ti):
            windows.append((np.nan, np.nan))
            continue

        a = float(ti) + float(offset_in_channel_s) + float(guard_s)
        b = a + float(channel_len_s) - 2.0 * float(guard_s)

        if end_time is not None and np.isfinite(end_time):
            b = min(b, float(end_time) - float(guard_s))

        if b <= a:
            windows.append((np.nan, np.nan))
            continue

        windows.append((a, b))
        window_lines.extend([a, b])
        shaded.append((a, b))

    return windows, marker_lines, window_lines, shaded


def analyze_track(
    x: np.ndarray,
    fs: int,
    f_start: float,
    f_ch_marker: float,
    f_end: float,
    channel_count: int,
    channel_order: list[str],
    start_to_ch1_s: float,
    spacing_s: float,
    time_radius_s: float,
    offset_in_channel_s: float,
    channel_len_s: float,
    guard_s: float,
    freq_tol_hz: float,
) -> dict:
    markers = detect_markers_pattern(
        x, fs,
        f_start=f_start,
        f_ch_marker=f_ch_marker,
        f_end=f_end,
        channel_count=channel_count,
        start_to_ch1_s=start_to_ch1_s,
        spacing_s=spacing_s,
        time_radius_s=time_radius_s,
        freq_tol_hz=freq_tol_hz,
    )

    windows_s, marker_lines, window_lines, shaded = build_channel_windows(
        markers,
        offset_in_channel_s=offset_in_channel_s,
        channel_len_s=channel_len_s,
        guard_s=guard_s,
        channel_count=channel_count,
    )

    targets = [float(f_ch_marker)] * channel_count

    freqs = []
    for i in range(channel_count):
        a, b = windows_s[i]
        if not np.isfinite(a) or not np.isfinite(b):
            freqs.append(None)
            continue

        f_est = targeted_peak_freq_in_segment(
            x, fs,
            start_s=float(a),
            end_s=float(b),
            target_hz=targets[i],
            tol_hz=freq_tol_hz
        )
        freqs.append(f_est)

    return {
        "markers": markers,
        "windows_s": windows_s,
        "marker_lines_s": marker_lines,
        "window_lines_s": window_lines,
        "shaded_ranges_s": shaded,
        "channel_freqs_hz": freqs,
        "targets_hz": targets,
        "channel_count": channel_count,
        "channel_order": list(channel_order),
    }


# ============================================================
# GUI
# ============================================================
class AudioCompareGUI:␊
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg_path = os.path.join(CFG_DIR, "config.yaml")
        self.runtime_cfg = load_config(self.cfg_path)

        self.system_var = tk.StringVar(value=str(self.runtime_cfg.get("system_layout", "5.1")))
        self.expected_channels = SYSTEM_LAYOUTS["5.1"]["channels"]
        self.expected_beeps = SYSTEM_LAYOUTS["5.1"]["beeps"]
        self.channel_order = list(SYSTEM_LAYOUTS["5.1"]["order"])
        self.root.title("Audio Test Pattern (5.1)")
        
        self.q = queue.Queue()

        self.x_ref = None  # (fs, x)
        self.x_tst = None  # (fs, x)
        self.x_noise = None  # (fs, x)
        self.wav_paths = dict(DEFAULT_WAV_PATHS)

        self.ref_analysis = None
        self.tst_analysis = None
        self.tb_config_path = TB_CONFIG_PATH
        tb_config = self._load_tb_config()
        self.tb_server_var = tk.StringVar(value=tb_config.get("server", ""))
        self.tb_port_var = tk.StringVar(value=str(tb_config.get("port", "1883")))
        self.tb_token_var = tk.StringVar(value=tb_config.get("token", ""))
        self.tb_server_var.trace_add("write", self._on_tb_config_change)
        self.tb_port_var.trace_add("write", self._on_tb_config_change)
        self.tb_token_var.trace_add("write", self._on_tb_config_change)

        autorun_cfg = self._load_autorun_config()
        self.autorun_days_var = tk.StringVar(value=autorun_cfg.get("days", "Mon,Tue,Wed,Thu,Fri"))
        self.autorun_time_var = tk.StringVar(value=autorun_cfg.get("time", "08:00:00"))
        
        self.last_tb_payload = None





        # ---------------- Device ----------------
        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Microphone (Input Device):").grid(row=0, column=0, sticky="w")
        self.device_var = tk.StringVar(value="")
        self.device_combo = ttk.Combobox(top, textvariable=self.device_var, width=60, state="readonly")
        self.device_combo.grid(row=0, column=1, padx=8, sticky="we")

        self.refresh_btn = ttk.Button(top, text="Refresh devices", command=self.refresh_devices)
        self.refresh_btn.grid(row=0, column=2, padx=6)

        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Surround sound:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.system_combo = ttk.Combobox(top, textvariable=self.system_var, width=10, state="readonly")
        self.system_combo["values"] = list(SYSTEM_LAYOUTS.keys())
        self.system_combo.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=(6, 0))
        self.system_combo.bind("<<ComboboxSelected>>", self.apply_system_layout)

        # ---------------- Params ----------------
        params = ttk.Frame(root, padding=(10, 0, 10, 10))
        params.pack(fill="x")

        audio_cfg = self.runtime_cfg.get("audio", {}) if isinstance(self.runtime_cfg.get("audio"), dict) else {}
        analysis_cfg = self.runtime_cfg.get("analysis", {}) if isinstance(self.runtime_cfg.get("analysis"), dict) else {}
        timing_cfg = self.runtime_cfg.get("timing", {}) if isinstance(self.runtime_cfg.get("timing"), dict) else {}
        marker_cfg = self.runtime_cfg.get("markers", {}) if isinstance(self.runtime_cfg.get("markers"), dict) else {}
        filter_cfg = self.runtime_cfg.get("filter", {}) if isinstance(self.runtime_cfg.get("filter"), dict) else {}

        ttk.Label(params, text="FS (Hz):").grid(row=0, column=0, sticky="w")
        self.fs_var = tk.StringVar(value=str(audio_cfg.get("samplerate", 48000)))
        ttk.Entry(params, textvariable=self.fs_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 14))

        ttk.Label(params, text="Duration (s):").grid(row=0, column=2, sticky="w")
        self.dur_var = tk.StringVar(value=str(audio_cfg.get("record_seconds", 110)))
        ttk.Entry(params, textvariable=self.dur_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 14))

        ttk.Label(params, text="REF vs TEST thr (Hz):").grid(row=0, column=4, sticky="w")
        self.thr_match_var = tk.StringVar(value=str(analysis_cfg.get("ref_vs_test_thr_hz", 50)))
        ttk.Entry(params, textvariable=self.thr_match_var, width=10).grid(row=0, column=5, sticky="w", padx=(6, 0))


        ttk.Label(params, text="Evaluation criteria").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.band_criteria_var = tk.StringVar(value=str(analysis_cfg.get("criteria", DEFAULT_BAND_CRITERIA)))
        self.band_criteria_combo = ttk.Combobox(
            params,
            textvariable=self.band_criteria_var,
            width=22,
            state="readonly",
        )
        self.band_criteria_combo["values"] = list(CRITERIA_THRESHOLDS.keys())
        self.band_criteria_combo.grid(row=1, column=1, sticky="w", padx=(6, 14), pady=(6, 0))
        self.band_criteria_combo.bind("<<ComboboxSelected>>", self.on_band_criteria_change)

        # ---------------- ThingsBoard ----------------
        tb_frame = ttk.LabelFrame(root, text="ThingsBoard (MQTT)", padding=10)
        tb_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(tb_frame, text="Server:").grid(row=0, column=0, sticky="w")
        ttk.Entry(tb_frame, textvariable=self.tb_server_var, width=28).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(tb_frame, text="Port:").grid(row=0, column=2, sticky="w")
        ttk.Entry(tb_frame, textvariable=self.tb_port_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(tb_frame, text="Token:").grid(row=0, column=4, sticky="w")
        ttk.Entry(tb_frame, textvariable=self.tb_token_var, width=36).grid(row=0, column=5, sticky="w", padx=(6, 0))

        # ---------------- Automation schedule ----------------
        auto_frame = ttk.LabelFrame(root, text="Automation schedule (systemd --user)", padding=10)
        auto_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(auto_frame, text="Days (Mon,Tue,...):").grid(row=0, column=0, sticky="w")
        ttk.Entry(auto_frame, textvariable=self.autorun_days_var, width=26).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(auto_frame, text="Hour (HH:MM:SS):").grid(row=0, column=2, sticky="w")
        ttk.Entry(auto_frame, textvariable=self.autorun_time_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Button(auto_frame, text="Apply schedule", command=self.apply_automation_schedule).grid(row=0, column=4, sticky="w")




        # ---------------- Filter ----------------
        filter_frame = ttk.LabelFrame(root, text="Bandpass filter", padding=10)
        filter_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.filter_enabled_var = tk.BooleanVar(value=bool(filter_cfg.get("enabled", FILTER_ENABLED)))
        self.filter_low_var = tk.StringVar(value=str(filter_cfg.get("low_hz", FILTER_LOW_HZ)))
        self.filter_high_var = tk.StringVar(value=str(filter_cfg.get("high_hz", FILTER_HIGH_HZ)))
        self.filter_order_var = tk.StringVar(value=str(filter_cfg.get("order", FILTER_ORDER)))

        ttk.Checkbutton(
            filter_frame,
            text="Bandpass filter (20–20kHz)",
            variable=self.filter_enabled_var,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(filter_frame, text="Low (Hz):").grid(row=0, column=1, sticky="e", padx=(12, 0))
        ttk.Entry(filter_frame, textvariable=self.filter_low_var, width=8).grid(row=0, column=2, sticky="w", padx=(4, 10))
        ttk.Label(filter_frame, text="High (Hz):").grid(row=0, column=3, sticky="e")
        ttk.Entry(filter_frame, textvariable=self.filter_high_var, width=8).grid(row=0, column=4, sticky="w", padx=(4, 10))
        ttk.Label(filter_frame, text="Order:").grid(row=0, column=5, sticky="e")
        ttk.Entry(filter_frame, textvariable=self.filter_order_var, width=6).grid(row=0, column=6, sticky="w")






        # ---------------- Pattern timing ----------------
        timing = ttk.LabelFrame(root, text="Pattern timing", padding=10)
        timing.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(timing, text="Channel spacing (s):").grid(row=0, column=0, sticky="w")
        self.spacing_var = tk.StringVar(value=str(timing_cfg.get("spacing_s", 16)))
        ttk.Entry(timing, textvariable=self.spacing_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(timing, text="Start -> Ch1 (s):").grid(row=0, column=2, sticky="w")
        self.start_to_ch1_var = tk.StringVar(value=str(timing_cfg.get("start_to_ch1_s", 4)))
        ttk.Entry(timing, textvariable=self.start_to_ch1_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(timing, text="Search radius (s):").grid(row=0, column=4, sticky="w")
        self.radius_var = tk.StringVar(value=str(timing_cfg.get("radius_s", 1)))
        ttk.Entry(timing, textvariable=self.radius_var, width=10).grid(row=0, column=5, sticky="w", padx=(6, 18))

        ttk.Label(timing, text="Freq tol (Hz):").grid(row=0, column=6, sticky="w")
        self.freq_tol_var = tk.StringVar(value=str(analysis_cfg.get("freq_tol_hz", 100)))
        ttk.Entry(timing, textvariable=self.freq_tol_var, width=10).grid(row=0, column=7, sticky="w", padx=(6, 0))

        # ---------------- Channel windows ----------------
        winbox = ttk.LabelFrame(root, text="Channel windows (inside each channel)", padding=10)
        winbox.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(winbox, text="Offset inside channel (s):").grid(row=0, column=0, sticky="w")
        self.offset_var = tk.StringVar(value=str(timing_cfg.get("offset_s", 3)))
        ttk.Entry(winbox, textvariable=self.offset_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(winbox, text="Channel length (s):").grid(row=0, column=2, sticky="w")
        self.chlen_var = tk.StringVar(value=str(timing_cfg.get("channel_len_s", 11)))
        ttk.Entry(winbox, textvariable=self.chlen_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(winbox, text="Guard (s):").grid(row=0, column=4, sticky="w")
        self.guard_var = tk.StringVar(value=str(timing_cfg.get("guard_s", 0.05)))
        ttk.Entry(winbox, textvariable=self.guard_var, width=10).grid(row=0, column=5, sticky="w", padx=(6, 0))

        # ---------------- Markers (Hz) ----------------
        markers = ttk.LabelFrame(root, text="Markers (Hz) — your pattern", padding=10)
        markers.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(markers, text="Start marker (Hz):").grid(row=0, column=0, sticky="w")
        self.f_start_var = tk.StringVar(value=str(marker_cfg.get("start_hz", 3000)))
        ttk.Entry(markers, textvariable=self.f_start_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 18))

        self.ch_marker_label = ttk.Label(markers, text="Ch1..Ch6 marker (Hz):")
        self.ch_marker_label.grid(row=0, column=2, sticky="w")
        self.f_ch_marker_var = tk.StringVar(value=str(marker_cfg.get("channel_marker_hz", 4500)))
        ttk.Entry(markers, textvariable=self.f_ch_marker_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 18))

        ttk.Label(markers, text="End marker (Hz):").grid(row=0, column=4, sticky="w")
        self.f_end_var = tk.StringVar(value=str(marker_cfg.get("end_hz", 3000)))
        ttk.Entry(markers, textvariable=self.f_end_var, width=10).grid(row=0, column=5, sticky="w", padx=(6, 0))

        # ---------------- Buttons ----------------
        btns = ttk.Frame(root, padding=10)
        btns.pack(fill="x")

        self.import_btn = ttk.Button(btns, text="Import", command=self.import_wav_files)
        self.rec_noise_btn = ttk.Button(btns, text="Record Noise", command=self.start_record_noise)
        self.rec_ref_btn = ttk.Button(btns, text="Record REF", command=lambda: self.start_record("ref"))
        self.rec_tst_btn = ttk.Button(btns, text="Record TEST", command=lambda: self.start_record("tst"))
        self.plot_ref_btn = ttk.Button(btns, text="Plot REF", command=lambda: self.plot_track("ref"), state="disabled")
        self.plot_tst_btn = ttk.Button(btns, text="Plot TEST", command=lambda: self.plot_track("tst"), state="disabled")
        self.an_ref_btn = ttk.Button(btns, text="Analyze REF", command=lambda: self.start_analyze("ref"), state="disabled")
        self.an_tst_btn = ttk.Button(btns, text="Analyze TEST", command=lambda: self.start_analyze("tst"), state="disabled")




        self.eval_btn = ttk.Button(btns, text="Evaluate", command=self.evaluate_channels, state="disabled")
        self.tb_send_btn = ttk.Button(btns, text="Thingsboard", command=self.send_thingsboard_now, state="disabled")


        self.compare_sweeps_btn = ttk.Button(
            btns,
            text="Sweeps",
            command=self.show_compare_sweeps_window,
            state="disabled"
        )


        self.compare_trim_btn = ttk.Button(
            btns,
            text="TRIM sweeps",
            command=self.show_trimmed_sweeps_window,
            state="disabled"
        )

        self.bands_btn = ttk.Button(
            btns,
            text="Bands",
            command=self.show_bands_window,
            state="disabled"
        )


        self.import_btn.pack(side="left", padx=(0, 8))
        self.rec_noise_btn.pack(side="left", padx=(0, 8))
        self.rec_ref_btn.pack(side="left", padx=(0, 8))
        self.rec_tst_btn.pack(side="left", padx=(0, 8))
        self.an_ref_btn.pack(side="left", padx=(0, 8))
        self.an_tst_btn.pack(side="left", padx=(0, 8))
        self.plot_ref_btn.pack(side="left", padx=(0, 8))
        self.plot_tst_btn.pack(side="left", padx=(0, 8))
        self.compare_trim_btn.pack(side="left", padx=(0, 8))
        self.compare_sweeps_btn.pack(side="left", padx=(0, 8))
        self.eval_btn.pack(side="left", padx=(0, 8))
        self.tb_send_btn.pack(side="left", padx=(0, 8))
        self.bands_btn.pack(side="left", padx=(0, 8))

        # ---------------- Status ----------------
        status = ttk.Frame(root, padding=(10, 0, 10, 10))
        status.pack(fill="x")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        self.file_status_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.file_status_var).pack(side="right")


        self.refresh_devices()
        self.apply_system_layout()
        self._install_runtime_config_traces()
        self._save_runtime_config()
        self.load_saved_wavs()
        self.root.after(100, self.poll_queue)

    def _as_float(self, var: tk.Variable, default: float) -> float:
        try:
            return float(str(var.get()).strip())
        except Exception:
            return float(default)

    def _as_int(self, var: tk.Variable, default: int) -> int:
        try:
            return int(float(str(var.get()).strip()))
        except Exception:
            return int(default)

    def _save_runtime_config(self, *_args) -> None:
        cfg = load_config(self.cfg_path)
        cfg["audio"] = {
            "samplerate": self._as_int(self.fs_var, 48000),
            "input_device": self.get_selected_device_index(),
            "record_seconds": self._as_float(self.dur_var, 110.0),
        }
        cfg["system_layout"] = self.system_var.get().strip() or "5.1"
        cfg["analysis"] = {
            "criteria": self.band_criteria_var.get().strip() or DEFAULT_BAND_CRITERIA,
            "ref_vs_test_thr_hz": self._as_float(self.thr_match_var, 50.0),
            "freq_tol_hz": self._as_float(self.freq_tol_var, 100.0),
        }
        cfg["timing"] = {
            "spacing_s": self._as_float(self.spacing_var, 16.0),
            "start_to_ch1_s": self._as_float(self.start_to_ch1_var, 4.0),
            "radius_s": self._as_float(self.radius_var, 1.0),
            "offset_s": self._as_float(self.offset_var, 3.0),
            "channel_len_s": self._as_float(self.chlen_var, 11.0),
            "guard_s": self._as_float(self.guard_var, 0.05),
        }
        cfg["markers"] = {
            "start_hz": self._as_float(self.f_start_var, 3000.0),
            "channel_marker_hz": self._as_float(self.f_ch_marker_var, 4500.0),
            "end_hz": self._as_float(self.f_end_var, 3000.0),
        }
        cfg["filter"] = {
            "enabled": bool(self.filter_enabled_var.get()),
            "low_hz": self._as_float(self.filter_low_var, FILTER_LOW_HZ),
            "high_hz": self._as_float(self.filter_high_var, FILTER_HIGH_HZ),
            "order": self._as_int(self.filter_order_var, FILTER_ORDER),
        }
        save_config(cfg, self.cfg_path)

    def _install_runtime_config_traces(self) -> None:
        watched_vars = [
            self.system_var,
            self.fs_var,
            self.dur_var,
            self.thr_match_var,
            self.band_criteria_var,
            self.spacing_var,
            self.start_to_ch1_var,
            self.radius_var,
            self.freq_tol_var,
            self.offset_var,
            self.chlen_var,
            self.guard_var,
            self.f_start_var,
            self.f_ch_marker_var,
            self.f_end_var,
            self.filter_enabled_var,
            self.filter_low_var,
            self.filter_high_var,
            self.filter_order_var,
            self.device_var,
        ]
        for var in watched_vars:
            var.trace_add("write", self._save_runtime_config)




    def apply_system_layout(self, _event=None):
        layout = self.system_var.get().strip()
        config = SYSTEM_LAYOUTS.get(layout, SYSTEM_LAYOUTS["5.1"])
        self.expected_channels = config["channels"]
        self.expected_beeps = config["beeps"]
        self.channel_order = list(config["order"])
        self.root.title(
            f"Audio Test Pattern ({layout}) — Start=3000, Ch1..Ch{self.expected_channels}=4500, End=3000"
        )
        if hasattr(self, "ch_marker_label"):
            self.ch_marker_label.config(text=f"Ch1..Ch{self.expected_channels} marker (Hz):")
        self.status_var.set(
            f"Ready. Layout: {layout} ({self.expected_channels} channels, {self.expected_beeps} beeps)."
        )









    def _load_tb_config(self) -> dict:
        if not os.path.exists(self.tb_config_path):
            return {}
        try:
            with open(self.tb_config_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_tb_config(self) -> None:␊
        data = {
            "server": self.tb_server_var.get().strip(),
            "port": self.tb_port_var.get().strip(),
            "token": self.tb_token_var.get().strip(),
        }
        try:
            with open(self.tb_config_path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except OSError:
            pass

    def _load_autorun_config(self) -> dict:
        if not os.path.exists(AUTORUN_CONFIG_PATH):
            return {}
        try:
            with open(AUTORUN_CONFIG_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _save_autorun_config(self, days: str, hour: str) -> None:
        data = {
            "enabled": True,
            "days": days,
            "time": hour,
        }
        try:
            with open(AUTORUN_CONFIG_PATH, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except OSError:
            pass

    def apply_automation_schedule(self) -> None:
        days = self.autorun_days_var.get().strip()
        hour = self.autorun_time_var.get().strip()
        if not days:
            messagebox.showerror("Automation", "Days cannot be empty.")
            return

        # backward-compatible: if user enters HH:MM, normalize to HH:MM:SS
        if len(hour) == 5 and hour[2] == ":":
            hour = f"{hour}:00"
            self.autorun_time_var.set(hour)

        if len(hour) != 8 or hour[2] != ":" or hour[5] != ":":
            messagebox.showerror("Automation", "Invalid hour format. Use HH:MM:SS")
            return
        try:
            hh, mm, ss = [int(part) for part in hour.split(":")]
        except ValueError:
            messagebox.showerror("Automation", "Invalid hour format. Use HH:MM:SS")
            return
        if not (0 <= hh <= 23 and 0 <= mm <= 59 and 0 <= ss <= 59):
            messagebox.showerror("Automation", "Hour out of range. Use HH:MM:SS (24h)")
            return

        unit_dir = os.path.expanduser("~/.config/systemd/user")
        os.makedirs(unit_dir, exist_ok=True)
        py_path = os.path.join(APP_DIR, "venv", "bin", "python")
        if not os.path.exists(py_path):
            py_path = "python3"

        service_body = "\n".join([
            "[Unit]",
            "Description=AudioCinema automated run",
            "After=default.target",
            "",
            "[Service]",
            "Type=oneshot",
            f"WorkingDirectory={APP_DIR}",
            f"ExecStart={py_path} {os.path.join(APP_DIR, 'src', 'main.py')} --autorun",
            "",
        ])
        timer_body = "\n".join([
            "[Unit]",
            "Description=Run AudioCinema automatic pipeline",
            "",
            "[Timer]",
            f"OnCalendar={days} *-*-* {hour}",
            "Persistent=true",
            "Unit=audiocinema.service",
            "",
            "[Install]",
            "WantedBy=timers.target",
            "",
        ])

        service_path = os.path.join(unit_dir, "audiocinema.service")
        timer_path = os.path.join(unit_dir, "audiocinema.timer")
        with open(service_path, "w", encoding="utf-8") as handle:
            handle.write(service_body)
        with open(timer_path, "w", encoding="utf-8") as handle:
            handle.write(timer_body)

        self._save_autorun_config(days, hour)

        try:
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "--user", "enable", "--now", "audiocinema.timer"], check=True)
        except Exception as exc:
            messagebox.showwarning("Automation", f"Schedule saved but systemd enable failed: {exc}")
            self.status_var.set("Automation schedule saved (systemd pending).")
            return

        self.status_var.set(f"Automation enabled: {days} {hour}")
        messagebox.showinfo("Automation", "Schedule applied and timer enabled.")

    def _on_tb_config_change(self, *_args) -> None:
        self._save_tb_config()

    def _band_results_to_payload(self, results: list[dict]) -> tuple[dict, dict, dict]:
        ref = {}
        cine = {}
        delta = {}
        for result in results:
            band = result.get("band")
            if band not in ("LFE", "LF", "MF", "HF"):
                continue
            ref_val = result.get("ref_db")
            cine_val = result.get("tst_db")
            delta_val = result.get("diff_db")
            ref[band] = float(ref_val) if ref_val is not None else 0.0
            cine[band] = float(cine_val) if cine_val is not None else 0.0
            delta[band] = float(delta_val) if delta_val is not None else 0.0
        return ref, cine, delta



    def _build_thingsboard_channel(
        self,
        score_text: str,
        status_text: str,
        band_results: list[dict],
    ) -> dict:
        ref, cine, delta = self._band_results_to_payload(band_results)
        return {
            "Score": score_text,
            "Status": status_text,
            "ref": ref,
            "cine": cine,
            "delta": delta,
        }






    def _send_thingsboard_telemetry(
        self,
        score_text: str,
        status_text: str,
        band_results: list[dict],
    ) -> None:
        
        telemetry = {
            "Canal1": self._build_thingsboard_channel(score_text, status_text, band_results)
        }
        self._send_thingsboard_payload(telemetry)

    def _send_thingsboard_payload(self, telemetry: dict) -> None:
        server = self.tb_server_var.get().strip()
        token = self.tb_token_var.get().strip()
        if not server or not token:
            self.status_var.set("ThingsBoard not configured")
            return
        try:
            port = int(self.tb_port_var.get().strip() or 1883)
        except ValueError:
            self.status_var.set("ThingsBoard port invalid")
            return

        def _send():
            try:
                send_to_thingsboard(server, port, token, telemetry)
            except Exception as exc:
                self.root.after(
                    0,
                    lambda err=exc: self.status_var.set(f"ThingsBoard send failed: {err}")
                )

        threading.Thread(target=_send, daemon=True).start()


    def send_thingsboard_now(self) -> None:
        if not self.last_tb_payload:
            messagebox.showwarning("ThingsBoard", "No evaluation data to send yet.")
            return
        self._send_thingsboard_payload(self.last_tb_payload)
        payload_text = json.dumps(self.last_tb_payload, indent=2)
        messagebox.showinfo("ThingsBoard payload", payload_text)





    def get_selected_delta_threshold_db(self) -> float:
        return get_delta_threshold_db_from_criteria(self.band_criteria_var.get())


    def get_selected_best_corr_threshold(self) -> float:
        return get_best_corr_threshold_from_criteria(self.band_criteria_var.get())



    def on_band_criteria_change(self, _event=None):
        criteria = self.band_criteria_var.get().strip()
        threshold_db = self.get_selected_delta_threshold_db()
        threshold_corr = self.get_selected_best_corr_threshold()
        msg = (
            f"Evaluation criteria set to {criteria} → Δ threshold={threshold_db:.1f} dB, "
            f"Best Corr threshold={threshold_corr:.2f}"
        )
        print(msg)
        if hasattr(self, "status_var"):
            self.status_var.set(msg)



    def _analysis_channel_count(self, analysis: dict | None) -> int:
        if analysis and isinstance(analysis, dict):
            count = analysis.get("channel_count")
            if isinstance(count, int) and count > 0:
                return count
            windows = analysis.get("windows_s")
            if isinstance(windows, list) and windows:
                return len(windows)
        return self.expected_channels

    def _analysis_channel_order(self, analysis: dict | None) -> list[str]:
        if analysis and isinstance(analysis, dict):
            order = analysis.get("channel_order")
            if isinstance(order, list) and order:
                return list(order)
        return list(self.channel_order)

    def _analysis_lfe_index(self, analysis: dict | None) -> int:
        order = self._analysis_channel_order(analysis)
        try:
            return order.index("LFE")
        except ValueError:
            return max(0, len(order) - 1)

    def _validate_analysis_layouts(self) -> tuple[int, list[str]] | None:
        if self.ref_analysis is None or self.tst_analysis is None:
            return None
        ref_count = self._analysis_channel_count(self.ref_analysis)
        tst_count = self._analysis_channel_count(self.tst_analysis)
        if ref_count != tst_count:
            messagebox.showerror("Error", "REF/TEST channel counts mismatch. Re-analyze both.")
            return None
        ref_order = self._analysis_channel_order(self.ref_analysis)
        tst_order = self._analysis_channel_order(self.tst_analysis)
        if ref_order != tst_order:
            messagebox.showerror("Error", "REF/TEST channel order mismatch. Re-analyze both.")
            return None
        return ref_count, ref_order



    def refresh_devices(self):
        devices = sd.query_devices()
        items = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                items.append(f"[{i}] {d['name']} (in:{d['max_input_channels']})")
        if not items:
            items = ["(No input devices found)"]

        current = self.device_var.get()
        self.device_combo["values"] = items
        if current in items:
            self.device_combo.set(current)
        else:
            self.device_combo.current(0)
            self.device_var.set(self.device_combo.get())

    def get_selected_device_index(self) -> int | None:
        text = self.device_var.get().strip()
        if not text.startswith("["):
            return None
        try:
            idx_str = text.split("]")[0][1:]
            return int(idx_str)
        except Exception:
            return None

    def get_params(self):
        try:
            fs = int(float(self.fs_var.get()))
            dur = float(self.dur_var.get())
            thr_match = float(self.thr_match_var.get())

            spacing_s = float(self.spacing_var.get())
            start_to_ch1_s = float(self.start_to_ch1_var.get())
            radius_s = float(self.radius_var.get())
            freq_tol = float(self.freq_tol_var.get())

            offset_s = float(self.offset_var.get())
            ch_len_s = float(self.chlen_var.get())
            guard_s = float(self.guard_var.get())

            f_start = float(self.f_start_var.get())
            f_ch_marker = float(self.f_ch_marker_var.get())
            f_end = float(self.f_end_var.get())

            if fs < 40000:
                raise ValueError("Use fs=48000 (min) to work reliably up to 20 kHz.")
            if dur < 10:
                raise ValueError(
                    f"Duration too short for {self.expected_channels} channels + end flag."
                )
            if thr_match <= 0:
                raise ValueError("REF vs TEST thr must be > 0.")
            if spacing_s <= 0:
                raise ValueError("Channel spacing must be > 0.")
            if start_to_ch1_s < 0:
                raise ValueError("Start -> Ch1 must be >= 0.")
            if radius_s <= 0:
                raise ValueError("Search radius must be > 0.")
            if freq_tol <= 0:
                raise ValueError("Freq tol must be > 0.")
            if ch_len_s <= 0:
                raise ValueError("Channel length must be > 0.")
            if guard_s < 0 or guard_s > 1.0:
                raise ValueError("Guard must be between 0 and 1s.")
            if offset_s < 0:
                raise ValueError("Offset must be >= 0.")

            for f in [f_start, f_ch_marker, f_end]:
                if f <= 0 or f >= fs / 2:
                    raise ValueError(f"Marker {f} Hz is invalid (Nyquist={fs/2:.0f}).")

            return (
                fs,
                dur,
                thr_match,
                spacing_s,
                start_to_ch1_s,
                radius_s,
                freq_tol,
                offset_s,
                ch_len_s,
                guard_s,
                f_start,
                f_ch_marker,
                f_end,
            )
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}")

    def get_filter_params(self):
        try:
            enabled = bool(self.filter_enabled_var.get())
            low = float(self.filter_low_var.get())
            high = float(self.filter_high_var.get())
            order = int(float(self.filter_order_var.get()))
            if low <= 0:
                raise ValueError("Filter low cutoff must be > 0.")
            if high <= low:
                raise ValueError("Filter high cutoff must be > low cutoff.")
            if order <= 0:
                raise ValueError("Filter order must be > 0.")
            return enabled, low, high, order
        except Exception as e:
            raise ValueError(f"Invalid filter parameters: {e}")



    def set_busy(self, busy: bool, msg: str):
        self.status_var.set(msg)
        state = "disabled" if busy else "normal"
        self.rec_noise_btn.config(state=state)
        self.rec_ref_btn.config(state=state)
        self.rec_tst_btn.config(state=state)
        self.refresh_btn.config(state=state)
        self.device_combo.config(state="disabled" if busy else "readonly")
        if not busy:
            self.update_action_states()
            self.tb_send_btn.config(state="normal" if self.last_tb_payload is not None else "disabled")


    def update_action_states(self):
        self.plot_ref_btn.config(state="normal" if self.x_ref is not None else "disabled")
        self.plot_tst_btn.config(state="normal" if self.x_tst is not None else "disabled")
        self.an_ref_btn.config(state="normal" if self.x_ref is not None else "disabled")
        self.an_tst_btn.config(state="normal" if self.x_tst is not None else "disabled")


        self.compare_trim_btn.config(
            state="normal" if (self.ref_analysis is not None and self.tst_analysis is not None) else "disabled"
        )

        self.compare_sweeps_btn.config(
            state="normal" if (self.ref_analysis is not None and self.tst_analysis is not None) else "disabled"
        )

        self.bands_btn.config(
            state="normal" if (self.ref_analysis is not None and self.tst_analysis is not None) else "disabled"
        )





        self.eval_btn.config(state="normal" if (self.ref_analysis is not None and self.tst_analysis is not None) else "disabled")

    def update_file_status(self):
        existing = []
        missing = []
        for key, path in self.wav_paths.items():
            name = os.path.basename(path)
            if os.path.exists(path):
                existing.append(name)
            else:
                missing.append(name)
        base_msg = f"WAVs: {', '.join(existing) if existing else 'none'}"
        if missing:
            base_msg = f"{base_msg} | Missing: {', '.join(missing)}"
        self.file_status_var.set(base_msg)

    def load_saved_wavs(self):
        loaded = []
        for key, path in self.wav_paths.items():
            if not os.path.exists(path):
                continue
            try:
                fs, x = load_wav_mono(path)
            except Exception as exc:
                print(f"[WARN] Failed to load {path}: {exc}")
                continue
            if key == "noise":
                self.x_noise = (fs, x)
                loaded.append("NOISE")
            elif key == "ref":
                self.x_ref = (fs, x)
                self.ref_analysis = None
                loaded.append("REF")
            elif key == "tst":
                self.x_tst = (fs, x)
                self.tst_analysis = None
                loaded.append("TEST")
        if loaded:
            self.status_var.set(f"Loaded saved WAVs: {', '.join(loaded)}.")
        self.update_action_states()
        self.update_file_status()

    def import_wav_files(self):
        filetypes = [("WAV files", "*.wav"), ("All files", "*.*")]
        selections = {
            "noise": ("Select NOISE WAV", None),
            "ref": ("Select REF WAV", None),
            "tst": ("Select TEST WAV", None),
        }
        for key in selections:
            title, _ = selections[key]
            path = filedialog.askopenfilename(title=title, filetypes=filetypes)
            if not path:
                continue
            try:
                fs, x = load_wav_mono(path)
                save_wav_mono(self.wav_paths[key], fs, x)
            except Exception as exc:
                messagebox.showerror("Import error", f"Failed to import {key.upper()}: {exc}")
                continue
            if key == "noise":
                self.x_noise = (fs, x)
            elif key == "ref":
                self.x_ref = (fs, x)
                self.ref_analysis = None
            else:
                self.x_tst = (fs, x)
                self.tst_analysis = None
        self.status_var.set("Import completed. WAVs saved in project folder.")
        self.update_action_states()
        self.update_file_status()



    def start_record(self, which: str):
        try:
            fs, dur, *_ = self.get_params()
            filter_enabled, filter_low, filter_high, filter_order = self.get_filter_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        dev = self.get_selected_device_index()
        if dev is None or "No input devices" in self.device_var.get():
            messagebox.showerror("Error", "Please select a valid input device.")
            return

        label = "REFERENCE" if which == "ref" else "TEST"
        self.set_busy(True, f"Recording {label}... ({dur:.1f}s)")

        def worker():
            try:
                x = record_audio(dur, fs, device=dev)
                if self.x_noise is not None:
                    noise_fs, noise = self.x_noise
                    if noise_fs == fs:
                        x = subtract_noise(x, noise)
                    else:
                        print(
                            f"[WARN] Noise fs mismatch (noise={noise_fs}, rec={fs}); skipping noise subtraction."
                        )
                if filter_enabled:
                    rms_before = rms_db_per_channel(x)
                    x = bandpass_filter(
                        x,
                        fs,
                        low_hz=filter_low,
                        high_hz=filter_high,
                        order=filter_order,
                        fade_ms=FILTER_FADE_MS,
                    )
                    rms_after = rms_db_per_channel(x)
                    msg = (
                        f"Filtering enabled: bandpass {filter_low:.0f}–{filter_high:.0f} Hz, "
                        f"order={filter_order}, fs={fs}. RMS dB: "
                        f"before={_format_rms_list(rms_before)}, after={_format_rms_list(rms_after)}"
                    )
                    print(msg)
                    if FILTER_DEBUG_SPECTRAL:
                        ratio = spectral_band_ratio_db(x, fs, filter_low, min(filter_high, fs / 2))
                        if ratio is not None:
                            print(f"Spectral ratio in/out band: {ratio:.1f} dB")
                else:
                    print(
                        f"Filtering disabled: raw audio (fs={fs}, "
                        f"low={filter_low:.0f} Hz, high={filter_high:.0f} Hz, order={filter_order})."
                    )
                self.q.put(("record_ok", which, fs, x))
            except Exception as ex:
                self.q.put(("record_err", str(ex)))

        threading.Thread(target=worker, daemon=True).start()

    def start_record_noise(self):
        try:
            fs, dur, *_ = self.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        dev = self.get_selected_device_index()
        if dev is None or "No input devices" in self.device_var.get():
            messagebox.showerror("Error", "Please select a valid input device.")
            return

        self.set_busy(True, f"Recording NOISE... ({dur:.1f}s)")

        def worker():
            try:
                x = record_audio(dur, fs, device=dev)
                self.q.put(("noise_ok", fs, x))
            except Exception as ex:
                self.q.put(("record_err", str(ex)))

        threading.Thread(target=worker, daemon=True).start()


    def start_analyze(self, which: str):
        if which == "ref":
            if self.x_ref is None:
                return
            fs, x = self.x_ref
            label = "REFERENCE"
        else:
            if self.x_tst is None:
                return
            fs, x = self.x_tst
            label = "TEST"

        try:
            (
                _,
                _,
                _,
                spacing_s,
                start_to_ch1_s,
                radius_s,
                freq_tol,
                offset_s,
                ch_len_s,
                guard_s,
                f_start,
                f_ch_marker,
                f_end,
            ) = self.get_params()
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        self.set_busy(True, f"Analyzing {label}: markers by expected times + targeted freq (±{freq_tol:.0f}Hz)...")

        def worker():
            try:
                out = analyze_track(
                    x, fs,
                    f_start=f_start,
                    f_ch_marker=f_ch_marker,
                    f_end=f_end,
                    channel_count=self.expected_channels,
                    channel_order=self.channel_order,
                    start_to_ch1_s=start_to_ch1_s,
                    spacing_s=spacing_s,
                    time_radius_s=radius_s,
                    offset_in_channel_s=offset_s,
                    channel_len_s=ch_len_s,
                    guard_s=guard_s,
                    freq_tol_hz=freq_tol,
                )
                self.q.put(("an_ok", which, out))
            except Exception as ex:
                self.q.put(("an_err", str(ex)))

        threading.Thread(target=worker, daemon=True).start()

    def plot_track(self, which: str):
        if which == "ref":
            if self.x_ref is None:
                return
            fs, x = self.x_ref
            an = self.ref_analysis
            title = "REFERENCE"
        else:
            if self.x_tst is None:
                return
            fs, x = self.x_tst
            an = self.tst_analysis
            title = "TEST"

        marker_lines = an.get("marker_lines_s", []) if an else []
        window_lines = an.get("window_lines_s", []) if an else []
        shaded = an.get("shaded_ranges_s", []) if an else None

        plot_wave_with_lines(
            x, fs,
            f"{title} waveform (orange=markers, green=window edges)",
            marker_lines_s=marker_lines,
            window_lines_s=window_lines,
            shaded_ranges_s=shaded
        )

    # NEW: plot Channel 1 stats (trimmed) using analysis windows_s[0]
    def plot_ch1_stats(self, which: str):
        if which == "ref":
            if self.x_ref is None or self.ref_analysis is None:
                messagebox.showwarning("Missing data", "Record + Analyze REF first.")
                return
            fs, x = self.x_ref
            an = self.ref_analysis
            title = "REF"
        else:
            if self.x_tst is None or self.tst_analysis is None:
                messagebox.showwarning("Missing data", "Record + Analyze TEST first.")
                return
            fs, x = self.x_tst
            an = self.tst_analysis
            title = "TEST"

        windows = an.get("windows_s", [])
        if not windows or len(windows) < 1:
            messagebox.showerror("Error", "No channel windows found. Run Analyze again.")
            return

        a, b = windows[0]  # Channel 1 window in seconds
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            messagebox.showerror("Error", "Channel 1 window is invalid. Check marker detection.")
            return

        # ✅ ESTE es el cálculo correcto de x_ch1: señal recortada SOLO del canal 1
        i0 = int(max(0, round(float(a) * fs)))
        i1 = int(min(len(x), round(float(b) * fs)))
        if i1 - i0 < int(0.05 * fs):
            messagebox.showerror("Error", "Channel 1 window too short.")
            return


        i0 = int(max(0, round(a * fs)))
        i1 = int(min(len(x), round(b * fs)))

        x_ch1 = np.asarray(x[i0:i1], dtype=np.float32)  # <- vector recortado del canal 1


        x_ch1_trim, cut0, cut1, dbg = trim_sweep_by_absstd(
            x_ch1, fs,
            sweep_s=10.0,
            win_s=0.20,
            hop_s=0.05,
            silence_hi=0.0005,     # TU rango de silencio
            active_mult=1.5,       # si corta tarde, baja a 1.5
            smooth_frames=9
        )

        print("DBG TRIM:", dbg)




        try:
            plot_channel1_mean_std(
                x, fs,
                ch1_start_s=float(a),
                ch1_end_s=float(b),
                title=f"{title} — Channel 1 (trimmed) + mean ± std"
            )

            # ✅ Gráfica nueva: mean vs time usando el RECORTE del canal 1
            plot_mean_vs_time_with_std_and_absstd(
                x_ch1, fs,
                title=f"{title} — Channel 1 (trimmed) — Mean vs Time (±1σ) + abs(std)",
                win_s=0.20,
                hop_s=0.05,
                show_band=True,
                use_twin_axis=True
            )

            plot_mean_vs_time(
                x_ch1_trim, fs,
                title=f"{title} — Channel 1 (TRIM by abs(std) silence) — Mean vs Time (+1σ) + abs(std)",
                win_s=0.20,
                hop_s=0.05,
                auto_trim=False
            )

        except Exception as e:
            messagebox.showerror("Plot error", str(e))



    #------------------------------------------NUEVO-----------------------------para todos los canales
    def plot_all_channels_stats(self, which: str):
        if which == "ref":
            if self.x_ref is None or self.ref_analysis is None:
                messagebox.showwarning("Missing data", "Record + Analyze REF first.")
                return
            fs, x = self.x_ref
            an = self.ref_analysis
            prefix = "REF"
        else:
            if self.x_tst is None or self.tst_analysis is None:
                messagebox.showwarning("Missing data", "Record + Analyze TEST first.")
                return
            fs, x = self.x_tst
            an = self.tst_analysis
            prefix = "TEST"

        windows = an.get("windows_s", [])
        channel_count = self._analysis_channel_count(an)
        if not windows or len(windows) < channel_count:
            messagebox.showerror("Error", "No valid channel windows found. Run Analyze again.")
            return

        for ch in range(channel_count):
            a, b = windows[ch]
            if (not np.isfinite(a)) or (not np.isfinite(b)) or (b <= a):
                print(f"[WARN] {prefix} Ch{ch+1}: invalid window -> {a}, {b}")
                continue

            i0 = int(max(0, round(a * fs)))
            i1 = int(min(len(x), round(b * fs)))
            if i1 - i0 < int(0.05 * fs):
                print(f"[WARN] {prefix} Ch{ch+1}: too short -> {i1-i0} samples")
                continue

            x_seg = np.asarray(x[i0:i1], dtype=np.float32)

            x_trim, dbg = get_trimmed_sweep_from_window(
                x, fs, windows[ch],
                silence_hi=0.0005,
                active_mult=2.0,
                sweep_s=10.0
            )

            plot_mean_vs_time_with_std_and_absstd(
                x_seg, fs,
                title=f"{prefix} — Channel {ch+1} (trimmed) — Mean vs Time (±1σ) + abs(std)",
                win_s=0.20,
                hop_s=0.05,
                show_band=True,
                use_twin_axis=True
            )




#----------------------------------------------------para BOTON Compare Sweeps Window
    def _build_channel_stats_tree(
        self,
        parent: tk.Widget,
        title: str,
        x: np.ndarray,
        fs: int,
        analysis: dict,
    ):
        ttk.Label(parent, text=title).pack(anchor="w")

        cols = ("ch", "win_start", "win_end", "trim_start", "trim_end", "trim_len", "rms_db", "notes")
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=10)
        headers = [
            "Ch",
            "Win start [s]",
            "Win end [s]",
            "Trim start [s]",
            "Trim end [s]",
            "Trim len [s]",
            "RMS [dB]",
            "Notes",
        ]
        for c, h in zip(cols, headers):
            tree.heading(c, text=h)

        tree.column("ch", width=90, anchor="center")
        tree.column("win_start", width=110, anchor="e")
        tree.column("win_end", width=110, anchor="e")
        tree.column("trim_start", width=110, anchor="e")
        tree.column("trim_end", width=110, anchor="e")
        tree.column("trim_len", width=110, anchor="e")
        tree.column("rms_db", width=90, anchor="e")
        tree.column("notes", width=200, anchor="w")

        yscroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="left", fill="y")

        windows = analysis.get("windows_s", [])
        channel_count = self._analysis_channel_count(analysis)
        for ch in range(channel_count):
            if ch >= len(windows):
                tree.insert("", "end", values=(str(ch + 1), "", "", "", "", "", "", "missing window"))
                continue

            a, b = windows[ch]
            if (not np.isfinite(a)) or (not np.isfinite(b)) or (b <= a):
                tree.insert("", "end", values=(str(ch + 1), "", "", "", "", "", "", "invalid window"))
                continue

            x_trim, dbg = get_trimmed_sweep_from_window(
                x, fs, windows[ch],
                silence_hi=0.0005,
                active_mult=2.0,
                sweep_s=10.0
            )

            trim_i0 = dbg.get("cut_i0_in_channel")
            trim_i1 = dbg.get("cut_i1_in_channel")
            trim_start_s = (float(trim_i0) / fs) if trim_i0 is not None else None
            trim_end_s = (float(trim_i1) / fs) if trim_i1 is not None else None
            trim_len_s = (float(trim_i1 - trim_i0) / fs) if (trim_i0 is not None and trim_i1 is not None) else None

            note = dbg.get("reason", "")
            rms_value = rms_db(x_trim) if x_trim.size else None

            tree.insert(
                "", "end",
                values=(
                    str(ch + 1),
                    f"{a:.3f}",
                    f"{b:.3f}",
                    "" if trim_start_s is None else f"{trim_start_s:.3f}",
                    "" if trim_end_s is None else f"{trim_end_s:.3f}",
                    "" if trim_len_s is None else f"{trim_len_s:.3f}",
                    "" if rms_value is None else f"{rms_value:.1f}",
                    note,
                ),
            )

    def show_compare_sweeps_window(self):
        if self.x_ref is None or self.ref_analysis is None or self.x_tst is None or self.tst_analysis is None:
            messagebox.showwarning("Missing data", "Record + Analyze REF and TEST first.")
            return

        fs_r, x_ref = self.x_ref
        fs_t, x_tst = self.x_tst
        if fs_r != fs_t:
            messagebox.showerror("Error", f"FS mismatch: REF={fs_r}, TEST={fs_t}")
            return

        w = tk.Toplevel(self.root)
        w.title("Compare sweeps — REF vs TEST")

        outer = ttk.Frame(w, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        left = ttk.Frame(outer)
        right = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._build_channel_stats_tree(left, "REF stats", x_ref, fs_r, self.ref_analysis)
        self._build_channel_stats_tree(right, "TEST stats", x_tst, fs_t, self.tst_analysis)



    def _compute_band_results(
        self,
        ch_idx: int,
        fs: int,
        x_ref: np.ndarray,
        x_tst: np.ndarray,
        win_ref: list,
        win_tst: list,
        subwoofer_index: int,
    ) -> tuple[list[dict], dict]:
        x_ref_trim, _ = get_trimmed_sweep_from_window(x_ref, fs, win_ref[ch_idx])
        x_tst_trim, _ = get_trimmed_sweep_from_window(x_tst, fs, win_tst[ch_idx])

        if ch_idx == subwoofer_index:
            fmin = 20.0
            fmax = min(200.0, fs / 2 - 50)
            win_s = 0.20
            hop_s = 0.06
            gate_db_below_global = 28.0
        else:
            fmin = 20.0
            fmax = min(20000.0, fs / 2 - 50)
            win_s = 0.06
            hop_s = 0.02
            gate_db_below_global = 35.0

        tR, fR = sweep_freq_track(
            x_ref_trim,
            fs,
            fmin=fmin,
            fmax=fmax,
            win_s=win_s,
            hop_s=hop_s,
            gate_db_below_global=gate_db_below_global,
        )
        tT, fT = sweep_freq_track(
            x_tst_trim,
            fs,
            fmin=fmin,
            fmax=fmax,
            win_s=win_s,
            hop_s=hop_s,
            gate_db_below_global=gate_db_below_global,
        )

        _, _, dbg_lag = best_xcorr_shift_on_freq_curve(
            tR,
            fR,
            tT,
            fT,
            dt=None,
            max_lag_s=3.0,
            min_valid_frac=0.70,
        )
        lag_s = float(dbg_lag.get("best_lag_s", 0.0))
        if "reason" not in dbg_lag:
            x_ref_trim, x_tst_trim = align_by_lag_s(x_ref_trim, x_tst_trim, fs, lag_s)

        freqs_ref, mag_ref = compute_spectrum(x_ref_trim, fs)
        freqs_tst, mag_tst = compute_spectrum(x_tst_trim, fs)
        nyquist = fs / 2.0
        band_tol_db = self.get_selected_delta_threshold_db()

        results = []
        for band_name, (f_lo, f_hi) in BAND_RANGES_HZ.items():
            f_hi_cap = min(f_hi, nyquist)
            if f_hi_cap <= f_lo:
                continue

            is_subwoofer = ch_idx == subwoofer_index
            is_lfe_band = band_name == "LFE"
            is_sub_bands = band_name in ("LF", "MF", "HF")
            if (is_lfe_band and not is_subwoofer) or (is_subwoofer and is_sub_bands):
                results.append(
                    {
                        "band": band_name,
                        "range": (f_lo, f_hi_cap),
                        "ref_db": 0.0,
                        "tst_db": 0.0,
                        "diff_db": 0.0,
                        "status": "OK",
                    }
                )
                continue

            ref_db = band_power_db(freqs_ref, mag_ref, f_lo, f_hi_cap)
            tst_db = band_power_db(freqs_tst, mag_tst, f_lo, f_hi_cap)
            diff_db = None
            status = "N/A"
            if ref_db is not None and tst_db is not None:
                diff_db = abs(ref_db - tst_db)
                status = "OK" if diff_db <= band_tol_db else "FAIL"

            results.append(
                {
                    "band": band_name,
                    "range": (f_lo, f_hi_cap),
                    "ref_db": ref_db,
                    "tst_db": tst_db,
                    "diff_db": diff_db,
                    "status": status,
                }
            )

        return results, {"lag_s": lag_s, "lag_reason": dbg_lag.get("reason")}







    def show_bands_window(self):
        if self.ref_analysis is None or self.tst_analysis is None or self.x_ref is None or self.x_tst is None:
            messagebox.showwarning("Missing data", "Record + Analyze REF and TEST first.")
            return

        fs_r, x_ref = self.x_ref
        fs_t, x_tst = self.x_tst
        if fs_r != fs_t:
            messagebox.showerror("Error", f"FS mismatch: REF={fs_r}, TEST={fs_t}")
            return
        fs = fs_r

        layout_info = self._validate_analysis_layouts()
        if layout_info is None:
            return
        channel_count, channel_order = layout_info
        subwoofer_index = max(0, channel_count - 1)

        w = tk.Toplevel(self.root)
        w.title("Bands — REF vs TEST")
        w.geometry("900x400")
        w.minsize(400, 100)

        scroll_container = ttk.Frame(w)
        scroll_container.pack(fill="both", expand=True)

        scroll_canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scroll_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=scroll_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        content = ttk.Frame(scroll_canvas)
        content_window = scroll_canvas.create_window((0, 0), window=content, anchor="nw")

        def _on_content_configure(_event):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            scroll_canvas.itemconfigure(content_window, width=event.width)

        content.bind("<Configure>", _on_content_configure)
        scroll_canvas.bind("<Configure>", _on_canvas_configure)

        top = ttk.Frame(content, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Channel:").pack(side="left")
        channel_var = tk.StringVar(value="1")
        channel_select = ttk.Combobox(top, textvariable=channel_var, state="readonly", width=6)
        channel_select["values"] = [str(i) for i in range(1, channel_count + 1)]
        channel_select.pack(side="left", padx=(6, 18))

        status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=status_var).pack(side="left")


        table_frame = ttk.Frame(content, padding=(10, 0, 10, 10))
        table_frame.pack(fill="x")

        cols = ("band", "range", "ref_db", "test_db", "diff_db", "status")
        tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=6)
        tree.heading("band", text="Band")
        tree.heading("range", text="Range [Hz]")
        tree.heading("ref_db", text="REF [dB]")
        tree.heading("test_db", text="TEST [dB]")
        tree.heading("diff_db", text="Δ [dB]")
        tree.heading("status", text="Match")
        tree.column("band", width=70, anchor="center")
        tree.column("range", width=120, anchor="center")
        tree.column("ref_db", width=120, anchor="e")
        tree.column("test_db", width=120, anchor="e")
        tree.column("diff_db", width=90, anchor="e")
        tree.column("status", width=90, anchor="center")
        tree.pack(fill="x")


        fig = Figure(figsize=(10.5, 6.5))
        ax_ref = fig.add_subplot(2, 1, 1)
        ax_tst = fig.add_subplot(2, 1, 2, sharex=ax_ref)

        plot_canvas = FigureCanvasTkAgg(fig, master=content)
        plot_canvas_widget = plot_canvas.get_tk_widget()
        plot_canvas_widget.pack(fill="both", expand=True, padx=10, pady=(0, 8))


        def format_db(value: float | None) -> str:
            return "" if value is None else f"{value:.2f}"

        def render():
            try:
                ch_idx = int(channel_var.get()) - 1
            except ValueError:
                ch_idx = 0

            if ch_idx < 0 or ch_idx >= channel_count:
                status_var.set("Invalid channel.")
                return

            win_ref = self.ref_analysis.get("windows_s", [])
            win_tst = self.tst_analysis.get("windows_s", [])
            if ch_idx >= len(win_ref) or ch_idx >= len(win_tst):
                status_var.set("Missing channel windows.")
                return



            results, lag_info = self._compute_band_results(
                ch_idx,
                fs,
                x_ref,
                x_tst,
                win_ref,
                win_tst,
                subwoofer_index,
            )

            nyquist = fs / 2.0

            x_ref_trim, _ = get_trimmed_sweep_from_window(x_ref, fs, win_ref[ch_idx])
            x_tst_trim, _ = get_trimmed_sweep_from_window(x_tst, fs, win_tst[ch_idx])
            if lag_info.get("lag_reason") is None:
                x_ref_trim, x_tst_trim = align_by_lag_s(
                    x_ref_trim,
                    x_tst_trim,
                    fs,
                    float(lag_info.get("lag_s", 0.0)),
                )




            freqs_ref, mag_ref = compute_spectrum(x_ref_trim, fs)
            freqs_tst, mag_tst = compute_spectrum(x_tst_trim, fs)

            ax_ref.clear()
            ax_tst.clear()

            if freqs_ref.size > 0:
                mag_ref_db = 20.0 * np.log10(mag_ref + 1e-12)
                ax_ref.plot(freqs_ref, mag_ref_db, linewidth=0.9)
            if freqs_tst.size > 0:
                mag_tst_db = 20.0 * np.log10(mag_tst + 1e-12)
                ax_tst.plot(freqs_tst, mag_tst_db, linewidth=0.9)

            channel_label = channel_order[ch_idx] if ch_idx < len(channel_order) else f"Ch{ch_idx + 1}"
            ax_ref.set_title(f"Reference — Ch{ch_idx + 1} ({channel_label})")
            ax_tst.set_title(f"Test — Ch{ch_idx + 1} ({channel_label})")
            ax_tst.set_xlabel("Frequency [Hz]")
            ax_ref.set_ylabel("Magnitude [dB]")
            ax_tst.set_ylabel("Magnitude [dB]")
            ax_ref.grid(True, linestyle=":")
            ax_tst.grid(True, linestyle=":")
            ax_ref.set_xlim(0.0, nyquist)

            for child in tree.get_children():
                tree.delete(child)

            for result in results:
                f_lo, f_hi_cap = result["range"]
                ax_ref.axvspan(f_lo, f_hi_cap, color="tab:blue", alpha=0.08)
                ax_tst.axvspan(f_lo, f_hi_cap, color="tab:blue", alpha=0.08)




                tree.insert(
                    "",
                    "end",
                    values=(
                        result["band"],
                        f"{f_lo:.0f}-{f_hi_cap:.0f}",
                        format_db(result["ref_db"]),
                        format_db(result["tst_db"]),
                        format_db(result["diff_db"]),
                        result["status"],
                    ),
                )

            plot_canvas.draw_idle()
            if lag_info.get("lag_reason") is not None:
                status_var.set(f"Alignment skipped: {lag_info['lag_reason']}")
            else:
                band_tol_db = self.get_selected_delta_threshold_db()
                status_var.set(
                    f"Band tolerance: ±{band_tol_db:.1f} dB | best lag: {lag_info.get('lag_s', 0.0):.3f}s"
                )

        channel_select.bind("<<ComboboxSelected>>", lambda _event: render())
        render()




    def poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if not msg:
                    continue

                if msg[0] == "record_ok":
                    _, which, fs, x = msg
                    if which == "ref":
                        self.x_ref = (fs, x)
                        self.ref_analysis = None
                        try:
                            save_wav_mono(self.wav_paths["ref"], fs, x)
                        except Exception as exc:
                            messagebox.showwarning("Save warning", f"REF save failed: {exc}")
                        self.status_var.set("✅ Reference recorded.")
                    else:
                        self.x_tst = (fs, x)
                        self.tst_analysis = None
                        try:
                            save_wav_mono(self.wav_paths["tst"], fs, x)
                        except Exception as exc:
                            messagebox.showwarning("Save warning", f"TEST save failed: {exc}")
                        self.status_var.set("✅ Test recorded.")
                    self.update_file_status()
                    self.set_busy(False, self.status_var.get())
                elif msg[0] == "noise_ok":
                    _, fs, x = msg
                    self.x_noise = (fs, x)
                    try:
                        save_wav_mono(self.wav_paths["noise"], fs, x)
                    except Exception as exc:
                        messagebox.showwarning("Save warning", f"NOISE save failed: {exc}")
                    self.status_var.set("✅ Noise recorded.")
                    self.update_file_status()
                    self.set_busy(False, self.status_var.get())

                elif msg[0] == "record_err":
                    _, err = msg
                    self.set_busy(False, "Ready.")
                    messagebox.showerror("Recording error", err)

                elif msg[0] == "an_ok":
                    _, which, out = msg
                    if which == "ref":
                        self.ref_analysis = out
                        self.status_var.set("✅ REF analyzed.")
                        self.show_analysis_window("REFERENCE", self.x_ref[0], out)
                    else:
                        self.tst_analysis = out
                        self.status_var.set("✅ TEST analyzed.")
                        self.show_analysis_window("TEST", self.x_tst[0], out)

                    self.set_busy(False, self.status_var.get())

                elif msg[0] == "an_err":
                    _, err = msg
                    self.set_busy(False, "Ready.")
                    messagebox.showerror("Analyze error", err)

        except queue.Empty:
            pass

        self.root.after(100, self.poll_queue)

    def show_analysis_window(self, title: str, fs: int, out: dict):
        w = tk.Toplevel(self.root)
        w.title(f"{title} — Markers + Channel Freqs (targeted)")


        markers = out.get("markers", {})
        start_time = markers.get("start_time", None)
        channel_count = self._analysis_channel_count(out)
        channel_order = self._analysis_channel_order(out)
        ch_times = markers.get("ch_times", [None] * channel_count)
        end_time = markers.get("end_time", None)

        windows = out.get("windows_s", [])
        freqs = out.get("channel_freqs_hz", [])
        targets = out.get("targets_hz", [])

        top = ttk.Frame(w, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=f"{title} | fs={fs} Hz").pack(side="left")

        mid = ttk.Frame(w, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)

        cols = ("ch", "marker_t", "win_a", "win_b", "target", "freq")
        tree = ttk.Treeview(mid, columns=cols, show="headings", height=10)
        tree.heading("ch", text="Ch")
        tree.heading("marker_t", text="Marker t [s]")
        tree.heading("win_a", text="Win start [s]")
        tree.heading("win_b", text="Win end [s]")
        tree.heading("target", text="Target [Hz]")
        tree.heading("freq", text="Found [Hz] (closest in ±tol)")

        tree.column("ch", width=90, anchor="center")
        tree.column("marker_t", width=110, anchor="e")
        tree.column("win_a", width=110, anchor="e")
        tree.column("win_b", width=110, anchor="e")
        tree.column("target", width=110, anchor="e")
        tree.column("freq", width=180, anchor="e")

        yscroll = ttk.Scrollbar(mid, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)

        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        for i in range(channel_count):
            mt = ch_times[i]
            a, b = windows[i] if i < len(windows) else (np.nan, np.nan)
            tf = targets[i] if i < len(targets) else np.nan
            ff = freqs[i] if i < len(freqs) else None

            channel_label = channel_order[i] if i < len(channel_order) else f"Ch{i + 1}"
            mt_txt = "" if mt is None else f"{mt:.3f}"
            a_txt = "" if not np.isfinite(a) else f"{a:.3f}"
            b_txt = "" if not np.isfinite(b) else f"{b:.3f}"
            tf_txt = "" if not np.isfinite(tf) else f"{tf:.1f}"
            ff_txt = "" if ff is None else f"{float(ff):.3f}"
            tree.insert("", "end", values=(f"{i + 1} ({channel_label})", mt_txt, a_txt, b_txt, tf_txt, ff_txt))


        if start_time is not None:
            ttk.Label(w, text=f"Start marker time: {start_time:.3f}s").pack(anchor="w", padx=10, pady=(0, 2))
        if end_time is not None:
            ttk.Label(w, text=f"End marker time: {end_time:.3f}s").pack(anchor="w", padx=10, pady=(0, 10))


    def evaluate_channels(self):
        if self.ref_analysis is None or self.tst_analysis is None or self.x_ref is None or self.x_tst is None:
            return

        fs_r, x_ref = self.x_ref
        fs_t, x_tst = self.x_tst
        if fs_r != fs_t:
            messagebox.showerror("Error", f"FS mismatch: REF={fs_r}, TEST={fs_t}")
            return
        fs = fs_r


        layout_info = self._validate_analysis_layouts()
        if layout_info is None:
            return
        channel_count, channel_order = layout_info
        subwoofer_index = max(0, channel_count - 1)

        win_ref = self.ref_analysis.get("windows_s", [])
        win_tst = self.tst_analysis.get("windows_s", [])
        if len(win_ref) < channel_count or len(win_tst) < channel_count:
            messagebox.showerror("Error", "Missing channel windows. Run Analyze again.")
            return

        # Ajustables (desde GUI si quieres luego)
        SILENCE_HI  = 0.0005
        ACTIVE_MULT = 2.0
        SWEEP_S     = 10.0

        # Comparación de barrido
        TOL_HZ = float(self.freq_tol_var.get())     # ya lo tienes en GUI (±100 típico)
        RANGE_TOL = 200.0
        MIN_VALID = 0.75
        THR_CORR = self.get_selected_best_corr_threshold()

        rows = []
        channel_payloads = {}
        any_failed = False
        best_corr_by_label = {}






        for ch in range(channel_count):
            ref_trim, _ = get_trimmed_sweep_from_window(
                x_ref, fs, win_ref[ch],
                silence_hi=SILENCE_HI,
                active_mult=ACTIVE_MULT,
                sweep_s=SWEEP_S,
                win_stat_s=0.20,
                hop_stat_s=0.05
            )

            tst_trim, _ = get_trimmed_sweep_from_window(
                x_tst, fs, win_tst[ch],
                silence_hi=SILENCE_HI,
                active_mult=ACTIVE_MULT,
                sweep_s=SWEEP_S,
                win_stat_s=0.20,
                hop_stat_s=0.05
            )









            # Track de frecuencia vs tiempo (ajuste especial para subwoofer en LFE)
            if ch == subwoofer_index:
                fmin = 20.0
                fmax = min(200.0, fs / 2 - 50)
                win_s = 0.20
                hop_s = 0.06
                gate_db_below_global = 28.0
            else:
                fmin = 20.0
                fmax = min(20000.0, fs / 2 - 50)
                win_s = 0.06
                hop_s = 0.02
                gate_db_below_global = 35.0

            tR, fR = sweep_freq_track(
                ref_trim,
                fs,
                fmin=fmin,
                fmax=fmax,
                win_s=win_s,
                hop_s=hop_s,
                gate_db_below_global=gate_db_below_global,
            )
            tT, fT = sweep_freq_track(
                tst_trim,
                fs,
                fmin=fmin,
                fmax=fmax,
                win_s=win_s,
                hop_s=hop_s,
                gate_db_below_global=gate_db_below_global,
            )







            _, met = compare_sweep_freq_curves_xcorr(
                tR, fR, tT, fT,
                tol_hz=TOL_HZ,
                max_lag_s=3,        # <-- AJUSTA: cuánto desfase máximo permites entre REF y TEST
                min_valid_frac=MIN_VALID,
                pctl_err=90.0,
                range_tol_hz=RANGE_TOL
            )




            corr = None
            if ref_trim.size >= int(0.5 * fs) and tst_trim.size >= int(0.5 * fs):
                a = normalize_for_compare(ref_trim)
                b = normalize_for_compare(tst_trim)
                n = min(a.size, b.size)
                a = a[:n]
                b = b[:n]
                denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)
                corr = float(np.dot(a, b) / denom)


            best_corr = met.get("best_corr", None)
            is_alive = (best_corr is not None) and np.isfinite(best_corr) and (best_corr >= THR_CORR)
            status = "LIVE" if is_alive else "DEAD"


            results, _ = self._compute_band_results(
                ch,
                fs,
                x_ref,
                x_tst,
                win_ref,
                win_tst,
                subwoofer_index,
            )
            band_ok = all(result["status"] == "OK" for result in results)


            evaluacion = "PASSED" if band_ok else "FAILED"
            if not band_ok:
                any_failed = True

            # Guarda métricas clave para tabla

            label = channel_order[ch] if ch < len(channel_order) else f"Ch{ch + 1}"
            best_corr_by_label[label] = met.get("best_corr", None)
            rows.append((
                ch + 1,
                label,
                status,
                met.get("best_corr", None),
                evaluacion,
            ))
            channel_key = f"Canal{ch + 1}"
            channel_payloads[channel_key] = self._build_thingsboard_channel(
                score_text=evaluacion,
                status_text=status,
                band_results=results,
            )



        left_best_corr = best_corr_by_label.get("FL", best_corr_by_label.get("L"))
        right_best_corr = best_corr_by_label.get("FR", best_corr_by_label.get("R"))
        if left_best_corr is None or right_best_corr is None:
            balance_db_text = "Balanced R & L: N/A"
        else:
            balance_db = abs(best_corr_to_db(left_best_corr) - best_corr_to_db(right_best_corr))
            balance_db_text = f"Balanced R & L: {balance_db:.2f} dB"


        # UI tabla
        w = tk.Toplevel(self.root)
        w.title("Evaluation (REF vs TEST) — Sweep frequency similarity per channel")

        top = ttk.Frame(w, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=f"PASS if cross-corr >= {THR_CORR:.2f}").pack(side="left")
        ttk.Label(top, text=balance_db_text).pack(side="left", padx=(16, 0))
        ttk.Label(top, text=("❌ SOME FAILED" if any_failed else "✅ ALL PASSED")).pack(side="right")

        mid = ttk.Frame(w, padding=(10, 0, 10, 10))
        mid.pack(fill="both", expand=True)


        cols = ("ch", "label", "status", "best_corr", "score")
        tree = ttk.Treeview(mid, columns=cols, show="headings", height=10)
        for c, h in zip(
            cols,
            [
                "Ch",
                "Label",
                "Status",
                "BEST_CORR",
                "Score",
            ],
        ):
            tree.heading(c, text=h)


        tree.column("ch", width=90, anchor="center")
        tree.column("label", width=70, anchor="center")
        tree.column("status", width=90, anchor="center")
        tree.column("best_corr", width=110, anchor="e")
        tree.column("score", width=120, anchor="center")

        yscroll = ttk.Scrollbar(mid, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=yscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)

        for r in rows:
            if len(r) < 5:
                r = tuple(r) + (None,) * (5 - len(r))
            ch, label, status, best_corr, evaluacion = r[:5]
            tree.insert(
                "", "end",
                values=(
                    str(ch),
                    label,
                    status,
                    "" if best_corr is None else f"{best_corr:.3f}",
                    evaluacion,
                )
            )

        if channel_payloads:
            self.last_tb_payload = channel_payloads
            self.tb_send_btn.config(state="normal")




    def show_trimmed_sweeps_window(self):
        if self.ref_analysis is None or self.tst_analysis is None or self.x_ref is None or self.x_tst is None:
            messagebox.showwarning("Missing data", "Record + Analyze REF and TEST first.")
            return

        # --- Parámetros de recorte por abs(std)
        SILENCE_HI  = 0.0005   # silencio ~ 0.0000–0.0005 (ajústalo si cambias mic/ganancia)
        ACTIVE_MULT = 2.0      # más bajo = detecta antes (1.5–2.5 típico)
        SWEEP_S     = 10.0

        # --- Criterio de PASS/FAIL (similitud sweep vs sweep)
        THR_CORR = 0.10

        fs_r, x_ref = self.x_ref
        fs_t, x_tst = self.x_tst
        if fs_r != fs_t:
            messagebox.showerror("Error", f"FS mismatch: REF={fs_r}, TEST={fs_t}")
            return
        fs = fs_r

        layout_info = self._validate_analysis_layouts()
        if layout_info is None:
            return
        channel_count, channel_order = layout_info

        win_ref = self.ref_analysis.get("windows_s", [])
        win_tst = self.tst_analysis.get("windows_s", [])
        if len(win_ref) < channel_count or len(win_tst) < channel_count:
            messagebox.showerror("Error", "Missing channel windows. Run Analyze again.")
            return

        # --- Crear ventana
        w = tk.Toplevel(self.root)
        w.title(f"TRIM sweeps (abs(std)) — REF vs TEST ({channel_count} channels)")

        header = ttk.Frame(w, padding=10)
        header.pack(fill="x")
        info = ttk.Label(
            header,
            text=f"Trim: abs(std) | SILENCE<= {SILENCE_HI:g} | active_mult={ACTIVE_MULT:.2f} | sweep={SWEEP_S:.1f}s | PASS if corr >= {THR_CORR:.2f}"
        )
        info.pack(side="left")

        # --- Figura: N filas x 2 columnas
        fig_height = max(12, channel_count * 2.6)
        fig = Figure(figsize=(14, fig_height), dpi=100)
        axes = []
        for r in range(channel_count):
            axL = fig.add_subplot(channel_count, 2, 2*r + 1)
            axR = fig.add_subplot(channel_count, 2, 2*r + 2, sharex=axL, sharey=axL)
            axes.append((axL, axR))

        # --- Procesar canales y dibujar
        any_failed = False
        for ch in range(channel_count):
            # Recorte REF
            ref_trim, dbg_r = get_trimmed_sweep_from_window(
                x_ref, fs, win_ref[ch],
                silence_hi=SILENCE_HI,
                active_mult=ACTIVE_MULT,
                sweep_s=SWEEP_S,
                win_stat_s=0.20,
                hop_stat_s=0.05
            )

            # Recorte TEST
            tst_trim, dbg_t = get_trimmed_sweep_from_window(
                x_tst, fs, win_tst[ch],
                silence_hi=SILENCE_HI,
                active_mult=ACTIVE_MULT,
                sweep_s=SWEEP_S,
                win_stat_s=0.20,
                hop_stat_s=0.05
            )

            # Correlación (si hay data suficiente)
            corr = None
            status = "FAILED"
            if ref_trim.size >= int(0.5 * fs) and tst_trim.size >= int(0.5 * fs):
                a = normalize_for_compare(ref_trim)
                b = normalize_for_compare(tst_trim)
                n = min(a.size, b.size)
                a = a[:n]
                b = b[:n]
                denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-20)
                corr = float(np.dot(a, b) / denom)
                status = "PASSED" if corr >= THR_CORR else "FAILED"
            else:
                status = "FAILED"

            if status == "FAILED":
                any_failed = True

            axL, axR = axes[ch]

            # Tiempo
            t_ref = np.arange(ref_trim.size) / fs if ref_trim.size else np.array([0.0])
            t_tst = np.arange(tst_trim.size) / fs if tst_trim.size else np.array([0.0])

            # Plot REF
            if ref_trim.size:
                axL.plot(t_ref, ref_trim, linewidth=0.7)
            channel_label = channel_order[ch] if ch < len(channel_order) else f"Ch{ch + 1}"
            axL.set_title(f"REF — Ch{ch+1} ({channel_label}) (TRIM)")

            # Plot TEST
            if tst_trim.size:
                axR.plot(t_tst, tst_trim, linewidth=0.7)
            corr_txt = "N/A" if corr is None else f"{corr:.4f}"
            axR.set_title(f"TEST — Ch{ch+1} ({channel_label}) | corr={corr_txt} | {status}")

            # Labels / grid
            axL.grid(True, linestyle=":")
            axR.grid(True, linestyle=":")

            if ch == channel_count - 1:
                axL.set_xlabel("Time [s]")
                axR.set_xlabel("Time [s]")
            axL.set_ylabel("Amp")

        fig.tight_layout()

        # --- Mostrar badge de resultado arriba
        result_txt = "✅ ALL PASSED" if not any_failed else "❌ SOME FAILED"
        result_lbl = ttk.Label(header, text=result_txt)
        result_lbl.pack(side="right")

        # --- Embebido en Tk
        canvas = FigureCanvasTkAgg(fig, master=w)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        # Toolbar (zoom/pan)
        toolbar = NavigationToolbar2Tk(canvas, w)
        toolbar.update()




def main():
    ensure_dirs()
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    AudioCompareGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()



