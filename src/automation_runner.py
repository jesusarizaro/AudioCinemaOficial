from __future__ import annotations

import json
from pathlib import Path

from app_platform import APP_DIR, CFG_DIR, DATA_DIR
from gui_app import (
    CRITERIA_THRESHOLDS,
    DEFAULT_BAND_CRITERIA,
    DEFAULT_WAV_PATHS,
    SYSTEM_LAYOUTS,
    analyze_track,
    load_wav_mono,
    record_audio,
    save_wav_mono,
    send_to_thingsboard,
)


DEFAULT_AUTOMATION = {
    "enabled": False,
    "days": "Mon,Tue,Wed,Thu,Fri",
    "time": "08:00",
}


def _load_yaml_config() -> dict:
    cfg_path = CFG_DIR / "config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_tb_config() -> dict:
    tb_path = CFG_DIR / "tb_config.json"
    if not tb_path.exists():
        return {}
    try:
        data = json.loads(tb_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _analyze(x: list, fs: int, layout: str) -> dict:
    sys_cfg = SYSTEM_LAYOUTS.get(layout, SYSTEM_LAYOUTS["5.1"])
    return analyze_track(
        x=x,
        fs=fs,
        f_start=3000.0,
        f_ch_marker=4500.0,
        f_end=3000.0,
        channel_count=sys_cfg["channels"],
        channel_order=sys_cfg["order"],
        start_to_ch1_s=4.0,
        spacing_s=16.0,
        time_radius_s=1.0,
        offset_in_channel_s=3.0,
        channel_len_s=11.0,
        guard_s=0.05,
        freq_tol_hz=100.0,
    )


def run_automation() -> int:
    cfg = _load_yaml_config()
    audio_cfg = cfg.get("audio", {}) if isinstance(cfg.get("audio"), dict) else {}
    layout = str(cfg.get("system_layout", "5.1")).strip() or "5.1"
    fs = int(audio_cfg.get("samplerate", 48000))
    duration = float(audio_cfg.get("record_seconds", 110.0))
    device = audio_cfg.get("input_device")
    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    ref_path = APP_DIR / str(paths.get("ref_wav", DEFAULT_WAV_PATHS["ref"]))
    test_path = APP_DIR / str(paths.get("test_wav", DEFAULT_WAV_PATHS["tst"]))

    if not ref_path.exists():
        raise FileNotFoundError(f"REF wav no existe: {ref_path}")

    print("[AUTORUN] Paso 1/5: Record TEST")
    x_test = record_audio(duration_s=duration, fs=fs, device=device)
    save_wav_mono(str(test_path), fs, x_test)

    print("[AUTORUN] Paso 2/5: Analyze REF")
    fs_ref, x_ref = load_wav_mono(str(ref_path))

    print("[AUTORUN] Paso 3/5: Analyze TEST")
    fs_tst, x_tst = load_wav_mono(str(test_path))
    if fs_ref != fs_tst:
        raise RuntimeError(f"FS mismatch REF={fs_ref} TEST={fs_tst}")
    ref_analysis = _analyze(x_ref, fs_ref, layout)
    tst_analysis = _analyze(x_tst, fs_tst, layout)

    print("[AUTORUN] Paso 4/5: Evaluate")
    ref_freqs = ref_analysis.get("channel_freqs_hz", [])
    tst_freqs = tst_analysis.get("channel_freqs_hz", [])
    criteria = cfg.get("analysis", {}).get("criteria", DEFAULT_BAND_CRITERIA)
    best_corr_thr = CRITERIA_THRESHOLDS.get(criteria, CRITERIA_THRESHOLDS[DEFAULT_BAND_CRITERIA])["best_corr"]
    # Aproximación: usamos cercania de frecuencia para marcar score
    valid_pairs = [
        abs(float(r) - float(t))
        for r, t in zip(ref_freqs, tst_freqs)
        if r is not None and t is not None
    ]
    mean_delta = (sum(valid_pairs) / len(valid_pairs)) if valid_pairs else 9999.0
    score = max(0.0, 1.0 - (mean_delta / 200.0))
    status = "PASSED" if score >= best_corr_thr else "FAILED"

    print("[AUTORUN] Paso 5/5: Thingsboard")
    tb_cfg = _load_tb_config()
    server = str(tb_cfg.get("server", "")).strip()
    token = str(tb_cfg.get("token", "")).strip()
    port = int(tb_cfg.get("port", 1883))
    if server and token:
        payload = {
            "autorun": True,
            "layout": layout,
            "score": round(score, 4),
            "status": status,
            "mean_freq_delta_hz": round(mean_delta, 2),
            "channels_valid": len(valid_pairs),
        }
        send_to_thingsboard(server, port, token, payload)
    else:
        print("[AUTORUN] ThingsBoard no configurado. Saltando envio.")

    print(f"[AUTORUN] Finalizado: status={status}, score={score:.4f}")
    return 0
