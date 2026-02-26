from __future__ import annotations

import json
import numpy as np

from app_platform import APP_DIR, CFG_DIR

from gui_app import (
    BAND_RANGES_HZ,
    DEFAULT_BAND_CRITERIA,
    DEFAULT_WAV_PATHS,
    SYSTEM_LAYOUTS,
    align_by_lag_s,
    analyze_track,
    band_power_db,
    best_xcorr_shift_on_freq_curve,
    compare_sweep_freq_curves_xcorr,
    compute_spectrum,
    get_best_corr_threshold_from_criteria,
    get_delta_threshold_db_from_criteria,
    get_trimmed_sweep_from_window,
    load_wav_mono,
    record_audio,
    save_wav_mono,
    send_to_thingsboard,
    sweep_freq_track,
)





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
    cfg = _load_yaml_config()
    timing_cfg = cfg.get("timing", {}) if isinstance(cfg.get("timing"), dict) else {}
    analysis_cfg = cfg.get("analysis", {}) if isinstance(cfg.get("analysis"), dict) else {}
    marker_cfg = cfg.get("markers", {}) if isinstance(cfg.get("markers"), dict) else {}
    sys_cfg = SYSTEM_LAYOUTS.get(layout, SYSTEM_LAYOUTS["5.1"])
    return analyze_track(
        x=x,
        fs=fs,
        f_start=float(marker_cfg.get("start_hz", 3000.0)),
        f_ch_marker=float(marker_cfg.get("channel_marker_hz", 4500.0)),
        f_end=float(marker_cfg.get("end_hz", 3000.0)),
        channel_count=sys_cfg["channels"],
        channel_order=sys_cfg["order"],
        start_to_ch1_s=float(timing_cfg.get("start_to_ch1_s", 4.0)),
        spacing_s=float(timing_cfg.get("spacing_s", 16.0)),
        time_radius_s=float(timing_cfg.get("radius_s", 1.0)),
        offset_in_channel_s=float(timing_cfg.get("offset_s", 3.0)),
        channel_len_s=float(timing_cfg.get("channel_len_s", 11.0)),
        guard_s=float(timing_cfg.get("guard_s", 0.05)),
        freq_tol_hz=float(analysis_cfg.get("freq_tol_hz", 100.0)),
    )


def _band_results_to_payload(results: list[dict]) -> tuple[dict, dict, dict]:
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


def _build_thingsboard_channel(score_text: str, status_text: str, band_results: list[dict]) -> dict:
    ref, cine, delta = _band_results_to_payload(band_results)
    return {
        "Score": score_text,
        "Status": status_text,
        "ref": ref,
        "cine": cine,
        "delta": delta,
    }


def _compute_band_results(
    ch_idx: int,
    fs: int,
    x_ref: np.ndarray,
    x_tst: np.ndarray,
    win_ref: list,
    win_tst: list,
    subwoofer_index: int,
    band_tol_db: float,
) -> list[dict]:
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

    return results




def run_automation() -> int:
    cfg = _load_yaml_config()
    audio_cfg = cfg.get("audio", {}) if isinstance(cfg.get("audio"), dict) else {}
    analysis_cfg = cfg.get("analysis", {}) if isinstance(cfg.get("analysis"), dict) else {}
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

    criteria = cfg.get("analysis", {}).get("criteria", DEFAULT_BAND_CRITERIA)


    best_corr_thr = get_best_corr_threshold_from_criteria(criteria)
    band_tol_db = get_delta_threshold_db_from_criteria(criteria)

    channel_count = int(ref_analysis.get("channel_count", 0) or 0)
    if channel_count <= 0:
        channel_count = SYSTEM_LAYOUTS.get(layout, SYSTEM_LAYOUTS["5.1"])["channels"]
    subwoofer_index = max(0, channel_count - 1)
    win_ref = ref_analysis.get("windows_s", [])
    win_tst = tst_analysis.get("windows_s", [])
    if len(win_ref) < channel_count or len(win_tst) < channel_count:
        raise RuntimeError("Missing channel windows. Re-run Analyze REF/TEST.")

    channel_payloads = {}
    for ch in range(channel_count):
        ref_trim, _ = get_trimmed_sweep_from_window(x_ref, fs_ref, win_ref[ch])
        tst_trim, _ = get_trimmed_sweep_from_window(x_tst, fs_tst, win_tst[ch])

        if ch == subwoofer_index:
            fmin = 20.0
            fmax = min(200.0, fs_ref / 2 - 50)
            win_s = 0.20
            hop_s = 0.06
            gate_db_below_global = 28.0
        else:
            fmin = 20.0
            fmax = min(20000.0, fs_ref / 2 - 50)
            win_s = 0.06
            hop_s = 0.02
            gate_db_below_global = 35.0

        tR, fR = sweep_freq_track(
            ref_trim,
            fs_ref,
            fmin=fmin,
            fmax=fmax,
            win_s=win_s,
            hop_s=hop_s,
            gate_db_below_global=gate_db_below_global,
        )
        tT, fT = sweep_freq_track(
            tst_trim,
            fs_tst,
            fmin=fmin,
            fmax=fmax,
            win_s=win_s,
            hop_s=hop_s,
            gate_db_below_global=gate_db_below_global,
        )
        _, met = compare_sweep_freq_curves_xcorr(
            tR,
            fR,
            tT,
            fT,
            tol_hz=float(analysis_cfg.get("freq_tol_hz", 100.0)),
            max_lag_s=3,
            min_valid_frac=0.75,
            pctl_err=90.0,
            range_tol_hz=200.0,
        )
        best_corr = met.get("best_corr", None)
        is_alive = (best_corr is not None) and np.isfinite(best_corr) and (best_corr >= best_corr_thr)
        status_text = "LIVE" if is_alive else "DEAD"

        band_results = _compute_band_results(
            ch_idx=ch,
            fs=fs_ref,
            x_ref=x_ref,
            x_tst=x_tst,
            win_ref=win_ref,
            win_tst=win_tst,
            subwoofer_index=subwoofer_index,
            band_tol_db=band_tol_db,
        )
        score_text = "PASSED" if all(r.get("status") == "OK" for r in band_results) else "FAILED"
        channel_payloads[f"Canal{ch + 1}"] = _build_thingsboard_channel(
            score_text=score_text,
            status_text=status_text,
            band_results=band_results,
        )

    

    print("[AUTORUN] Paso 5/5: Thingsboard")
    tb_cfg = _load_tb_config()
    server = str(tb_cfg.get("server", "")).strip()
    token = str(tb_cfg.get("token", "")).strip()
    port = int(tb_cfg.get("port", 1883))
    if server and token:
        send_to_thingsboard(server, port, token, channel_payloads)
    else:
        print("[AUTORUN] ThingsBoard no configurado. Saltando envio.")

    print(f"[AUTORUN] Finalizado: canales_enviados={len(channel_payloads)}")
    return 0
