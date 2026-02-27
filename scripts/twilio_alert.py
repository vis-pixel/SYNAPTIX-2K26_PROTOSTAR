"""
BioRhythm Fusion Band — Twilio Voice Call Alert
Triggers an automated phone call with ALL sensor readings when CRITICAL risk fires.

Signals included in call message:
  HR, HRV, ECG, SpO2, GSR/EDA, Temperature, Respiration,
  AccelX/Y/Z, Activity Label

Setup:
  pip install twilio
  Add to .env:
    TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
    TWILIO_AUTH_TOKEN=your_token
    ALERT_PHONE_TO=+91XXXXXXXXXX
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger("biorhythm.twilio_alert")

# ─── Config ───────────────────────────────────────────────────────────────────
TWILIO_FROM  = os.environ.get("TWILIO_CALL_FROM", "+16186533274")
TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
ALERT_TO     = os.environ.get("ALERT_PHONE_TO", "")

# ─── Call deduplication ───────────────────────────────────────────────────────
_last_call_time = {}
CALL_COOLDOWN_SECONDS = 120

def _is_in_cooldown(key="default"):
    now = datetime.now().timestamp()
    return (now - _last_call_time.get(key, 0)) < CALL_COOLDOWN_SECONDS


# ─── Field name aliases (smartwatch / research dataset columns) ───────────────
FIELD_ALIASES = {
    "heart_rate":   ["hr", "heart_rate", "pulse", "HeartRate"],
    "hrv_rmssd":    ["hrv", "hrv_rmssd", "rmssd", "HRV"],
    "ecg":          ["ecg", "mlii", "v5", "avr", "ecg_signal"],
    "spo2":         ["spo2", "SpO2", "blood_oxygen", "oxygen", "spo2_pct"],
    "gsr_eda":      ["gsr", "eda", "electrodermal", "sc", "skin_conductance"],
    "temperature":  ["temperature", "temp", "skin_temp", "body_temp", "Temp"],
    "respiration":  ["respiration", "resp", "resp_rate", "rr", "thorax", "breathing_rate"],
    "accel_x":      ["accel_x", "acceleration_x", "ax", "chest_acc_x", "acc_x"],
    "accel_y":      ["accel_y", "acceleration_y", "ay", "chest_acc_y", "acc_y"],
    "accel_z":      ["accel_z", "acceleration_z", "az", "chest_acc_z", "acc_z"],
    "activity":     ["activity_label", "activity", "label", "activity_type"],
}

def _get(data: dict, field: str, default=None):
    """Get a value from data using any known alias for the field."""
    for alias in FIELD_ALIASES.get(field, [field]):
        if alias in data:
            try:
                return data[alias]
            except Exception:
                pass
    return default


def _say(val, unit="", fmt=".0f"):
    """Format a value for TTS — 'not available' if missing."""
    if val is None or val == "" or val == 0:
        return "not available"
    try:
        if fmt:
            return f"{float(val):{fmt}} {unit}".strip()
        return f"{val} {unit}".strip()
    except Exception:
        return str(val)


def build_twiml_message(risk_score: int, vitals: dict) -> str:
    """Build full TwiML spoken message with ALL sensor readings."""
    hr     = _get(vitals, "heart_rate")
    hrv    = _get(vitals, "hrv_rmssd")
    ecg    = _get(vitals, "ecg")
    spo2   = _get(vitals, "spo2")
    gsr    = _get(vitals, "gsr_eda")
    temp   = _get(vitals, "temperature") or vitals.get("skin_temp") or vitals.get("temperature")
    resp   = _get(vitals, "respiration")
    ax     = _get(vitals, "accel_x")
    ay     = _get(vitals, "accel_y")
    az     = _get(vitals, "accel_z")
    act    = _get(vitals, "activity", "unknown")

    # BP (not in FIELD_ALIASES but common)
    bp_s   = vitals.get("bp_systolic")
    bp_d   = vitals.get("bp_diastolic")

    lines = []
    lines.append(f"Heart rate: {_say(hr, 'B P M')}.")
    lines.append(f"H R V R M S S D: {_say(hrv, 'milliseconds')}.")
    if ecg is not None:
        lines.append(f"E C G signal: {_say(ecg, '', '.3f')}.")
    lines.append(f"Blood oxygen S P O 2: {_say(spo2, 'percent')}.")
    lines.append(f"Electrodermal activity G S R: {_say(gsr, 'microsiemens')}.")
    lines.append(f"Skin temperature: {_say(temp, 'degrees Celsius')}.")
    lines.append(f"Respiration rate: {_say(resp, 'breaths per minute')}.")
    if ax is not None or ay is not None or az is not None:
        lines.append(f"Accelerometer — X: {_say(ax)}, Y: {_say(ay)}, Z: {_say(az)}.")
    if bp_s and bp_d:
        lines.append(f"Blood pressure: {_say(bp_s, '', '.0f')} over {_say(bp_d, 'millimeters mercury', '.0f')}.")
    lines.append(f"Activity label: {act}.")

    sensor_block = " ".join(lines)

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" rate="slow">
        CRITICAL HEALTH ALERT from BioRhythm Fusion Band.
        Risk score: {risk_score} percent. This is a CRITICAL level. Immediate attention required.
    </Say>
    <Pause length="1"/>
    <Say voice="alice">
        Sensor readings at time of alert.
        {sensor_block}
    </Say>
    <Pause length="1"/>
    <Say voice="alice">
        Please open the BioRhythm dashboard immediately and seek medical attention if needed.
        Risk score: {risk_score} percent. CRITICAL.
    </Say>
</Response>"""
    return twiml


def trigger_critical_call(risk_score: int, vitals: dict) -> dict:
    """
    Trigger a Twilio voice call when risk score >= 80%.

    Returns: dict with 'success', 'call_sid', 'message'
    """
    if not TWILIO_SID or not TWILIO_TOKEN:
        return {
            "success": False,
            "call_sid": None,
            "message": "Twilio not configured. Add TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN to .env",
        }
    if not ALERT_TO:
        return {
            "success": False,
            "call_sid": None,
            "message": "No recipient. Add ALERT_PHONE_TO=+91XXXXXXXXXX to .env",
        }
    if _is_in_cooldown():
        return {
            "success": False,
            "call_sid": None,
            "message": f"Cooldown active — last call was within {CALL_COOLDOWN_SECONDS}s",
        }

    try:
        from twilio.rest import Client
    except ImportError:
        return {
            "success": False,
            "call_sid": None,
            "message": "Run: pip install twilio",
        }

    twiml = build_twiml_message(risk_score, vitals)

    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        call = client.calls.create(
            twiml=twiml,
            to=ALERT_TO,
            from_=TWILIO_FROM,
        )
        _last_call_time["default"] = datetime.now().timestamp()
        msg = f"Call to {ALERT_TO} | Risk={risk_score}%"
        logger.warning(f"📞 CRITICAL CALL → {msg} | SID: {call.sid}")
        return {"success": True, "call_sid": call.sid, "message": msg}
    except Exception as e:
        logger.error(f"Twilio call failed: {e}")
        return {"success": False, "call_sid": None, "message": str(e)}
