"""
BioRhythm Fusion Band — Standalone Risk Prediction Server
Run: python scripts/risk_server.py
Dashboard: http://localhost:5000/live.html
API:       POST http://localhost:5000/live-data
"""

import json
import os
import pickle
import sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

# Load .env manually (no pip dependency)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Import Twilio alert module
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from twilio_alert import trigger_critical_call, ALERT_TO
    _TWILIO_OK = True
except Exception as _te:
    _TWILIO_OK = False
    print(f"  [WARN] Twilio alert module: {_te}")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_DIR = ROOT / "models"

# ─── Load model ───────────────────────────────────────────────────────────────
_MODEL = None
_CONFIG = None

try:
    config_path = MODEL_DIR / "lstm_model_config.json"
    model_path  = MODEL_DIR / "lstm_autoencoder.pkl"
    if config_path.exists() and model_path.exists():
        with open(config_path) as f:
            _CONFIG = json.load(f)
        with open(model_path, "rb") as f:
            _MODEL = pickle.load(f)
        print(f"  [OK] LSTM model loaded")
    else:
        print(f"  [WARN] Model not found — rule-based fallback")
except Exception as e:
    print(f"  [WARN] Model load failed: {e} — using fallback")

# ─── Risk computation ─────────────────────────────────────────────────────────
def compute_risk(data):
    hr   = float(data.get("heart_rate",  72))
    spo2 = float(data.get("spo2", 98))
    temp = float(data.get("skin_temp", data.get("temperature", 36.5)))
    hrv  = float(data.get("hrv_rmssd", 55))
    resp = float(data.get("respiration", 15))
    bp_s = float(data.get("bp_systolic", 120))

    risk = 0.0
    if hr > 100:    risk += (hr - 100) * 1.5
    if hr < 55:     risk += (55 - hr) * 1.2
    if spo2 < 95:   risk += (95 - spo2) * 8.0
    if temp > 37.5: risk += (temp - 37.5) * 20
    if temp < 35.5: risk += (35.5 - temp) * 15
    if hrv < 30:    risk += (30 - hrv) * 0.8
    if bp_s > 140:  risk += (bp_s - 140) * 1.0
    # Cross-signal patterns
    if hr > 85 and hrv < 35:    risk += 15
    if temp > 37.2 and hr > 90: risk += 20
    if spo2 < 95 and resp > 20: risk += 25

    if _MODEL is not None and _CONFIG is not None:
        try:
            norm_ranges = _CONFIG.get("normalization_ranges", {})
            features = ["heart_rate","hrv_rmssd","spo2","eda","skin_temp",
                        "respiration","steps","bp_systolic","bp_diastolic",
                        "sleep_quality","hour_of_day"]
            vitals_map = {
                "heart_rate": hr, "hrv_rmssd": hrv, "spo2": spo2,
                "eda": float(data.get("eda", 2.0)), "skin_temp": temp,
                "respiration": resp, "steps": float(data.get("steps", 0)),
                "bp_systolic": bp_s,
                "bp_diastolic": float(data.get("bp_diastolic", 78)),
                "sleep_quality": min(1.0, float(data.get("sleep_hours", 7)) / 9),
                "hour_of_day": 12.0,
            }
            def _norm(v, col):
                lo, hi = norm_ranges.get(col, [0, 1])
                return max(0.0, min(1.0, (float(v) - lo) / (hi - lo)))
            vec = [_norm(vitals_map.get(c, 0), c) for c in features]
            seq_len = _CONFIG["architecture"]["seq_len"]
            window = [vec[:] for _ in range(seq_len)]
            err = _MODEL.reconstruction_error(window)
            hm = _CONFIG["statistics"]["healthy_error_mean"]
            hs = _CONFIG["statistics"]["healthy_error_std"]
            lstm_risk = max(0.0, min(100.0, (err - hm) / max(hs * 3, 1e-8) * 100))
            risk = risk * 0.4 + lstm_risk * 0.6
        except Exception as e:
            print(f"  [WARN] LSTM inference failed: {e}")

    risk = max(0, min(100, int(risk)))
    level = "Normal" if risk < 60 else "Warning" if risk < 80 else "Critical"
    return {"risk_score": risk, "alert_level": level, "reconstruction_error": round(risk / 100, 4)}

# ─── MIME types ───────────────────────────────────────────────────────────────
MIME = {".html":"text/html",".css":"text/css",".js":"application/javascript",
        ".json":"application/json",".png":"image/png",".ico":"image/x-icon",".svg":"image/svg+xml"}

# ─── HTTP Handler ─────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path == "/live-data":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body.decode("utf-8"))
                result = compute_risk(data)

                # ─── Trigger Twilio call on CRITICAL ─────────────────
                call_result = None
                if result["alert_level"] == "Critical" and _TWILIO_OK:
                    print(f"  ⚠️  CRITICAL! Risk={result['risk_score']}% → Triggering call to {ALERT_TO}")
                    call_result = trigger_critical_call(result["risk_score"], data)
                    result["call_triggered"] = call_result
                    print(f"  📞 Call result: {call_result.get('message', '?')}")
                else:
                    result["call_triggered"] = None

                self._send_json(200, result)
                print(f"  /live-data → risk={result['risk_score']}% ({result['alert_level']}){' 📞' if call_result and call_result.get('success') else ''}")
            except Exception as e:
                self._send_json(400, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Not found"})

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "":
            path = "/live.html"
        file_path = FRONTEND_DIR / path.lstrip("/")
        if file_path.is_file():
            ext = file_path.suffix.lower()
            mime = MIME.get(ext, "application/octet-stream")
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(content))
            self._cors()
            self.end_headers()
            self.wfile.write(content)
        else:
            self._send_json(404, {"error": f"File not found: {path}"})

    def _send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress default request logging

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = 5000
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n{'='*55}")
    print(f"  BioRhythm Fusion Band — Risk Server")
    print(f"{'='*55}")
    print(f"  Dashboard: http://localhost:{port}/live.html")
    print(f"  API:       POST http://localhost:{port}/live-data")
    print(f"{'='*55}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("  Stopped.")
