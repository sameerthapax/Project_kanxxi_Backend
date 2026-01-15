import os
import hashlib
import time
import numpy as np
import soundfile as sf
from flask import Flask, request, send_file, jsonify, make_response
from flask_cors import CORS

from tts.coqui_tts import CoquiTacotron2TTS
from services.translate_nllb import get_translator

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

MAX_CHARS = int(os.getenv("MAX_CHARS", "300"))
PORT = int(os.getenv("PORT", "4000"))
ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGIN}})

tts_engine = CoquiTacotron2TTS(
    tts_checkpoint=os.path.join(MODELS, "best_model_5338.pth"),
    tts_config_path=os.path.join(MODELS, "config.json"),
    device="cpu",
)

_rate_state = {}

def _client_ip():
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"

def _origin_allowed():
    origin = request.headers.get("Origin", "")
    return origin == ALLOWED_ORIGIN

def _rate_limited(ip):
    now = time.time()
    window_start, count = _rate_state.get(ip, (now, 0))
    if now - window_start >= RATE_LIMIT_WINDOW_SEC:
        window_start = now
        count = 0
    count += 1
    _rate_state[ip] = (window_start, count)
    if count > RATE_LIMIT_MAX:
        retry_after = int(RATE_LIMIT_WINDOW_SEC - (now - window_start))
        return True, max(retry_after, 0)
    return False, 0

def _validate_text(text):
    if not text:
        return jsonify({"error": "missing_text"}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": "text_too_long", "max_chars": MAX_CHARS}), 400
    return None

def _synthesize_to_file(text):
    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(OUTPUTS, f"{key}.wav")

    if not os.path.exists(out_path):
        wav = tts_engine.tts_to_wav(text).astype(np.float32)

        # Normalize to avoid clipping distortion (your wav can exceed [-1, 1])
        peak = float(np.max(np.abs(wav)))
        if peak > 0:
            wav = wav / peak

        # Small headroom to avoid hitting full scale
        wav = wav * 0.98

        # Write 16-bit PCM WAV
        sf.write(out_path, wav, tts_engine.sample_rate, subtype="PCM_16")

    return send_file(out_path, mimetype="audio/wav")

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "sample_rate": tts_engine.sample_rate})

@app.post("/api/tts")
def tts():
    if not _origin_allowed():
        return jsonify({"error": "origin_not_allowed"}), 403

    ip = _client_ip()
    limited, retry_after = _rate_limited(ip)
    if limited:
        resp = make_response(jsonify({"error": "rate_limited"}), 429)
        resp.headers["Retry-After"] = str(retry_after)
        return resp

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    error = _validate_text(text)
    if error:
        return error

    return _synthesize_to_file(text)

@app.post("/api/tts-english")
def tts_english():
    if not _origin_allowed():
        return jsonify({"error": "origin_not_allowed"}), 403

    ip = _client_ip()
    limited, retry_after = _rate_limited(ip)
    if limited:
        resp = make_response(jsonify({"error": "rate_limited"}), 429)
        resp.headers["Retry-After"] = str(retry_after)
        return resp

    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    error = _validate_text(text)
    if error:
        return error

    translator = get_translator()
    translated = translator.translate_en_to_ne(text,max_new_tokens=96,num_beams=1).text.strip()
    if not translated:
        return jsonify({"error": "translation_failed"}), 500

    error = _validate_text(translated)
    if error:
        return error

    return _synthesize_to_file(translated)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
