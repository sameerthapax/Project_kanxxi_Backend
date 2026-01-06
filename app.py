import os
import hashlib
import numpy as np
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from scipy.io.wavfile import write as wavwrite

from tts.coqui_tts import CoquiTacotron2TTS

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
OUTPUTS = os.path.join(BASE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

MAX_CHARS = int(os.getenv("MAX_CHARS", "300"))

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

tts_engine = CoquiTacotron2TTS(
    tts_checkpoint=os.path.join(MODELS, "best_model.pth"),
    tts_config_path=os.path.join(MODELS, "config.json"),
    device="cpu",
)

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "sample_rate": tts_engine.sample_rate})

@app.post("/api/tts")
def tts():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "missing_text"}), 400
    if len(text) > MAX_CHARS:
        return jsonify({"error": "text_too_long", "max_chars": MAX_CHARS}), 400

    key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    out_path = os.path.join(OUTPUTS, f"{key}.wav")

    if not os.path.exists(out_path):
        wav = tts_engine.tts_to_wav(text)  # float32 -1..1
        wav = np.clip(wav, -1.0, 1.0)
        wav_i16 = (wav * 32767.0).astype(np.int16)

        wavwrite(out_path, tts_engine.sample_rate, wav_i16)

    return send_file(out_path, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)