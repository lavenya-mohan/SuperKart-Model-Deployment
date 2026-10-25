from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
model = joblib.load(ROOT / "superkart_model.joblib")
with open(ROOT / "schema.json") as f:
    schema = json.load(f)
FEATURES = schema["features"]
NUMERIC = schema["numeric"]

def prepare(frame):
    missing = [col for col in FEATURES if col not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = frame[FEATURES].copy()
    if "Product_Sugar_Content" in frame.columns:
        frame["Product_Sugar_Content"] = frame["Product_Sugar_Content"].replace({"reg": "Regular"})
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame.isna().any().any():
        raise ValueError("Null values are not supported in prediction inputs")
    return frame

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/v1/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected one JSON object"}), 400
        frame = prepare(pd.DataFrame([payload]))
        value = float(model.predict(frame)[0])
        return jsonify({"prediction": value})
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/v1/predictbatch")
def predictbatch():
    try:
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify({"error": "Upload CSV using multipart field 'file'"}), 400
        frame = prepare(pd.read_csv(uploaded))
        if frame.empty:
            return jsonify({"error": "CSV is empty"}), 400
        predictions = model.predict(frame)
        return jsonify({str(i): float(value) for i, value in enumerate(predictions)})
    except (ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
