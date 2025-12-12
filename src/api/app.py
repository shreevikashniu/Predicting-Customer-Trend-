# api/app.py
from flask import Flask, request, jsonify
from pathlib import Path
import pickle
import numpy as np

MODEL_PATH = Path("../models/rf_model.pkl")
ENC_PATH = Path("../models/encoder.pkl")

app = Flask(__name__)

model = None
encoder = None

def load():
    global model, encoder
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENC_PATH, "rb") as f:
        encoder = pickle.load(f)

@app.before_first_request
def startup():
    load()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok"}), 200

@app.route("/predict_repeat", methods=["POST"])
def predict_repeat():
    """
    Expects JSON body:
    {
      "product": "A",
      "quantity": 2,
      "total_spent": 120.5,
      "total_quantity": 10,
      "month": 7,
      "dayofweek": 2,
      "recency_days": 14
    }
    """
    if model is None or encoder is None:
        load()
    data = request.get_json()
    required = ["product","quantity","total_spent","total_quantity","month","dayofweek","recency_days"]
    if not data or not all(k in data for k in required):
        return jsonify({"error":"missing fields", "required": required}), 400
    try:
        prod = data["product"]
        qty = float(data["quantity"])
        total_spent = float(data["total_spent"])
        total_quantity = float(data["total_quantity"])
        month = int(data["month"])
        dayofweek = int(data["dayofweek"])
        recency = float(data["recency_days"])

        prod_ohe = encoder.transform([[prod]])
        features = np.hstack([prod_ohe, [[qty, total_spent, total_quantity, month, dayofweek, recency]]])
        pred = model.predict(features)
        prob = model.predict_proba(features).tolist() if hasattr(model,"predict_proba") else None

        return jsonify({"repeat_30d": int(pred[0]), "probabilities": prob}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
