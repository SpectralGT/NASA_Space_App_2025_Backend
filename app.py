# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
app = Flask(__name__)

CORS(app)
# Load model
model = joblib.load("random_forest.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()          # Expect JSON: { "features": [0, 0,0,0,0,0,0,0,0] }
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
