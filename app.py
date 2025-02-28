from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import logging

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
CORS(app)  # Enable CORS for front-end requests

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model and scaler
MODEL_PATH = "marketing_campaign_model.pkl"
SCALER_PATH = "scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Model or scaler file is missing!")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
logging.info("✅ Model loaded successfully. Expecting 10 features.")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the front-end

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data or len(data["features"]) != 10:
            return jsonify({"error": "Expected 10 features, got {}".format(len(data.get("features", [])))}), 400

        input_data = np.array(data["features"]).reshape(1, -1)

        # Scale input data
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
