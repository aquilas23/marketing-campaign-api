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

# Define model path
MODEL_PATH = "marketing_campaign_model.pkl"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained model
try:
    model = joblib.load(MODEL_PATH)
    expected_features = model.n_features_in_
    logging.info(f"Model loaded successfully. Expecting {expected_features} features.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise e

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the front-end HTML page

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()

        # Validate JSON structure
        if "features" not in data:
            logging.warning("Received request with missing 'features' key.")
            return jsonify({"error": "Missing 'features' key in request"}), 400

        features = data["features"]

        # Ensure input is a list of numbers
        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            logging.warning("Invalid input: Features must be a list of numbers.")
            return jsonify({"error": "Invalid input: 'features' must be a list of numbers"}), 400

        # Convert input list to NumPy array
        input_data = np.array(features).reshape(1, -1)

        # Validate input shape
        if input_data.shape[1] != expected_features:
            logging.warning(f"Feature length mismatch: Expected {expected_features}, got {input_data.shape[1]}")
            return jsonify({"error": f"Expected {expected_features} features, got {input_data.shape[1]}"}), 400

        # Make prediction
        prediction = model.predict(input_data)[0]
        logging.info(f"Prediction made successfully: {prediction}")

        # Return response
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Use Heroku's port or default to 8080
    logging.info(f"Starting Flask server on port {port}...")
    app.run(debug=True, host="0.0.0.0", port=port)
