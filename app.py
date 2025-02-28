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
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the trained model
model = joblib.load(MODEL_PATH)
logging.info(f"Model loaded successfully. Expecting {model.n_features_in_} features.")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Serve the front-end

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()
        logging.info(f"Received data: {data}")

        # Validate JSON structure
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request"}), 400

        # Convert input to NumPy array
        input_data = np.array(data["features"]).reshape(1, -1)

        # Ensure input matches model expectations
        expected_features = model.n_features_in_
        if input_data.shape[1] != expected_features:
            error_msg = f"Expected {expected_features} features, got {input_data.shape[1]}"
            logging.error(error_msg)
            return jsonify({"error": error_msg}), 400

        # Log input values
        logging.info(f"Input data for prediction: {input_data}")

        # Make prediction
        prediction = model.predict(input_data)[0]
        logging.info(f"Prediction: {prediction}")

        # Return response
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Use Heroku's port if available, otherwise 8080 for local testing
    logging.info(f"Starting Flask server on port {port}...")
    app.run(debug=True, host="0.0.0.0", port=port)
