import pandas as pd
import joblib
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Load Artifacts ---
try:
    model = joblib.load('flight_price_model.pkl')
    with open('model_metadata.json', 'r') as f:
        meta_data = json.load(f)
    expected_columns = meta_data['columns']
    print("✅ Model and metadata loaded.")
except FileNotFoundError:
    print("❌ Error: Artifacts not found. Please run Step 1 to save model/metadata.")
    model = None

@app.route('/')
def home():
    return "Voyage Analytics Flight Price API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # 1. Get JSON data
        req_data = request.get_json()

        # 2. Convert to DataFrame (ensuring correct column order)
        # We wrap scalar values in lists ([value]) to create a valid DataFrame
        input_data = pd.DataFrame({col: [req_data.get(col)] for col in expected_columns})

        # 3. Predict (Pipeline handles preprocessing automatically)
        prediction = model.predict(input_data)[0]

        return jsonify({
            'status': 'success',
            'predicted_price': float(prediction)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Run on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
