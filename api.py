import pandas as pd
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==========================================
# 1. LOAD ARTIFACTS
# ==========================================

# --- A. Flight Price Model ---
try:
    flight_model = joblib.load('flight_price_model.pkl')
    with open('model_metadata.json', 'r') as f:
        flight_meta = json.load(f)
    flight_cols = flight_meta['columns']
    print("✅ Flight Price Model loaded.")
except FileNotFoundError:
    flight_model = None
    print("⚠️ Flight Price Model NOT found. (Run flight training script)")

# --- B. Gender Classification Model ---
try:
    gender_model = joblib.load('gender_classification_model.pkl')
    print("✅ Gender Classification Model loaded.")
except FileNotFoundError:
    gender_model = None
    print("⚠️ Gender Model NOT found. (Run gender training script)")

# --- C. Hotel Recommendation Model ---
try:
    rec_artifacts = joblib.load('hotel_recommendation_model.pkl')
    rec_model = rec_artifacts['model']
    user_encoder = rec_artifacts['user_encoder']
    hotel_encoder = rec_artifacts['hotel_encoder']
    interaction_matrix = rec_artifacts['interaction_matrix']
    print("✅ Hotel Recommendation Model loaded.")
except FileNotFoundError:
    rec_model = None
    print("⚠️ Recommendation Model NOT found. (Run recommendation training script)")


# ==========================================
# 2. API ENDPOINTS
# ==========================================

@app.route('/')
def home():
    status = {
        "service": "Voyage Analytics AI Engine",
        "models_loaded": {
            "flight_price": flight_model is not None,
            "gender_class": gender_model is not None,
            "hotel_recs": rec_model is not None
        }
    }
    return jsonify(status)

# --- Objective 1-7: Flight Price Prediction ---
@app.route('/predict', methods=['POST'])
def predict_flight():
    if not flight_model:
        return jsonify({'error': 'Flight model not initialized'}), 500
    
    try:
        # 1. Receive Payload
        req_data = request.get_json()
        
        # 2. Preprocess (Ensure correct column ordering for the model)
        # Wrap scalar values in lists to create DataFrame
        input_dict = {col: [req_data.get(col)] for col in flight_cols}
        input_data = pd.DataFrame(input_dict)
        
        # 3. Predict
        prediction = flight_model.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_price': float(prediction)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- Objective 8: Gender Classification ---
@app.route('/predict_gender', methods=['POST'])
def predict_gender():
    if not gender_model:
        return jsonify({'error': 'Gender model not initialized'}), 500
    
    try:
        name = request.get_json().get('name')
        if not name:
            return jsonify({'error': 'Parameter "name" is required'}), 400
            
        # The pipeline expects an iterable (list), so we wrap the string
        prediction = gender_model.predict([name])[0]
        
        return jsonify({
            'status': 'success',
            'input_name': name,
            'predicted_gender': str(prediction)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# --- Objective 9: Hotel Recommendations ---
@app.route('/recommend_hotels', methods=['POST'])
def recommend_hotels():
    if not rec_model:
        return jsonify({'error': 'Recommendation model not initialized'}), 500
    
    try:
        user_code = request.get_json().get('user_code')
        if user_code is None:
            return jsonify({'error': 'Parameter "user_code" is required'}), 400

        # 1. Encode User ID
        try:
            u_idx = user_encoder.transform([user_code])[0]
        except ValueError:
            return jsonify({'status': 'error', 'message': 'User unknown (Cold Start)'})

        # 2. Get User Vector & Find Neighbors
        # interaction_matrix is a DataFrame/Matrix where index=user_idx
        user_vector = interaction_matrix.iloc[u_idx, :].values.reshape(1, -1)
        distances, indices = rec_model.kneighbors(user_vector, n_neighbors=3)
        
        # 3. Get Recommendations from Nearest Neighbor (Simplified Collaborative Filtering)
        # We look at the most similar user (index 1, because index 0 is self)
        similar_user_idx = indices.flatten()[1]
        
        # Find hotels the similar user liked/booked
        sim_user_vector = interaction_matrix.iloc[similar_user_idx]
        # Get indices of hotels where booking count > 0
        recommended_hotel_indices = sim_user_vector[sim_user_vector > 0].index.tolist()
        
        # Decode back to Hotel Names
        recommended_hotels = hotel_encoder.inverse_transform(recommended_hotel_indices)
        
        return jsonify({
            'status': 'success',
            'user_code': int(user_code),
            'similar_to_user_idx': int(similar_user_idx),
            'recommendations': list(recommended_hotels[:5]) # Top 5
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Listen on all interfaces
    app.run(host='0.0.0.0', port=5000)