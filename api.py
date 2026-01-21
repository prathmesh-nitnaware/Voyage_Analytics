import pandas as pd
import joblib
import json
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

try:
    flight_model = joblib.load('flight_price_model.pkl')
    with open('model_metadata.json', 'r') as f:
        flight_meta = json.load(f)
    flight_cols = flight_meta['columns']
    print("✅ Flight Price Model loaded.")
except FileNotFoundError:
    flight_model = None
    print("⚠️ Flight Price Model NOT found. (Run flight training script)")

try:
    gender_model = joblib.load('gender_classification_model.pkl')
    print("✅ Gender Classification Model loaded.")
except FileNotFoundError:
    gender_model = None
    print("⚠️ Gender Model NOT found. (Run gender training script)")

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

@app.route('/predict', methods=['POST'])
def predict_flight():
    if not flight_model:
        return jsonify({'error': 'Flight model not initialized'}), 500
    
    try:
        req_data = request.get_json()
        
        input_dict = {col: [req_data.get(col)] for col in flight_cols}
        input_data = pd.DataFrame(input_dict)
        
        prediction = flight_model.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_price': float(prediction)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/predict_gender', methods=['POST'])
def predict_gender():
    if not gender_model:
        return jsonify({'error': 'Gender model not initialized'}), 500
    
    try:
        name = request.get_json().get('name')
        if not name:
            return jsonify({'error': 'Parameter "name" is required'}), 400
            
        prediction = gender_model.predict([name])[0]
        
        return jsonify({
            'status': 'success',
            'input_name': name,
            'predicted_gender': str(prediction)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/recommend_hotels', methods=['POST'])
def recommend_hotels():
    if not rec_model:
        return jsonify({'error': 'Recommendation model not initialized'}), 500
    
    try:
        user_code = request.get_json().get('user_code')
        if user_code is None:
            return jsonify({'error': 'Parameter "user_code" is required'}), 400

        try:
            u_idx = user_encoder.transform([user_code])[0]
        except ValueError:
            return jsonify({'status': 'error', 'message': 'User unknown (Cold Start)'})

        user_vector = interaction_matrix.iloc[u_idx, :].values.reshape(1, -1)
        distances, indices = rec_model.kneighbors(user_vector, n_neighbors=3)
        
        similar_user_idx = indices.flatten()[1]
        
        sim_user_vector = interaction_matrix.iloc[similar_user_idx]
        recommended_hotel_indices = sim_user_vector[sim_user_vector > 0].index.tolist()
        
        recommended_hotels = hotel_encoder.inverse_transform(recommended_hotel_indices)
        
        return jsonify({
            'status': 'success',
            'user_code': int(user_code),
            'similar_to_user_idx': int(similar_user_idx),
            'recommendations': list(recommended_hotels[:5]) 
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)