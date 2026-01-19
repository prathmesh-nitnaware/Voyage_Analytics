FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt || pip install flask pandas joblib scikit-learn requests scipy

COPY api.py .

COPY flight_price_model.pkl .
COPY model_metadata.json .

COPY gender_classification_model.pkl .

COPY hotel_recommendation_model.pkl .

EXPOSE 5000

CMD ["python", "api.py"]