FROM python:3.9-slim

WORKDIR /app
COPY api.py .
COPY flight_price_model.pkl .
COPY model_metadata.json .

RUN pip install flask pandas joblib scikit-learn requests

EXPOSE 5000

CMD ["python", "api.py"]
