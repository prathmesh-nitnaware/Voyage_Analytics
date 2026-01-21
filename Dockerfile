FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask requests

COPY api.py .

COPY flight_price_model.pkl .
COPY model_metadata.json .
COPY gender_classification_model.pkl .
COPY hotel_recommendation_model.pkl .

EXPOSE 5000

CMD ["python", "api.py"]