FROM python:3.9-slim

WORKDIR /app

# 1. Copy Requirements
COPY requirements.txt .

# 2. Install Dependencies
# We explicitly install flask and requests here to guarantee they exist
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir flask requests

# 3. Copy API Code
COPY api.py .

# 4. Copy Artifacts (Models & Metadata)
COPY flight_price_model.pkl .
COPY model_metadata.json .
COPY gender_classification_model.pkl .
COPY hotel_recommendation_model.pkl .

# 5. Expose Port
EXPOSE 5000

# 6. Run API
CMD ["python", "api.py"]