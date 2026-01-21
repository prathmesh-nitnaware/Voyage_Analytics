# Voyage Analytics 🚀  
### Integrating MLOps in Travel – Productionization of ML Systems

Voyage Analytics is an end-to-end **Machine Learning + MLOps project** focused on the **travel domain**.  
It demonstrates how machine learning models are not only trained, but **deployed, served, monitored, and automated** in a production-like environment.

The project covers **flight price prediction**, **hotel recommendation**, and **user attribute classification**, along with API deployment and MLOps components.

---

## 📌 Project Objectives

- Build multiple ML models for travel-related use cases
- Serve models via a backend API
- Integrate frontend for user interaction
- Demonstrate MLOps concepts such as:
  - Model versioning
  - Experiment tracking
  - Pipeline automation
  - Deployment readiness

---

## 🧠 Use Cases Implemented

1. **Flight Price Prediction**
   - Predicts flight ticket prices based on travel features

2. **Hotel Recommendation System**
   - Recommends hotels using collaborative filtering

3. **Gender Classification**
   - Predicts gender from user input (demonstration model)

4. **Smart Trip Planning**
   - Combines flight + hotel cost under a given budget

---

## 🏗️ Project Architecture (High Level)
```
Raw Data
↓
Data Cleaning & Feature Engineering
↓
Model Training & Experiment Tracking
↓
Saved Models + Metadata
↓
Flask API (Model Serving)
↓
Streamlit Frontend
↓
Deployment Configuration (Docker / Kubernetes)
```


---

## 📂 Repository Structure
```
Voyage_Analytics/
│
├── Submission.ipynb # Data analysis & model training notebook
├── api.py # Flask backend API for model inference
├── app.py # Streamlit frontend application
│
├── flights.csv # Raw flight dataset
├── hotels.csv # Raw hotel dataset
├── users.csv # Raw user dataset
│
├── cleaned_flights.csv # Cleaned flight data
├── cleaned_hotels.csv # Cleaned hotel data
├── cleaned_users.csv # Cleaned user data
│
├── flight_price_model.pkl # Trained flight price prediction model
├── hotel_recommendation_model.pkl # Trained hotel recommendation model
├── gender_classification_model.pkl # Trained gender classification model
│
├── model_columns.json # Model input feature schema
├── model_metadata.json # Model metadata
│
├── voyage_automation_dag.py # Workflow automation (Airflow DAG)
├── mlruns/ # MLflow experiment tracking
│
├── Dockerfile # Docker configuration
├── deployment.yaml # Kubernetes deployment config
├── service.yaml # Kubernetes service config
├── requirements.txt # Project dependencies
└── README.md # Project documentation
```


---

## ⚙️ Tech Stack Used

- **Programming Language:** Python
- **ML Libraries:** Scikit-learn, Pandas, NumPy
- **Model Serving:** Flask
- **Frontend:** Streamlit
- **Experiment Tracking:** MLflow
- **Workflow Automation:** Apache Airflow (DAG)
- **Containerization:** Docker
- **Deployment:** Kubernetes (YAML configs)

---

## ▶️ How to Run the Project Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

2️⃣ Start Backend API
```
python api.py
```

API runs at:
```
http://localhost:5000
```

3️⃣ Start Frontend Application
```
streamlit run app.py
```

## 📊 Machine Learning & MLOps Concepts Demonstrated

- Train/Test split and model evaluation  
- Model serialization (.pkl)  
- Feature schema management  
- REST API for ML inference  
- Separation of training and serving  
- Experiment tracking with MLflow  
- Pipeline automation concepts  
- Deployment-ready configuration  

## 📌 Key Learnings

- ML models must be monitored and maintained after deployment  
- Real-world ML systems require automation and versioning  
- MLOps bridges the gap between ML development and production  
- Travel domain data is dynamic and requires continuous model updates  

## 📜 Conclusion

Voyage Analytics demonstrates a production-oriented ML system, moving beyond notebooks to APIs, automation, and deployment. The project highlights how MLOps practices are essential for scalable and reliable machine learning systems, especially in dynamic domains like travel.
