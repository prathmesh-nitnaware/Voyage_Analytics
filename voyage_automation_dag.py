from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

default_args = {
    'owner': 'voyage-analytics',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def train_model():
    """Loads data, trains the model, and saves the new version."""
    print("🔄 Starting scheduled model retraining...")
    
    flights = pd.read_csv("cleaned_flights.csv")
    X = flights.drop('price', axis=1)
    y = flights['price']
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    
    model.fit(X, y)
    
    joblib.dump(model, 'flight_price_model.pkl')
    print(" Model retrained and saved successfully.")

with DAG(
    'voyage_retraining_pipeline',
    default_args=default_args,
    description='Weekly retraining of Flight Price Predictor',
    schedule_interval=timedelta(days=7), 
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    start_task = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "Starting Voyage Analytics Pipeline..."'
    )

    train_task = PythonOperator(
        task_id='retrain_model',
        python_callable=train_model
    )

    deploy_task = BashOperator(
        task_id='deploy_container',
        bash_command='docker build -t voyage-api . && docker restart voyage-api'
    )

    end_task = BashOperator(
        task_id='notify_success',
        bash_command='echo "Pipeline Finished! New model is live."'
    )

    start_task >> train_task >> deploy_task >> end_task