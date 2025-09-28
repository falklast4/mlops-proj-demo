from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.models import Variable
import mlflow
import logging

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['mlops-team@company.com']
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML Model Training Pipeline',
    schedule_interval='@daily',  # Run daily
    catchup=False,
    tags=['ml', 'training', 'distilbert']
)

def check_data_quality(**context):
    """Data quality checks"""
    logging.info("Running data quality checks...")
    
    # Add your data quality checks here
    # Example: Check data freshness, completeness, schema validation
    
    # For demo purposes, we'll simulate checks
    import random
    if random.random() > 0.95:  # 5% chance of failure
        raise ValueError("Data quality check failed!")
    
    logging.info("Data quality checks passed!")
    return True

def fetch_training_data(**context):
    """Fetch and prepare training data"""
    logging.info("Fetching training data...")
    
    # This would typically:
    # 1. Connect to your data warehouse
    # 2. Execute feature engineering queries
    # 3. Save processed data to staging area
    
    # For demo, we'll create a sample dataset
    import pandas as pd
    import os
    
    # Sample data - replace with your actual data pipeline
    data = {
        'text': [
            'I love this product!',
            'This is terrible.',
            'Amazing quality and service.',
            'Worst experience ever.',
            'Highly recommend this!',
            'Complete waste of money.',
        ] * 100,  # Repeat to have more samples
        'label': [1, 0, 1, 0, 1, 0] * 100
    }
    
    df = pd.DataFrame(data)
    
    # Save to shared storage (in real scenario, this would be S3, GCS, etc.)
    os.makedirs('/tmp/airflow_data', exist_ok=True)
    df.to_csv('/tmp/airflow_data/training_data.csv', index=False)
    
    # Log data statistics
    logging.info(f"Training data shape: {df.shape}")
    logging.info(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    return '/tmp/airflow_data/training_data.csv'

def evaluate_model(**context):
    """Evaluate trained model"""
    logging.info("Evaluating model performance...")
    
    # This would typically:
    # 1. Load the trained model
    # 2. Run evaluation on test set
    # 3. Check if model meets quality thresholds
    # 4. Generate model report
    
    # For demo, simulate evaluation metrics
    import random
    
    metrics = {
        'accuracy': 0.85 + random.random() * 0.1,
        'precision': 0.82 + random.random() * 0.1,
        'recall': 0.78 + random.random() * 0.1,
        'f1_score': 0.80 + random.random() * 0.1
    }
    
    logging.info(f"Model metrics: {metrics}")
    
    # Check if model meets minimum thresholds
    min_accuracy = float(Variable.get("min_model_accuracy", default_var=0.80))
    
    if metrics['accuracy'] < min_accuracy:
        raise ValueError(f"Model accuracy {metrics['accuracy']:.3f} below threshold {min_accuracy}")
    
    # Store metrics in XCom for next tasks
    return metrics

def register_model(**context):
    """Register model in MLflow"""
    logging.info("Registering model...")
    
    # Get metrics from previous task
    ti = context['ti']
    metrics = ti.xcom_pull(task_ids='evaluate_model')
    
    # In real scenario, you would:
    # 1. Load the actual trained model
    # 2. Register it in MLflow with proper versioning
    # 3. Tag it appropriately
    
    model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logging.info(f"Registered model version: {model_version}")
    logging.info(f"Model metrics: {metrics}")
    
    # Store model version for deployment pipeline
    Variable.set("latest_model_version", model_version)
    
    return model_version

def notify_training_completion(**context):
    """Send notification about training completion"""
    ti = context['ti']
    model_version = ti.xcom_pull(task_ids='register_model')
    metrics = ti.xcom_pull(task_ids='evaluate_model')
    
    message = f"""
    ML Training Pipeline Completed Successfully!
    
    Model Version: {model_version}
    Accuracy: {metrics['accuracy']:.3f}
    F1 Score: {metrics['f1_score']:.3f}
    
    Model is ready for deployment review.
    """
    
    logging.info(message)
    # In real scenario, send to Slack, email, etc.
    
    return True

# Task definitions
data_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=check_data_quality,
    dag=dag
)

fetch_data = PythonOperator(
    task_id='fetch_training_data',
    python_callable=fetch_training_data,
    dag=dag
)

# Training task using Kubernetes Pod Operator
train_model = KubernetesPodOperator(
    task_id='train_model',
    name='ml-training-pod',
    namespace='mlops-system',
    image='your-registry/ml-service-training:latest',
    cmds=['python'],
    arguments=[
        'train/train_distilbert.py',
        '--epochs', '{{ var.value.training_epochs }}',
        '--batch-size', '{{ var.value.training_batch_size }}',
        '--learning-rate', '{{ var.value.training_learning_rate }}'
    ],
    env_vars={
        'MLFLOW_TRACKING_URI': '{{ var.value.mlflow_tracking_uri }}',
        'DATA_PATH': '/tmp/airflow_data/training_data.csv'
    },
    volumes=[],  # Add volume mounts for data access
    volume_mounts=[],
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

register_model_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag
)

# Log training metrics to database
log_training_metrics = PostgresOperator(
    task_id='log_training_metrics',
    postgres_conn_id='postgres_mlops',
    sql="""
    INSERT INTO training_runs (
        run_date, 
        model_version, 
        accuracy, 
        f1_score, 
        status
    ) VALUES (
        '{{ ds }}',
        '{{ ti.xcom_pull(task_ids="register_model") }}',
        {{ ti.xcom_pull(task_ids="evaluate_model")["accuracy"] }},
        {{ ti.xcom_pull(task_ids="evaluate_model")["f1_score"] }},
        'completed'
    );
    """,
    dag=dag
)

notify_completion = PythonOperator(
    task_id='notify_completion',
    python_callable=notify_training_completion,
    dag=dag
)

# Task dependencies
data_quality_check >> fetch_data >> train_model >> evaluate_model_task
evaluate_model_task >> register_model_task >> [log_training_metrics, notify_completion]