from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.models import Variable
import logging

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'email': ['mlops-team@company.com']
}

dag = DAG(
    'ml_deployment_pipeline',
    default_args=default_args,
    description='ML Model Deployment Pipeline',
    schedule_interval=None,  # Triggered manually or by training pipeline
    catchup=False,
    tags=['ml', 'deployment', 'kubernetes']
)

def check_model_readiness(**context):
    """Check if model is ready for deployment"""
    model_version = Variable.get("latest_model_version")
    
    if not model_version:
        raise ValueError("No model version available for deployment")
    
    # Additional checks:
    # 1. Model exists in MLflow
    # 2. Model passes validation tests
    # 3. Model meets performance criteria
    
    logging.info(f"Model {model_version} is ready for deployment")
    return model_version

def run_model_validation(**context):
    """Run comprehensive model validation"""
    ti = context['ti']
    model_version = ti.xcom_pull(task_ids='check_model_readiness')
    
    # Validation tests:
    # 1. Schema validation
    # 2. Performance benchmarks
    # 3. Bias testing
    # 4. Security scans
    
    logging.info(f"Running validation for model {model_version}")
    
    # Simulate validation results
    validation_results = {
        'schema_valid': True,
        'performance_acceptable': True,
        'bias_check_passed': True,
        'security_scan_passed': True
    }
    
    if not all(validation_results.values()):
        raise ValueError(f"Model validation failed: {validation_results}")
    
    logging.info("Model validation passed!")
    return validation_results

def deploy_to_staging(**context):
    """Deploy model to staging environment"""
    ti = context['ti']
    model_version = ti.xcom_pull(task_ids='check_model_readiness')
    
    logging.info(f"Deploying model {model_version} to staging")
    
    # Update staging deployment with new model version
    # This would typically use kubectl or Helm
    
    return f"staging-deployment-{model_version}"

def run_integration_tests(**context):
    """Run integration tests against staging deployment"""
    ti = context['ti']
    deployment_id = ti.xcom_pull(task_ids='deploy_to_staging')
    
    logging.info(f"Running integration tests for {deployment_id}")
    
    # Integration tests:
    # 1. API endpoint tests
    # 2. Load testing
    # 3. End-to-end workflow tests
    
    import requests
    import time
    
    # Wait for deployment to be ready
    time.sleep(30)
    
    # Test staging endpoint
    staging_url = "http://ml-service-staging.mlops-system.svc.cluster.local"
    
    test_cases = [
        {"text": "This is a positive test"},
        {"text": "This is a negative test"},
        {"text": "Edge case with special characters !@#$%"}
    ]
    
    for i, test_case in enumerate(test_cases):
        try:
            response = requests.post(
                f"{staging_url}/predict",
                json=test_case,
                timeout=10
            )
            
            if response.status_code != 200:
                raise ValueError(f"Test case {i} failed with status {response.status_code}")
            
            result = response.json()
            required_fields = ['prediction', 'latency_ms', 'model_version']
            
            if not all(field in result for field in required_fields):
                raise ValueError(f"Test case {i} missing required fields")
            
            logging.info(f"Test case {i} passed: {result}")
            
        except Exception as e:
            raise ValueError(f"Integration test {i} failed: {str(e)}")
    
    logging.info("All integration tests passed!")
    return True

def approve_production_deployment(**context):
    """Manual approval step for production deployment"""
    # In real scenario, this would:
    # 1. Send notification to stakeholders
    # 2. Wait for manual approval
    # 3. Check business rules
    
    logging.info("Production deployment approved (simulated)")
    return True

def deploy_to_production(**context):
    """Deploy model to production with blue-green strategy"""
    ti = context['ti']
    model_version = ti.xcom_pull(task_ids='check_model_readiness')
    
    logging.info(f"Deploying model {model_version} to production")
    
    # Implement blue-green deployment
    # 1. Determine current active deployment (blue/green)
    # 2. Deploy to inactive environment
    # 3. Run smoke tests
    # 4. Switch traffic
    
    return f"production-deployment-{model_version}"

def run_production_smoke_tests(**context):
    """Run smoke tests against production deployment"""
    ti = context['ti']
    deployment_id = ti.xcom_pull(task_ids='deploy_to_production')
    
    logging.info(f"Running production smoke tests for {deployment_id}")
    
    # Smoke tests for production
    production_url = "http://ml-service.mlops-system.svc.cluster.local"
    
    # Health check
    response = requests.get(f"{production_url}/healthz", timeout=5)
    if response.status_code != 200:
        raise ValueError("Production health check failed")
    
    # Quick prediction test
    test_payload = {"text": "Production smoke test"}
    response = requests.post(
        f"{production_url}/predict",
        json=test_payload,
        timeout=10
    )
    
    if response.status_code != 200:
        raise ValueError("Production prediction test failed")
    
    result = response.json()
    if 'prediction' not in result:
        raise ValueError("Production prediction missing required fields")
    
    logging.info("Production smoke tests passed!")
    return True

# Task definitions
wait_for_training = ExternalTaskSensor(
    task_id='wait_for_training_completion',
    external_dag_id='ml_training_pipeline',
    external_task_id='notify_completion',
    timeout=3600,  # 1 hour timeout
    poke_interval=300,  # Check every 5 minutes
    dag=dag
)

check_model_readiness_task = PythonOperator(
    task_id='check_model_readiness',
    python_callable=check_model_readiness,
    dag=dag
)

validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=run_model_validation,
    dag=dag
)

deploy_staging = KubernetesPodOperator(
    task_id='deploy_to_staging',
    name='staging-deployment-pod',
    namespace='mlops-system',
    image='bitnami/kubectl:latest',
    cmds=['kubectl'],
    arguments=[
        'set', 'image',
        'deployment/ml-service-staging',
        f'ml-service=your-registry/ml-service:{{ ti.xcom_pull(task_ids="check_model_readiness") }}',
        '-n', 'mlops-staging'
    ],
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag
)

integration_tests = PythonOperator(
    task_id='run_integration_tests',
    python_callable=run_integration_tests,
    dag=dag
)

approval_gate = PythonOperator(
    task_id='approve_production_deployment',
    python_callable=approve_production_deployment,
    dag=dag
)

deploy_production = KubernetesPodOperator(
    task_id='deploy_to_production',
    name='production-deployment-pod',
    namespace='mlops-system',
    image='bitnami/kubectl:latest',
    cmds=['bash'],
    arguments=[
        '-c',
        '''
        # Determine current active deployment
        CURRENT=$(kubectl get ingress ml-service-main -n mlops-prod -o jsonpath='{.spec.rules[0].http.paths[0].backend.service.name}')
        
        if [ "$CURRENT" = "ml-service-blue" ]; then
            TARGET="green"
        else
            TARGET="blue"
        fi
        
        # Deploy to target environment
        kubectl set image deployment/ml-service-$TARGET ml-service={{ params.image_tag }} -n mlops-prod
        kubectl rollout status deployment/ml-service-$TARGET -n mlops-prod --timeout=600s
        
        # Switch traffic (blue-green)
        kubectl patch ingress ml-service-main -n mlops-prod --type='json' -p='[
          {
            "op": "replace",
            "path": "/spec/rules/0/http/paths/0/backend/service/name",
            "value": "ml-service-'$TARGET'"
          }
        ]'
        '''
    ],
    params={'image_tag': '{{ ti.xcom_pull(task_ids="check_model_readiness") }}'},
    is_delete_operator_pod=True,
    get_logs=True,
    dag=dag
)

production_smoke_tests = PythonOperator(
    task_id='run_production_smoke_tests',
    python_callable=run_production_smoke_tests,
    dag=dag
)

# Final notification
notify_deployment_success = BashOperator(
    task_id='notify_deployment_success',
    bash_command='''
    echo "Model deployment completed successfully"
    echo "Model Version: {{ ti.xcom_pull(task_ids='check_model_readiness') }}"
    echo "Deployment Time: {{ ts }}"
    # Send to Slack, email, etc.
    ''',
    dag=dag
)

# Task dependencies
wait_for_training >> check_model_readiness_task >> validate_model
validate_model >> deploy_staging >> integration_tests
integration_tests >> approval_gate >> deploy_production
deploy_production >> production_smoke_tests >> notify_deployment_success