from prometheus_client import Counter, Histogram, Gauge, Info
import time
import functools
from typing import Callable, Any

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status', 'model_version']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'model_version'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

ERROR_COUNT = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type']
)

# ML-specific metrics
ML_PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Total ML predictions made',
    ['model_type', 'model_version']
)

ML_PREDICTION_DURATION = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction duration in seconds',
    ['model_type', 'model_version'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

MODEL_LOAD_TIME = Histogram(
    'ml_model_load_duration_seconds',
    'Time taken to load ML model',
    ['model_type', 'model_version']
)

# System metrics
MODEL_MEMORY_USAGE = Gauge(
    'ml_model_memory_bytes',
    'Memory usage by ML model',
    ['model_type', 'model_version']
)

MODEL_VERSION_GAUGE = Gauge(
    'ml_model_version_info',
    'Information about current model version',
    ['model_type', 'model_version', 'build_info']
)

ACTIVE_REQUESTS = Gauge(
    'ml_active_requests',
    'Number of currently active requests'
)

# Model performance metrics
PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence',
    'Confidence scores of predictions',
    ['model_type', 'model_version'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

def track_predictions(model_type: str, model_version: str):
    """Decorator to track prediction metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            ACTIVE_REQUESTS.inc()
            
            try:
                result = await func(*args, **kwargs)
                
                # Track prediction
                ML_PREDICTIONS_TOTAL.labels(
                    model_type=model_type,
                    model_version=model_version
                ).inc()
                
                # Track confidence if available
                if isinstance(result, dict) and 'confidence' in result:
                    PREDICTION_CONFIDENCE.labels(
                        model_type=model_type,
                        model_version=model_version
                    ).observe(result['confidence'])
                
                return result
                
            finally:
                # Track duration
                duration = time.time() - start_time
                ML_PREDICTION_DURATION.labels(
                    model_type=model_type,
                    model_version=model_version
                ).observe(duration)
                
                ACTIVE_REQUESTS.dec()
        
        return wrapper
    return decorator

def setup_metrics():
    """Initialize metrics on startup"""
    # Set initial model info
    MODEL_VERSION_GAUGE.labels(
        model_type="distilbert",
        model_version="v1.0",
        build_info="production"
    ).set(1)

# Data drift detection metrics
DATA_DRIFT_SCORE = Gauge(
    'ml_data_drift_score',
    'Data drift detection score',
    ['feature_name', 'model_version']
)

PREDICTION_DISTRIBUTION = Histogram(
    'ml_prediction_distribution',
    'Distribution of prediction values',
    ['model_type', 'model_version', 'prediction_class']
)

# Business metrics
PREDICTION_BY_SENTIMENT = Counter(
    'ml_predictions_by_sentiment_total',
    'Total predictions by sentiment class',
    ['sentiment', 'model_version']
)

def track_business_metrics(prediction_result: dict, model_version: str):
    """Track business-specific metrics"""
    if 'sentiment' in prediction_result:
        PREDICTION_BY_SENTIMENT.labels(
            sentiment=prediction_result['sentiment'],
            model_version=model_version
        ).inc()