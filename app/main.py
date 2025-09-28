from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
import time
import uuid
from typing import Dict, Any
import logging

from .models.base_model import ModelManager
from .schemas.prediction import PredictionRequest, PredictionResponse
from .database import get_db, save_prediction_log
from .metrics import (
    REQUEST_COUNT, REQUEST_DURATION, ERROR_COUNT, 
    MODEL_VERSION_GAUGE, setup_metrics
)
from .config import settings

# Initialize FastAPI app
app = FastAPI(
    title="ML/LLM Inference Service",
    description="Production-grade ML/LLM inference with MLOps",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    await model_manager.load_model()
    setup_metrics()
    logging.info(f"Service started with model version: {model_manager.model_version}")

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "model_version": model_manager.model_version,
        "timestamp": time.time()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    db = Depends(get_db)
):
    """Main prediction endpoint"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Increment request counter
        REQUEST_COUNT.labels(
            method="POST", 
            endpoint="/predict",
            model_version=model_manager.model_version
        ).inc()
        
        # Make prediction
        prediction = await model_manager.predict(request.dict())
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Create response
        response = PredictionResponse(
            request_id=request_id,
            prediction=prediction,
            latency_ms=round(latency_ms, 2),
            model_version=model_manager.model_version,
            timestamp=time.time()
        )
        
        # Log to database
        await save_prediction_log(
            db=db,
            request_id=request_id,
            model_version=model_manager.model_version,
            latency_ms=latency_ms,
            input_data=request.dict(),
            prediction=prediction
        )
        
        # Record metrics
        REQUEST_DURATION.labels(
            method="POST",
            endpoint="/predict",
            model_version=model_manager.model_version
        ).observe(latency_ms / 1000)
        
        return response
        
    except Exception as e:
        # Increment error counter
        ERROR_COUNT.labels(
            method="POST",
            endpoint="/predict",
            error_type=type(e).__name__
        ).inc()
        
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))