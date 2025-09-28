from pydantic import BaseModel, ConfigDict
from typing import Dict, Any

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=()) 
    
    request_id: str
    prediction: Dict[str, Any]
    latency_ms: float
    model_version: str
    timestamp: float