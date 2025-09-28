from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self):
        self.model = None
        self.model_version: Optional[str] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    @abstractmethod
    async def load(self, model_path: str, model_version: str) -> None:
        """Load the model"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction"""
        pass
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self):
        self.current_model: Optional[BaseModel] = None
        self.model_version: Optional[str] = None
    
    async def load_model(self, model_type: str = "distilbert"):
        """Load the specified model type"""
        if model_type == "distilbert":
            from .distilbert_model import DistilBERTModel
            self.current_model = DistilBERTModel()
        elif model_type == "sklearn":
            from .sklearn_model import SklearnModel
            self.current_model = SklearnModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load the model
        model_path = os.getenv("MODEL_PATH", "/app/models")
        model_version = os.getenv("MODEL_VERSION", "latest")
        
        await self.current_model.load(model_path, model_version)
        self.model_version = self.current_model.model_version
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction using current model"""
        if not self.current_model:
            raise RuntimeError("No model loaded")
        
        return await self.current_model.predict(input_data)
    
    def is_loaded(self) -> bool:
        """Check if a model is loaded"""
        return self.current_model is not None and self.current_model.is_loaded()