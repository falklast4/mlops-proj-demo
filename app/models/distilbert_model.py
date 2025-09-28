import asyncio
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Dict, Any
import mlflow.pytorch

from .base_model import BaseModel

class DistilBERTModel(BaseModel):
    """DistilBERT text classification model"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = 512
    
    async def load(self, model_path: str, model_version: str) -> None:
        """Load DistilBERT model"""
        def _load_model():
            try:
                # Try loading from MLflow first
                model_uri = f"models:/sentiment-classifier/{model_version}"
                self.model = mlflow.pytorch.load_model(model_uri)
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            except Exception:
                # Fallback to HuggingFace
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased', 
                    num_labels=2
                )
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            
            self.model.to(self.device)
            self.model.eval()
            self.model_version = model_version
        
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(
            self.executor, _load_model
        )
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make sentiment prediction"""
        text = input_data.get('text', '')
        if not text:
            raise ValueError("Text input is required")
        
        def _predict():
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = float(predictions[0][predicted_class])
            
            # Map to labels
            labels = {0: 'negative', 1: 'positive'}
            
            return {
                'sentiment': labels[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    'negative': float(predictions[0][0]),
                    'positive': float(predictions[0][1])
                }
            }
        
        # Run prediction in thread pool
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _predict
        )
        
        return result