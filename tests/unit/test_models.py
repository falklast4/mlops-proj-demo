import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import torch
from transformers import DistilBertTokenizer

from app.models.distilbert_model import DistilBERTModel
from app.models.base_model import ModelManager

class TestDistilBERTModel:
    """Test suite for DistilBERT model implementation"""
    
    @pytest.fixture
    def model(self):
        """Create a DistilBERT model instance for testing"""
        return DistilBERTModel()
    
    @pytest.mark.asyncio
    async def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.model is None
        assert model.tokenizer is None
        assert not model.is_loaded()
        assert model.device is not None
    
    @pytest.mark.asyncio
    async def test_model_loading_success(self, model):
        """Test successful model loading"""
        with patch('mlflow.pytorch.load_model') as mock_load_model, \
             patch('transformers.DistilBertTokenizer.from_pretrained') as mock_tokenizer:
            
            # Mock the model and tokenizer
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_load_model.return_value = mock_model
            mock_tokenizer.return_value = Mock()
            
            await model.load("/fake/path", "v1.0")
            
            assert model.is_loaded()
            assert model.model_version == "v1.0"
            mock_model.to.assert_called_once_with(model.device)
            mock_model.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_loading_fallback(self, model):
        """Test model loading with MLflow failure fallback"""
        with patch('mlflow.pytorch.load_model', side_effect=Exception("MLflow error")), \
             patch('transformers.DistilBertForSequenceClassification.from_pretrained') as mock_model_class, \
             patch('transformers.DistilBertTokenizer.from_pretrained') as mock_tokenizer:
            
            mock_model = Mock()
            mock_model.to.return_value = mock_model
            mock_model.eval.return_value = None
            mock_model_class.return_value = mock_model
            mock_tokenizer.return_value = Mock()
            
            await model.load("/fake/path", "v1.0")
            
            assert model.is_loaded()
            mock_model_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prediction_success(self, model):
        """Test successful prediction"""
        # Setup mocked model
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        # Mock tokenizer output
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer.return_value.to.return_value = mock_inputs
        
        # Mock model output
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[0.1, 0.9]])  # Positive prediction
        mock_model.return_value = mock_outputs
        
        model.model = mock_model
        model.tokenizer = mock_tokenizer
        model.model_version = "v1.0"
        
        result = await model.predict({"text": "This is great!"})
        
        assert "sentiment" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["sentiment"] in ["positive", "negative"]
        assert 0 <= result["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_prediction_empty_text(self, model):
        """Test prediction with empty text"""
        model.model = Mock()
        model.tokenizer = Mock()
        
        with pytest.raises(ValueError, match="Text input is required"):
            await model.predict({"text": ""})
    
    @pytest.mark.asyncio
    async def test_prediction_model_not_loaded(self, model):
        """Test prediction when model is not loaded"""
        with pytest.raises(AttributeError):
            await model.predict({"text": "test"})

class TestModelManager:
    """Test suite for ModelManager"""
    
    @pytest.fixture
    def manager(self):
        """Create a ModelManager instance for testing"""
        return ModelManager()
    
    @pytest.mark.asyncio
    async def test_model_manager_initialization(self, manager):
        """Test ModelManager initialization"""
        assert manager.current_model is None
        assert manager.model_version is None
        assert not manager.is_loaded()
    
    @pytest.mark.asyncio
    async def test_load_distilbert_model(self, manager):
        """Test loading DistilBERT model through manager"""
        with patch.dict('os.environ', {'MODEL_PATH': '/test/path', 'MODEL_VERSION': 'v1.0'}), \
             patch('app.models.distilbert_model.DistilBERTModel.load') as mock_load:
            
            mock_load.return_value = None
            await manager.load_model("distilbert")
            
            assert manager.current_model is not None
            assert manager.model_version == "v1.0"
            mock_load.assert_called_once_with('/test/path', 'v1.0')
    
    @pytest.mark.asyncio
    async def test_load_unknown_model_type(self, manager):
        """Test loading unknown model type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            await manager.load_model("unknown_model")
    
    @pytest.mark.asyncio
    async def test_predict_without_loaded_model(self, manager):
        """Test prediction without loaded model"""
        with pytest.raises(RuntimeError, match="No model loaded"):
            await manager.predict({"text": "test"})