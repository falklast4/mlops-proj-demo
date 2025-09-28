import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
import json

from app.main import app

class TestAPIIntegration:
    """Integration tests for the FastAPI application"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data
        assert "model_version" in data
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_success(self, async_client):
        """Test successful prediction"""
        with patch('app.models.base_model.ModelManager.predict') as mock_predict:
            mock_predict.return_value = {
                "sentiment": "positive",
                "confidence": 0.95,
                "probabilities": {"positive": 0.95, "negative": 0.05}
            }
            
            payload = {"text": "I love this product!"}
            response = await async_client.post("/predict", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "request_id" in data
            assert "prediction" in data
            assert "latency_ms" in data
            assert "model_version" in data
            assert "timestamp" in data
            
            assert data["prediction"]["sentiment"] == "positive"
            assert data["latency_ms"] >= 0
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_validation_error(self, async_client):
        """Test prediction with validation error"""
        # Test missing text field
        payload = {}
        response = await async_client.post("/predict", json=payload)
        assert response.status_code == 422
        
        # Test empty text
        payload = {"text": ""}
        response = await async_client.post("/predict", json=payload)
        assert response.status_code in [400, 422]
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_model_error(self, async_client):
        """Test prediction with model error"""
        with patch('app.models.base_model.ModelManager.predict', side_effect=Exception("Model error")):
            payload = {"text": "Test text"}
            response = await async_client.post("/predict", json=payload)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        
        # Check for some expected metrics
        content = response.text
        assert "http_requests_total" in content or "# HELP" in content
    
    @pytest.mark.asyncio
    async def test_predict_endpoint_load_test(self, async_client):
        """Test prediction endpoint under load"""
        with patch('app.models.base_model.ModelManager.predict') as mock_predict:
            mock_predict.return_value = {
                "sentiment": "positive",
                "confidence": 0.8,
                "probabilities": {"positive": 0.8, "negative": 0.2}
            }
            
            # Send multiple concurrent requests
            tasks = []
            for i in range(10):
                payload = {"text": f"Test message {i}"}
                task = async_client.post("/predict", json=payload)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert "prediction" in data
                assert "latency_ms" in data