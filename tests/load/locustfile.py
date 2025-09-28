from locust import HttpUser, task, between
import json
import random

class MLServiceUser(HttpUser):
    """Load testing user for ML service"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup test data"""
        self.test_texts = [
            "I love this product, it's amazing!",
            "This service is terrible and slow.",
            "Great quality and fast delivery!",
            "Could be better, not satisfied.",
            "Excellent customer support team.",
            "Worst experience I've ever had.",
            "Highly recommend to everyone!",
            "Complete waste of money and time.",
            "Perfect for my needs, very happy.",
            "Poor quality, returned immediately."
        ]
    
    @task(10)
    def predict_sentiment(self):
        """Main prediction task"""
        text = random.choice(self.test_texts)
        payload = {"text": text}
        
        with self.client.post("/predict", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data and "latency_ms" in data:
                    # Check latency
                    latency = data["latency_ms"]
                    if latency > 2000:  # 2 second threshold
                        response.failure(f"High latency: {latency}ms")
                    else:
                        response.success()
                else:
                    response.failure("Missing required fields in response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check task"""
        with self.client.get("/healthz", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Service not healthy")
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def metrics_check(self):
        """Check metrics endpoint"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics endpoint failed: {response.status_code}")

class StressTestUser(HttpUser):
    """Stress test user with higher load"""
    
    wait_time = between(0.1, 0.5)  # Much shorter wait time
    
    def on_start(self):
        self.test_text = "Stress test message for high load testing"
    
    @task
    def stress_predict(self):
        """High-frequency prediction requests"""
        payload = {"text": self.test_text}
        
        with self.client.post("/predict", 
                             json=payload,
                             catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Request failed: {response.status_code}")
            elif response.elapsed.total_seconds() > 5:
                response.failure(f"Request too slow: {response.elapsed.total_seconds()}s")
            else:
                response.success()