# MLOps Project Demo

Production-grade ML/LLM inference service with comprehensive MLOps pipeline.

[![CI](https://github.com/lawrencekiba/mlops-proj-demo/workflows/CI%20Pipeline/badge.svg)](https://github.com/lawrencekiba/mlops-proj-demo/actions)
[![Deploy Dev](https://github.com/lawrencekiba/mlops-proj-demo/workflows/Deploy%20to%20Development/badge.svg)](https://github.com/lawrencekiba/mlops-proj-demo/actions)
[![Deploy Prod](https://github.com/lawrencekiba/mlops-proj-demo/workflows/Promote%20to%20Production/badge.svg)](https://github.com/lawrencekiba/mlops-proj-demo/actions)

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Kubernetes cluster (Minikube/Kind for local)
- Helm 3.x
- Python 3.9+

### Local Development
```bash
# Clone repository
git clone https://github.com/lawrencekiba/mlops-proj-demo.git
cd mlops-proj-demo

# Start local environment
docker-compose up -d

# Wait for services to be ready
docker-compose logs -f ml-api

# Test the API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this service!"}'