#!/bin/bash

# MLOps Local Environment Setup Script
set -e

echo "Setting up MLOps local environment..."

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker is required but not installed."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo "ERROR: Docker Compose is required but not installed."
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo "WARNING: kubectl not found. Kubernetes deployment will not be available."
    fi
    
    if ! command -v helm &> /dev/null; then
        echo "WARNING: Helm not found. Helm deployment will not be available."
    fi
    
    echo "Prerequisites check completed"
}

# Setup Python environment
setup_python() {
    echo "Setting up Python environment..."
    
    if command -v python3.9 &> /dev/null; then
        PYTHON_CMD=python3.9
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    else
        echo "ERROR: Python 3.9+ is required"
        exit 1
    fi
    
    # Create virtual environment
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    echo "Python environment setup completed"
}

# Setup pre-commit hooks
setup_precommit() {
    echo "Setting up pre-commit hooks..."
    
    if source venv/bin/activate && command -v pre-commit &> /dev/null; then
        pre-commit install
        echo "Pre-commit hooks installed"
    else
        echo "WARNING: Pre-commit not available, skipping..."
    fi
}

# Build Docker images
build_images() {
    echo "Building Docker images..."
    
    # Build API image
    docker build -f docker/Dockerfile.api -t ml-service:latest .
    
    # Build training image
    docker build -f docker/Dockerfile.training -t ml-service-training:latest .
    
    echo "Docker images built successfully"
}

# Start local services
start_services() {
    echo "Starting local services..."
    
    # Create necessary directories
    mkdir -p logs models data
    
    # Start services with Docker Compose
    docker-compose up -d
    
    echo "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    echo "Waiting for PostgreSQL..."
    until docker-compose exec -T postgres pg_isready -U mlops_user -d mlops_db; do
        sleep 2
    done
    
    # Wait for MLflow
    echo "Waiting for MLflow..."
    until curl -f http://localhost:5000 &> /dev/null; do
        sleep 2
    done
    
    # Wait for ML API
    echo "Waiting for ML API..."
    until curl -f http://localhost:8000/healthz &> /dev/null; do
        sleep 2
    done
    
    echo "All services are ready"
}

# Run initial tests
run_tests() {
    echo "Running initial tests..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run unit tests
    pytest tests/unit/ -v --tb=short
    
    # Test API endpoints
    echo "Testing API endpoints..."
    
    # Health check
    curl -f http://localhost:8000/healthz || {
        echo "ERROR: Health check failed"
        exit 1
    }
    
    # Prediction test
    RESPONSE=$(curl -s -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a test message"}')
    
    echo "Prediction response: $RESPONSE"
    
    if echo "$RESPONSE" | grep -q "prediction"; then
        echo "Prediction endpoint working"
    else
        echo "ERROR: Prediction endpoint failed"
        exit 1
    fi
    
    echo "All tests passed"
}

# Setup Kubernetes (optional)
setup_kubernetes() {
    if ! command -v kubectl &> /dev/null || ! command -v helm &> /dev/null; then
        echo "WARNING: Kubernetes tools not available, skipping K8s setup"
        return
    fi
    
    echo "Setting up Kubernetes environment..."
    
    # Check if cluster is available
    if ! kubectl cluster-info &> /dev/null; then
        echo "WARNING: No Kubernetes cluster available, skipping K8s setup"
        return
    fi
    
    # Create namespace
    kubectl create namespace mlops-system --dry-run=client -o yaml | kubectl apply -f -
    
    # Install with Helm
    helm upgrade --install ml-service ./deploy/helm/ml-service \
        --namespace mlops-system \
        --set image.repository=ml-service \
        --set image.tag=latest \
        --set postgresql.enabled=true \
        --set redis.enabled=true \
        --wait
    
    echo "Kubernetes deployment completed"
    echo "Access the service:"
    echo "   kubectl port-forward svc/ml-service 8080:80 -n mlops-system"
}

# Display service URLs
show_urls() {
    echo ""
    echo "Service URLs:"
    echo "- ML API:          http://localhost:8000"
    echo "- Grafana:         http://localhost:3000"
    echo "- Prometheus:      http://localhost:9090"
    echo "- MLflow:          http://localhost:5000"
    echo ""
    echo "Default credentials:"
    echo "  Grafana: admin / admin123"
    echo ""
    echo "Quick test:"
    echo '  curl -X POST "http://localhost:8000/predict" \'
    echo '    -H "Content-Type: application/json" \'
    echo '    -d '\''{"text": "I love this service!"}'\'''
    echo ""
}

# Cleanup function
cleanup() {
    echo "Cleaning up..."
    docker-compose down
    echo "Cleanup completed"
}

# Main execution
main() {
    case "${1:-all}" in
        "prereq")
            check_prerequisites
            ;;
        "python")
            setup_python
            ;;
        "docker")
            build_images
            ;;
        "services")
            start_services
            ;;
        "test")
            run_tests
            ;;
        "k8s")
            setup_kubernetes
            ;;
        "cleanup")
            cleanup
            ;;
        "all")
            check_prerequisites
            setup_python
            setup_precommit
            build_images
            start_services
            run_tests
            setup_kubernetes
            show_urls
            ;;
        *)
            echo "Usage: $0 {prereq|python|docker|services|test|k8s|cleanup|all}"
            echo ""
            echo "Commands:"
            echo "  prereq   - Check prerequisites"
            echo "  python   - Setup Python environment"
            echo "  docker   - Build Docker images"
            echo "  services - Start local services"
            echo "  test     - Run tests"
            echo "  k8s      - Setup Kubernetes"
            echo "  cleanup  - Stop and cleanup"
            echo "  all      - Run complete setup"
            exit 1
            ;;
    esac
}

# Handle interruption
trap cleanup EXIT

# Run main function
main "$@"