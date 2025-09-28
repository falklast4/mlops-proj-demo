-- Initial database setup for MLOps system
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Prediction logs table
CREATE TABLE prediction_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(255) NOT NULL UNIQUE,
    model_version VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL DEFAULT 'distilbert',
    input_text TEXT NOT NULL,
    prediction JSONB NOT NULL,
    confidence FLOAT,
    latency_ms FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_agent TEXT,
    ip_address INET,
    
    -- Indexes for performance
    INDEX idx_prediction_logs_created_at ON prediction_logs(created_at),
    INDEX idx_prediction_logs_model_version ON prediction_logs(model_version),
    INDEX idx_prediction_logs_latency ON prediction_logs(latency_ms),
    INDEX idx_prediction_logs_confidence ON prediction_logs(confidence)
);

-- Training runs table
CREATE TABLE training_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_date DATE NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL DEFAULT 'distilbert',
    dataset_version VARCHAR(100),
    hyperparameters JSONB,
    metrics JSONB,
    accuracy FLOAT,
    f1_score FLOAT,
    precision_score FLOAT,
    recall_score FLOAT,
    training_duration_seconds INTEGER,
    status VARCHAR(20) DEFAULT 'running',
    error_message TEXT,
    artifact_location TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT check_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    
    -- Indexes
    INDEX idx_training_runs_status ON training_runs(status),
    INDEX idx_training_runs_model_version ON training_runs(model_version),
    INDEX idx_training_runs_created_at ON training_runs(created_at),
    INDEX idx_training_runs_accuracy ON training_runs(accuracy)
);

-- Model deployments table
CREATE TABLE model_deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(100) NOT NULL,
    environment VARCHAR(50) NOT NULL,
    deployment_strategy VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'deploying',
    traffic_percentage INTEGER DEFAULT 0,
    health_check_url TEXT,
    deployed_by VARCHAR(100),
    deployment_config JSONB,
    rollback_version VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    activated_at TIMESTAMP WITH TIME ZONE,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT check_environment CHECK (environment IN ('development', 'staging', 'production')),
    CONSTRAINT check_deployment_status CHECK (status IN ('deploying', 'active', 'inactive', 'failed', 'rolled_back')),
    CONSTRAINT check_traffic_percentage CHECK (traffic_percentage >= 0 AND traffic_percentage <= 100),
    
    -- Indexes
    INDEX idx_deployments_environment ON model_deployments(environment),
    INDEX idx_deployments_status ON model_deployments(status),
    INDEX idx_deployments_model_version ON model_deployments(model_version),
    INDEX idx_deployments_created_at ON model_deployments(created_at)
);

-- Model performance monitoring
CREATE TABLE model_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(100) NOT NULL,
    metric_date DATE NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    avg_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    error_rate FLOAT,
    avg_confidence FLOAT,
    prediction_distribution JSONB,
    drift_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint for one record per model per date
    UNIQUE(model_version, metric_date),
    
    -- Indexes
    INDEX idx_performance_metrics_date ON model_performance_metrics(metric_date),
    INDEX idx_performance_metrics_model ON model_performance_metrics(model_version),
    INDEX idx_performance_metrics_drift ON model_performance_metrics(drift_score)
);

-- Data drift detection
CREATE TABLE data_drift_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(100) NOT NULL,
    report_date DATE NOT NULL,
    drift_detected BOOLEAN DEFAULT FALSE,
    drift_score FLOAT,
    feature_drifts JSONB,
    reference_period_start DATE,
    reference_period_end DATE,
    current_period_start DATE,
    current_period_end DATE,
    drift_threshold FLOAT DEFAULT 0.1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    INDEX idx_drift_reports_date ON data_drift_reports(report_date),
    INDEX idx_drift_reports_model ON data_drift_reports(model_version),
    INDEX idx_drift_reports_detected ON data_drift_reports(drift_detected)
);

-- Alerts and notifications
CREATE TABLE system_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    model_version VARCHAR(100),
    environment VARCHAR(50),
    metric_value FLOAT,
    threshold_value FLOAT,
    status VARCHAR(20) DEFAULT 'active',
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT check_alert_severity CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT check_alert_status CHECK (status IN ('active', 'acknowledged', 'resolved', 'suppressed')),
    
    -- Indexes
    INDEX idx_alerts_status ON system_alerts(status),
    INDEX idx_alerts_severity ON system_alerts(severity),
    INDEX idx_alerts_created_at ON system_alerts(created_at),
    INDEX idx_alerts_model_version ON system_alerts(model_version)
);

-- Create views for common queries
CREATE VIEW model_performance_summary AS
SELECT 
    model_version,
    COUNT(*) as total_predictions,
    AVG(latency_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
    AVG(confidence) as avg_confidence,
    DATE_TRUNC('day', created_at) as prediction_date
FROM prediction_logs 
GROUP BY model_version, DATE_TRUNC('day', created_at);

CREATE VIEW active_deployments AS
SELECT 
    model_version,
    environment,
    deployment_strategy,
    traffic_percentage,
    activated_at,
    deployed_by
FROM model_deployments 
WHERE status = 'active'
ORDER BY environment, activated_at DESC;

-- Function to calculate daily model metrics
CREATE OR REPLACE FUNCTION calculate_daily_metrics(target_date DATE DEFAULT CURRENT_DATE)
RETURNS VOID AS $$
BEGIN
    INSERT INTO model_performance_metrics (
        model_version,
        metric_date,
        total_predictions,
        avg_latency_ms,
        p95_latency_ms,
        p99_latency_ms,
        error_rate,
        avg_confidence,
        prediction_distribution
    )
    SELECT 
        model_version,
        target_date,
        COUNT(*) as total_predictions,
        AVG(latency_ms) as avg_latency_ms,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency_ms,
        0.0 as error_rate, -- Calculate from logs if error tracking is implemented
        AVG(confidence) as avg_confidence,
        jsonb_build_object(
            'positive', COUNT(*) FILTER (WHERE prediction->>'sentiment' = 'positive'),
            'negative', COUNT(*) FILTER (WHERE prediction->>'sentiment' = 'negative')
        ) as prediction_distribution
    FROM prediction_logs 
    WHERE DATE(created_at) = target_date
    GROUP BY model_version
    ON CONFLICT (model_version, metric_date) 
    DO UPDATE SET
        total_predictions = EXCLUDED.total_predictions,
        avg_latency_ms = EXCLUDED.avg_latency_ms,
        p95_latency_ms = EXCLUDED.p95_latency_ms,
        p99_latency_ms = EXCLUDED.p99_latency_ms,
        avg_confidence = EXCLUDED.avg_confidence,
        prediction_distribution = EXCLUDED.prediction_distribution;
END;
$ LANGUAGE plpgsql;

-- Create a trigger to automatically calculate metrics
CREATE OR REPLACE FUNCTION trigger_daily_metrics()
RETURNS TRIGGER AS $
BEGIN
    -- Schedule metric calculation for the date of the new prediction
    PERFORM calculate_daily_metrics(DATE(NEW.created_at));
    RETURN NEW;
END;
$ LANGUAGE plpgsql;

-- Trigger on prediction_logs insert (with some delay to batch updates)
-- In production, this would be better handled by a scheduled job