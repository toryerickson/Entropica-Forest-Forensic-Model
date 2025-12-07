# =============================================================================
# EFM DEPLOYMENT CONFIGURATION
# =============================================================================
# 
# This directory contains deployment configurations for the EFM system.
# Target: Kubernetes cluster (simulated via Docker Compose for development)
#
# Architecture:
# - efm-orchestrator: The control plane (FastAPI service)
# - efm-agents: Simulated AI agents (can be replaced with real agents)
# - redis: State store for production (optional)
# - prometheus: Metrics collection (optional)
#
# =============================================================================

# -----------------------------------------------------------------------------
# docker-compose.yml
# -----------------------------------------------------------------------------
# 
# Usage:
#   docker-compose up -d          # Start all services
#   docker-compose logs -f        # View logs
#   docker-compose down           # Stop all services
#
# version: "3.8"
# 
# services:
#   efm-orchestrator:
#     build:
#       context: .
#       dockerfile: Dockerfile.orchestrator
#     ports:
#       - "8000:8000"
#     environment:
#       - EFM_NODE_ID=orchestrator_primary
#       - EFM_LOG_LEVEL=INFO
#       - REDIS_URL=redis://redis:6379
#     depends_on:
#       - redis
#     healthcheck:
#       test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
#       interval: 10s
#       timeout: 5s
#       retries: 3
# 
#   efm-agent-simulator:
#     build:
#       context: .
#       dockerfile: Dockerfile.agent
#     environment:
#       - ORCHESTRATOR_URL=http://efm-orchestrator:8000
#       - NUM_AGENTS=50
#       - TICK_INTERVAL_MS=100
#     depends_on:
#       - efm-orchestrator
# 
#   redis:
#     image: redis:7-alpine
#     ports:
#       - "6379:6379"
#     volumes:
#       - redis_data:/data
# 
#   prometheus:
#     image: prom/prometheus:latest
#     ports:
#       - "9090:9090"
#     volumes:
#       - ./prometheus.yml:/etc/prometheus/prometheus.yml
# 
# volumes:
#   redis_data:
#
# -----------------------------------------------------------------------------

"""
Kubernetes Deployment Manifests
================================

Below are the K8s manifests for production deployment.
Save each section to a separate YAML file.
"""

NAMESPACE_YAML = """
apiVersion: v1
kind: Namespace
metadata:
  name: efm-system
  labels:
    app.kubernetes.io/name: efm
    app.kubernetes.io/component: governance
"""

ORCHESTRATOR_DEPLOYMENT_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: efm-orchestrator
  namespace: efm-system
  labels:
    app: efm-orchestrator
spec:
  replicas: 3  # HA setup
  selector:
    matchLabels:
      app: efm-orchestrator
  template:
    metadata:
      labels:
        app: efm-orchestrator
    spec:
      containers:
      - name: orchestrator
        image: efm/orchestrator:latest
        ports:
        - containerPort: 8000
        env:
        - name: EFM_NODE_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: EFM_LOG_LEVEL
          value: "INFO"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: efm-secrets
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /status
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
"""

ORCHESTRATOR_SERVICE_YAML = """
apiVersion: v1
kind: Service
metadata:
  name: efm-orchestrator
  namespace: efm-system
spec:
  selector:
    app: efm-orchestrator
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: v1
kind: Service
metadata:
  name: efm-orchestrator-external
  namespace: efm-system
spec:
  selector:
    app: efm-orchestrator
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
"""

AGENT_SIDECAR_YAML = """
# This sidecar is injected into agent pods for d-CTM local consensus
apiVersion: v1
kind: ConfigMap
metadata:
  name: efm-sidecar-config
  namespace: efm-system
data:
  sidecar.yaml: |
    orchestrator_url: http://efm-orchestrator:8000
    local_consensus:
      enabled: true
      timeout_ms: 500
    phi_extraction:
      method: transformer_v_mean
      dimensions: 48
      normalize: true
    reporting:
      interval_ms: 100
      batch_size: 10
"""

HPA_YAML = """
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: efm-orchestrator-hpa
  namespace: efm-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: efm-orchestrator
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""


# =============================================================================
# DOCKERFILE CONTENTS
# =============================================================================

DOCKERFILE_ORCHESTRATOR = """
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "src.efm_orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKERFILE_AGENT = """
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Run agent simulator
CMD ["python", "-m", "src.agent_simulator"]
"""

REQUIREMENTS_TXT = """
# EFM Dependencies
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
numpy>=1.24.0
httpx>=0.24.0
redis>=4.5.0
prometheus-client>=0.17.0
"""


# =============================================================================
# DEPLOYMENT HELPER
# =============================================================================

def generate_deployment_files(output_dir: str = "/home/claude/efm-booklet4/deploy"):
    """Generate all deployment configuration files"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/k8s", exist_ok=True)
    
    # Write K8s manifests
    manifests = {
        "namespace.yaml": NAMESPACE_YAML,
        "orchestrator-deployment.yaml": ORCHESTRATOR_DEPLOYMENT_YAML,
        "orchestrator-service.yaml": ORCHESTRATOR_SERVICE_YAML,
        "agent-sidecar-config.yaml": AGENT_SIDECAR_YAML,
        "hpa.yaml": HPA_YAML,
    }
    
    for filename, content in manifests.items():
        with open(f"{output_dir}/k8s/{filename}", 'w') as f:
            f.write(content.strip())
    
    # Write Dockerfiles
    with open(f"{output_dir}/Dockerfile.orchestrator", 'w') as f:
        f.write(DOCKERFILE_ORCHESTRATOR.strip())
    
    with open(f"{output_dir}/Dockerfile.agent", 'w') as f:
        f.write(DOCKERFILE_AGENT.strip())
    
    # Write requirements
    with open(f"{output_dir}/requirements.txt", 'w') as f:
        f.write(REQUIREMENTS_TXT.strip())
    
    # Write docker-compose
    docker_compose = """
version: "3.8"

services:
  efm-orchestrator:
    build:
      context: ..
      dockerfile: deploy/Dockerfile.orchestrator
    ports:
      - "8000:8000"
    environment:
      - EFM_NODE_ID=orchestrator_primary
      - EFM_LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  efm-agent-simulator:
    build:
      context: ..
      dockerfile: deploy/Dockerfile.agent
    environment:
      - ORCHESTRATOR_URL=http://efm-orchestrator:8000
      - NUM_AGENTS=10
      - TICK_INTERVAL_MS=500
    depends_on:
      efm-orchestrator:
        condition: service_healthy
"""
    
    with open(f"{output_dir}/docker-compose.yml", 'w') as f:
        f.write(docker_compose.strip())
    
    print(f"Generated deployment files in {output_dir}/")
    print(f"  - k8s/: Kubernetes manifests")
    print(f"  - Dockerfile.orchestrator")
    print(f"  - Dockerfile.agent")
    print(f"  - requirements.txt")
    print(f"  - docker-compose.yml")
    
    return output_dir


if __name__ == "__main__":
    generate_deployment_files()
