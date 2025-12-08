# EFM Use Case Templates

This document provides deployment templates for integrating the Entropica Forensic Model into various system architectures.

---

## Use Case A: Robotic Swarm (Search and Rescue)

### Goal

Autonomously coordinate hundreds of agents with local cognition, high-fidelity forensic logging, and regulatory trust.

### System Context

| Component | Role |
|-----------|------|
| **Capsule Node** | Each robot runs a capsule for self-monitoring |
| **EFMCore (Edge)** | Local governance and drift detection |
| **d-CTM Cluster** | Edge cluster aggregates local state |
| **ZK-SP Layer** | Aggregates proofs for mission audits |

### Why EFM?

- **Capsule-level symbolic failure capture**: No need for external observer
- **Immune response**: PARTITION/ESCALATE prevents cascade errors
- **Verifiable autonomy**: ZK-SP ensures agents follow internal rules
- **Hereditary tracking**: Parent-child robot spawning maintains lineage

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MISSION CONTROL (Cloud)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ ZK-SP Agg   │  │ Global CTM  │  │ Audit Log   │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    EDGE CLUSTER (Field)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    d-CTM Nodes                       │    │
│  │  [Node 1] ◄──► [Node 2] ◄──► [Node 3]              │    │
│  │      │             │             │                   │    │
│  │      ▼             ▼             ▼                   │    │
│  │  ┌───────┐    ┌───────┐    ┌───────┐               │    │
│  │  │EFMCore│    │EFMCore│    │EFMCore│               │    │
│  │  └───┬───┘    └───┬───┘    └───┬───┘               │    │
│  └──────┼────────────┼────────────┼─────────────────────┘    │
└─────────┼────────────┼────────────┼─────────────────────────┘
          │            │            │
          ▼            ▼            ▼
     ┌────────┐   ┌────────┐   ┌────────┐
     │Robot 1 │   │Robot 2 │   │Robot 3 │  ... Robot N
     │Capsule │   │Capsule │   │Capsule │
     └────────┘   └────────┘   └────────┘
```

### Configuration

```python
ROBOTIC_SWARM_CONFIG = {
    "deployment": "robotic_swarm",
    "phi_dimensions": 32,
    "tick_interval_ms": 50,
    "max_agents": 10000,
    "phi_source": "state_action_embedding",
    "hardware": {
        "cpu": "NVIDIA Jetson AGX Orin",
        "memory": "32GB",
        "zk_acceleration": "GPU (CUDA)"
    },
    "network": {
        "protocol": "mesh",
        "latency_max_ms": 100,
        "redundancy": "multi-path"
    }
}
```

### Key Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Drift detection latency | < 100ms | Real-time response required |
| Consensus timeout | 500ms | Field conditions vary |
| Proof generation | < 1s | Mission logging |
| Agent failure recovery | < 5s | Safety critical |

---

## Use Case B: Multi-Agent LLM Audit

### Goal

Monitor multiple LLM instances for drift, bias emergence, and policy violations with cryptographic audit trails.

### System Context

| Component | Role |
|-----------|------|
| **LLM Instance** | Production model serving requests |
| **EFMCore Sidecar** | Monitors hidden states per generation |
| **d-CTM Cluster** | Correlates drift across instances |
| **ZK-SP Layer** | Provides regulatory compliance proofs |

### Why EFM?

- **Hidden state monitoring**: Φ extracted from transformer layers
- **Bias drift detection**: TPE identifies gradual shifts
- **Regulatory compliance**: ZK proofs for auditors without exposing model weights
- **Multi-model correlation**: Detect systematic drift across fleet

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLIANCE LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Regulator   │  │ Audit Trail │  │ Alert Dash  │         │
│  │ Interface   │  │ (Immutable) │  │             │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    GOVERNANCE LAYER                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                 Global ZK-SP Aggregator              │    │
│  └─────────────────────────┬───────────────────────────┘    │
│                            │                                 │
│  ┌─────────────────────────┴───────────────────────────┐    │
│  │                    d-CTM Cluster                     │    │
│  │     [EFM Node 1] ◄──► [EFM Node 2] ◄──► [EFM Node 3]│    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   LLM + EFM  │  │   LLM + EFM  │  │   LLM + EFM  │       │
│  │   Sidecar    │  │   Sidecar    │  │   Sidecar    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Φ Extraction Pipeline

```python
LLM_OVERLAY_PHI_PIPELINE = {
    "input": "LLM hidden states from selected layers",
    "preprocessing": [
        "layer_selection",      # Choose layers 12, 18, 24
        "attention_pooling",    # Pool attention heads
        "normalize"             # L2 normalization
    ],
    "embedding": {
        "method": "linear_projection",
        "input_dim": 4096,      # Model hidden dim
        "output_dim": 128       # EFM Φ dimensions
    },
    "output": "128D vector per generation step",
    "update_frequency": "per_token or per_response"
}
```

### Configuration

```python
LLM_AUDIT_CONFIG = {
    "deployment": "llm_overlay",
    "phi_dimensions": 128,
    "tick_interval_ms": 1000,  # Per response
    "max_agents": 50000,       # LLM instances
    "phi_source": "llm_hidden_state",
    "hardware": {
        "cpu": "Cloud instance 32+ vCPUs",
        "memory": "256GB+ (colocated with LLM)",
        "zk_acceleration": "ASIC clusters"
    },
    "compliance": {
        "audit_retention_days": 365,
        "proof_generation": "hourly",
        "regulator_access": "read-only API"
    }
}
```

---

## Use Case C: Autonomous Finance Agents

### Goal

Deploy trading agents with real-time risk monitoring, regulatory compliance, and Byzantine fault tolerance.

### System Context

| Component | Role |
|-----------|------|
| **Trading Agent** | Executes trades based on strategy |
| **EFMCore** | Monitors decision drift and risk |
| **d-CTM Cluster** | Consensus on market state |
| **ZK-SP Layer** | Regulatory audit trail |

### Why EFM?

- **Decision drift detection**: Catch strategy drift before losses
- **Hereditary risk**: Track cascading failures from model updates
- **Regulatory proof**: Demonstrate compliance without revealing strategy
- **Byzantine tolerance**: Handle malicious or faulty agents

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REGULATORY INTERFACE                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ SEC/FINRA   │  │ Risk Mgmt   │  │ Compliance  │         │
│  │ Reporting   │  │ Dashboard   │  │ Officer     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                    GOVERNANCE CLUSTER                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    d-CTM + ZK-SP (High Availability, FPGA Accel)    │    │
│  │    3+ nodes, Byzantine fault tolerant               │    │
│  └─────────────────────────┬───────────────────────────┘    │
└─────────────────────────────┼───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Trading Agent 1 │ │  Trading Agent 2 │ │  Trading Agent N │
│  ┌────────────┐  │ │  ┌────────────┐  │ │  ┌────────────┐  │
│  │  Strategy  │  │ │  │  Strategy  │  │ │  │  Strategy  │  │
│  │  Engine    │  │ │  │  Engine    │  │ │  │  Engine    │  │
│  └─────┬──────┘  │ │  └─────┬──────┘  │ │  └─────┬──────┘  │
│        │         │ │        │         │ │        │         │
│  ┌─────▼──────┐  │ │  ┌─────▼──────┐  │ │  ┌─────▼──────┐  │
│  │  EFMCore   │  │ │  │  EFMCore   │  │ │  │  EFMCore   │  │
│  │  Sidecar   │  │ │  │  Sidecar   │  │ │  │  Sidecar   │  │
│  └────────────┘  │ │  └────────────┘  │ │  └────────────┘  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

### Configuration

```python
FINTECH_CONFIG = {
    "deployment": "fintech",
    "phi_dimensions": 64,
    "tick_interval_ms": 10,     # High frequency
    "max_agents": 100000,
    "phi_source": "transaction_embedding",
    "hardware": {
        "cpu": "Intel Xeon / AMD EPYC 16+ cores",
        "memory": "64-128GB",
        "zk_acceleration": "FPGA (Xilinx Alveo U250)",
        "hsm": "Required for production keys"
    },
    "compliance": {
        "audit_retention_days": 2555,  # 7 years
        "proof_generation": "per_trade",
        "circuit_breaker": {
            "max_drift": 0.5,
            "action": "halt_trading"
        }
    }
}
```

---

## Use Case D: Edge AI (IoT Sensors)

### Goal

Deploy cognitive monitoring on resource-constrained edge devices with cloud aggregation.

### System Context

| Component | Role |
|-----------|------|
| **Edge Device** | Sensor with minimal compute |
| **EFMCore (Lite)** | Lightweight local monitoring |
| **Edge Gateway** | Aggregates local states |
| **Cloud d-CTM** | Global consensus and proofs |

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLOUD                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              d-CTM + ZK-SP Aggregator                │    │
│  └─────────────────────────┬───────────────────────────┘    │
└─────────────────────────────┼───────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│   Edge Gateway   │ │   Edge Gateway   │ │   Edge Gateway   │
│   (EFMCore)      │ │   (EFMCore)      │ │   (EFMCore)      │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
         │                    │                    │
    ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
    │         │          │         │          │         │
    ▼         ▼          ▼         ▼          ▼         ▼
┌──────┐ ┌──────┐   ┌──────┐ ┌──────┐   ┌──────┐ ┌──────┐
│Sensor│ │Sensor│   │Sensor│ │Sensor│   │Sensor│ │Sensor│
│(Lite)│ │(Lite)│   │(Lite)│ │(Lite)│   │(Lite)│ │(Lite)│
└──────┘ └──────┘   └──────┘ └──────┘   └──────┘ └──────┘
```

### Configuration

```python
EDGE_AI_CONFIG = {
    "deployment": "edge_ai",
    "phi_dimensions": 16,       # Minimal
    "tick_interval_ms": 100,
    "max_agents": 1000,         # Per gateway
    "phi_source": "sensor_embedding",
    "hardware": {
        "cpu": "ARM Cortex-A76 or equivalent, 4+ cores",
        "memory": "4-8GB",
        "zk_acceleration": "Software (batch to cloud)"
    },
    "power": {
        "max_tdp": "15W",
        "battery_aware": True
    }
}
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Select deployment profile (Edge/FinTech/Robotic/LLM)
- [ ] Configure Φ dimensions and tick interval
- [ ] Set up d-CTM cluster (minimum 3 nodes for BFT)
- [ ] Configure ZK-SP acceleration (CPU/GPU/FPGA/ASIC)
- [ ] Install TPM 2.0 on all nodes
- [ ] Set up monitoring dashboard

### Deployment

- [ ] Deploy EFMCore containers/sidecars
- [ ] Verify d-CTM consensus formation
- [ ] Test drift injection and detection
- [ ] Validate ZK proof generation and verification
- [ ] Configure alerting and escalation paths
- [ ] Document recovery procedures

### Post-Deployment

- [ ] Monitor drift rates and TTF distribution
- [ ] Review audit logs weekly
- [ ] Test failover procedures monthly
- [ ] Update threat model quarterly
- [ ] Rotate keys per security policy

---

## Summary Matrix

| Use Case | Φ Dims | Tick (ms) | ZK Accel | Key Metric |
|----------|--------|-----------|----------|------------|
| Robotic Swarm | 32 | 50 | GPU | Recovery time |
| LLM Audit | 128 | 1000 | ASIC | Drift detection |
| FinTech | 64 | 10 | FPGA | Trade latency |
| Edge AI | 16 | 100 | Cloud | Power consumption |
