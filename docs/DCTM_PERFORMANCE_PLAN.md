# d-CTM Performance Plan

This document specifies the simulation harness and performance metrics for validating the distributed Cognitive Trace Memory (d-CTM) system.

---

## 1. Simulation Objectives

### Primary Goals

1. **Latency Characterization**: Measure end-to-end latency from capsule state change to consensus update
2. **Drift Detection Accuracy**: Validate TPE and IA-BIM detect drift within specified thresholds
3. **Proof Generation Throughput**: Measure ZK-SP proof generation and verification rates
4. **Fault Tolerance**: Verify Byzantine consensus under adversarial conditions
5. **Scalability**: Establish performance curves for 10 → 100 → 1,000 → 10,000 nodes

### Success Criteria

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Drift detection latency | < 100ms | < 500ms |
| Consensus formation | < 1s | < 5s |
| Proof generation | > 100/s | > 10/s |
| Proof verification | > 1000/s | > 100/s |
| Node failure recovery | < 30s | < 60s |
| Byzantine tolerance | 33% | 25% |

---

## 2. Simulation Architecture

### 2.1 Test Harness

```python
class DCTMSimulator:
    """
    Simulation harness for d-CTM performance testing.
    
    Components:
    - Node simulator (configurable count)
    - Network simulator (configurable latency/loss)
    - Workload generator (configurable patterns)
    - Metrics collector
    """
    
    def __init__(self, config: SimConfig):
        self.nodes: List[SimulatedNode] = []
        self.network: NetworkSimulator = NetworkSimulator(config.network)
        self.workload: WorkloadGenerator = WorkloadGenerator(config.workload)
        self.metrics: MetricsCollector = MetricsCollector()
        
    def run(self, duration_ticks: int) -> SimulationResult:
        """Execute simulation and return metrics."""
        pass
```

### 2.2 Simulated Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMULATION HARNESS                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Workload Generator                      │    │
│  │  - Capsule state streams                                │    │
│  │  - Drift injection                                      │    │
│  │  - Byzantine behavior                                   │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Network Simulator                       │    │
│  │  - Configurable latency                                 │    │
│  │  - Packet loss                                          │    │
│  │  - Partition simulation                                 │    │
│  └─────────────────────────┬───────────────────────────────┘    │
│                            │                                     │
│          ┌─────────────────┼─────────────────┐                  │
│          │                 │                 │                  │
│          ▼                 ▼                 ▼                  │
│    ┌──────────┐      ┌──────────┐      ┌──────────┐            │
│    │  Node 1  │◄────►│  Node 2  │◄────►│  Node N  │            │
│    │ (EFMCore)│      │ (EFMCore)│      │ (EFMCore)│            │
│    └──────────┘      └──────────┘      └──────────┘            │
│          │                 │                 │                  │
│          └─────────────────┼─────────────────┘                  │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Metrics Collector                       │    │
│  │  - Latency histograms                                   │    │
│  │  - Throughput counters                                  │    │
│  │  - Error rates                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Test Scenarios

### 3.1 Scenario A: Baseline Performance

**Objective**: Establish baseline metrics under normal operation

**Configuration**:
```python
SCENARIO_A_CONFIG = {
    "nodes": 10,
    "capsules_per_node": 100,
    "duration_ticks": 1000,
    "network": {
        "latency_ms": 10,
        "loss_rate": 0.0,
        "partition": False
    },
    "workload": {
        "drift_injection_rate": 0.0,
        "byzantine_nodes": 0
    }
}
```

**Metrics to Collect**:
- Tick processing time (mean, p50, p95, p99)
- Consensus formation time
- Memory usage per node
- CPU utilization per node

### 3.2 Scenario B: Drift Detection

**Objective**: Measure drift detection accuracy and latency

**Configuration**:
```python
SCENARIO_B_CONFIG = {
    "nodes": 10,
    "capsules_per_node": 100,
    "duration_ticks": 1000,
    "network": {
        "latency_ms": 10,
        "loss_rate": 0.0
    },
    "workload": {
        "drift_injection": {
            "rate": 0.01,  # 1% of capsules per tick
            "severity_distribution": {
                "MILD": 0.6,
                "MODERATE": 0.3,
                "SEVERE": 0.08,
                "CRITICAL": 0.02
            }
        }
    }
}
```

**Metrics to Collect**:
- Detection latency by severity
- False positive rate
- False negative rate
- Cascade detection accuracy

### 3.3 Scenario C: Node Failure

**Objective**: Measure failover and recovery performance

**Configuration**:
```python
SCENARIO_C_CONFIG = {
    "nodes": 10,
    "capsules_per_node": 100,
    "duration_ticks": 1000,
    "failures": [
        {"tick": 200, "type": "node_crash", "node_id": 3},
        {"tick": 400, "type": "node_crash", "node_id": 7},
        {"tick": 600, "type": "network_partition", "nodes": [1, 2, 3]}
    ]
}
```

**Metrics to Collect**:
- Time to detect failure
- Time to redistribute capsules
- Time to restore consensus
- Data loss (if any)

### 3.4 Scenario D: Byzantine Behavior

**Objective**: Verify Byzantine fault tolerance

**Configuration**:
```python
SCENARIO_D_CONFIG = {
    "nodes": 10,
    "capsules_per_node": 100,
    "duration_ticks": 1000,
    "workload": {
        "byzantine_nodes": 3,  # 30% Byzantine
        "byzantine_behavior": [
            "send_conflicting_messages",
            "delay_responses",
            "forge_proofs"
        ]
    }
}
```

**Metrics to Collect**:
- Consensus stability under Byzantine load
- Detection of Byzantine behavior
- Recovery time after Byzantine node eviction

### 3.5 Scenario E: Scale Test

**Objective**: Establish scalability limits

**Configuration**:
```python
SCENARIO_E_CONFIGS = [
    {"nodes": 10, "capsules_per_node": 100},
    {"nodes": 100, "capsules_per_node": 100},
    {"nodes": 1000, "capsules_per_node": 100},
    {"nodes": 10000, "capsules_per_node": 100},
]
```

**Metrics to Collect**:
- Throughput vs. node count
- Latency vs. node count
- Memory vs. node count
- Proof size vs. swarm size

---

## 4. Metrics Specification

### 4.1 Latency Metrics

```python
@dataclass
class LatencyMetrics:
    # Tick processing
    tick_latency_mean_ms: float
    tick_latency_p50_ms: float
    tick_latency_p95_ms: float
    tick_latency_p99_ms: float
    
    # Drift detection
    drift_detection_latency_ms: float  # From drift start to detection
    
    # Consensus
    consensus_formation_ms: float  # From proposal to quorum
    
    # Proof generation
    proof_generation_ms: float  # Time to generate agent proof
    proof_verification_ms: float  # Time to verify proof
    
    # Recovery
    failover_latency_ms: float  # Time to redistribute after node failure
```

### 4.2 Throughput Metrics

```python
@dataclass
class ThroughputMetrics:
    # Processing
    ticks_per_second: float
    capsules_processed_per_second: float
    
    # Consensus
    proposals_per_second: float
    votes_per_second: float
    
    # Proofs
    proofs_generated_per_second: float
    proofs_verified_per_second: float
```

### 4.3 Accuracy Metrics

```python
@dataclass
class AccuracyMetrics:
    # Drift detection
    true_positive_rate: float  # Detected real drift
    false_positive_rate: float  # Flagged non-drift as drift
    false_negative_rate: float  # Missed real drift
    
    # Severity classification
    severity_accuracy: float  # Correct severity assignment
    
    # Cascade prediction
    cascade_prediction_accuracy: float
```

### 4.4 Resource Metrics

```python
@dataclass
class ResourceMetrics:
    # Per-node
    cpu_utilization_percent: float
    memory_usage_mb: float
    network_bandwidth_mbps: float
    
    # Per-capsule
    memory_per_capsule_kb: float
    
    # Per-proof
    proof_size_bytes: int
```

---

## 5. Test Execution Plan

### 5.1 Phase 1: Unit Tests (Week 1)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| TPE unit | Drift velocity calculation | < 1% error vs. analytical |
| CAC unit | Trace level determination | 100% correct |
| RPC unit | Lineage validation | 100% correct |
| BFT unit | Consensus formation | Quorum reached |

### 5.2 Phase 2: Integration Tests (Week 2)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| 3-node cluster | Basic consensus | Consensus in < 1s |
| Drift detection | End-to-end drift flow | Detection in < 100ms |
| Failover | Single node failure | Recovery in < 30s |
| ZK-SP integration | Proof generation | Valid proofs |

### 5.3 Phase 3: Performance Tests (Week 3)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| Scenario A | Baseline | Meet all baseline metrics |
| Scenario B | Drift detection | < 5% false negative |
| Scenario C | Node failure | < 30s recovery |
| Scenario D | Byzantine | Consensus maintained |

### 5.4 Phase 4: Scale Tests (Week 4)

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| 100-node | Medium scale | Linear scaling |
| 1000-node | Large scale | Sub-linear acceptable |
| 10000-node | Stress test | Identify limits |

---

## 6. Reporting

### 6.1 Report Template

```markdown
# d-CTM Performance Report

## Executive Summary
- Test date: YYYY-MM-DD
- Configuration: [nodes, capsules, duration]
- Overall result: PASS/FAIL

## Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Drift detection latency | < 100ms | XXms | ✅/❌ |
| Consensus formation | < 1s | XXms | ✅/❌ |
| Proof generation | > 100/s | XX/s | ✅/❌ |

## Detailed Results

### Latency Distribution
[Histogram of tick latency]

### Throughput Over Time
[Line chart of ops/second]

### Resource Utilization
[CPU/Memory charts]

## Recommendations
1. ...
2. ...

## Appendix
- Full configuration
- Raw data files
- Error logs
```

### 6.2 Automated Reporting

```python
def generate_report(results: SimulationResult) -> str:
    """Generate markdown report from simulation results."""
    
    report = f"""
# d-CTM Performance Report

## Executive Summary
- Test date: {results.timestamp}
- Configuration: {results.config}
- Overall result: {"PASS" if results.all_passed() else "FAIL"}

## Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Drift detection | < 100ms | {results.drift_latency_ms:.1f}ms | {"✅" if results.drift_latency_ms < 100 else "❌"} |
| Consensus | < 1s | {results.consensus_ms:.1f}ms | {"✅" if results.consensus_ms < 1000 else "❌"} |
| Proof gen | > 100/s | {results.proof_rate:.1f}/s | {"✅" if results.proof_rate > 100 else "❌"} |

## Latency Percentiles

| Percentile | Value |
|------------|-------|
| p50 | {results.latency.p50_ms:.1f}ms |
| p95 | {results.latency.p95_ms:.1f}ms |
| p99 | {results.latency.p99_ms:.1f}ms |
"""
    return report
```

---

## 7. Implementation

### 7.1 Simulation Harness Code

```python
"""
d-CTM Performance Simulation Harness

Usage:
    python dctm_benchmark.py --scenario A --nodes 10 --duration 1000
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict


@dataclass
class SimConfig:
    nodes: int = 10
    capsules_per_node: int = 100
    duration_ticks: int = 1000
    network_latency_ms: float = 10.0
    drift_injection_rate: float = 0.0
    byzantine_nodes: int = 0


class DCTMBenchmark:
    def __init__(self, config: SimConfig):
        self.config = config
        self.metrics = defaultdict(list)
        
    def run(self) -> Dict:
        """Run full benchmark suite."""
        
        # Initialize nodes
        nodes = self._create_nodes()
        
        # Run simulation
        for tick in range(self.config.duration_ticks):
            tick_start = time.time()
            
            # Process each node
            for node in nodes:
                node.tick()
            
            # Inject drift if configured
            if np.random.random() < self.config.drift_injection_rate:
                self._inject_drift(nodes)
            
            # Measure consensus
            consensus_start = time.time()
            self._form_consensus(nodes)
            consensus_time = (time.time() - consensus_start) * 1000
            self.metrics['consensus_ms'].append(consensus_time)
            
            # Record tick time
            tick_time = (time.time() - tick_start) * 1000
            self.metrics['tick_ms'].append(tick_time)
        
        return self._compute_summary()
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics."""
        return {
            'tick_latency': {
                'mean': np.mean(self.metrics['tick_ms']),
                'p50': np.percentile(self.metrics['tick_ms'], 50),
                'p95': np.percentile(self.metrics['tick_ms'], 95),
                'p99': np.percentile(self.metrics['tick_ms'], 99)
            },
            'consensus_latency': {
                'mean': np.mean(self.metrics['consensus_ms']),
                'max': np.max(self.metrics['consensus_ms'])
            },
            'drift_detection': {
                'count': len(self.metrics['drift_detected']),
                'latency_mean': np.mean(self.metrics['drift_latency_ms']) if self.metrics['drift_latency_ms'] else 0
            }
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', choices=['A', 'B', 'C', 'D', 'E'], default='A')
    parser.add_argument('--nodes', type=int, default=10)
    parser.add_argument('--duration', type=int, default=1000)
    args = parser.parse_args()
    
    config = SimConfig(
        nodes=args.nodes,
        duration_ticks=args.duration
    )
    
    benchmark = DCTMBenchmark(config)
    results = benchmark.run()
    
    print(f"\n=== d-CTM Benchmark Results (Scenario {args.scenario}) ===")
    print(f"Nodes: {args.nodes}, Duration: {args.duration} ticks")
    print(f"\nTick Latency:")
    print(f"  Mean: {results['tick_latency']['mean']:.2f}ms")
    print(f"  P95:  {results['tick_latency']['p95']:.2f}ms")
    print(f"  P99:  {results['tick_latency']['p99']:.2f}ms")


if __name__ == '__main__':
    main()
```

---

## 8. Expected Results

Based on architectural analysis, expected performance:

| Scale | Tick Latency (p95) | Consensus | Proof Gen |
|-------|-------------------|-----------|-----------|
| 10 nodes | 5ms | 50ms | 200/s |
| 100 nodes | 20ms | 200ms | 150/s |
| 1,000 nodes | 100ms | 1s | 100/s |
| 10,000 nodes | 500ms | 5s | 50/s |

**Note**: Actual results will vary based on hardware and network conditions.

---

## 9. Continuous Performance Monitoring

### 9.1 Production Metrics

Once deployed, continuously monitor:

```python
PRODUCTION_METRICS = [
    "efm_tick_duration_seconds",
    "efm_drift_detection_latency_seconds",
    "efm_consensus_formation_seconds",
    "efm_proof_generation_rate",
    "efm_node_health_status",
    "efm_capsule_count_total",
    "efm_quarantine_count"
]
```

### 9.2 Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Tick latency p95 | > 200ms | > 500ms |
| Consensus time | > 2s | > 5s |
| Drift detection | > 500ms | > 1s |
| Node failures | > 1 | > 2 |
| Proof generation rate | < 50/s | < 10/s |
