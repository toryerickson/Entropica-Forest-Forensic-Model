# EFMCore API Specification

## Overview

The `EFMCore` class implements the integrated control loop for the Entropica Forensic Model (EFM), coordinating capsule introspection, projection-based forecasting, forensic tracing, and autonomous response.

This API enables integration into robotic swarms, distributed agents, or intelligent systems requiring symbolic introspection and fault-tolerant autonomy.

---

## Class: EFMCore

```python
class EFMCore:
    def __init__(self, bim: BIM, csl: CSL, ctm: CTM) -> None:
        """
        Initializes the EFM core with required dependencies.
        
        Parameters:
            bim: BehaviorIntegrityModule - Tracks symbolic deviation between steps
            csl: CausalSymbolicLearner - Learns causality motifs from failure history
            ctm: CognitiveTraceMatrix - Stores symbolic trace graph
        """
```

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `bim` | `BehaviorIntegrityModule` | Tracks symbolic deviation between steps |
| `csl` | `CausalSymbolicLearner` | Learns and applies causality motifs from failure history |
| `ctm` | `CognitiveTraceMatrix` | Stores symbolic trace graph |

---

## Core Methods

### `tick(capsule_state: CapsuleState) -> Tuple[int, List[DSLAction]]`

Executes a single lifecycle loop for a capsule.

```python
def tick(self, capsule_state: CapsuleState) -> Tuple[int, List[DSLAction]]:
    """
    Updates state and outputs trace level and DSL actions.
    
    Parameters:
        capsule_state: Current state of the capsule including phi, entropy, stability
        
    Returns:
        trace_level: Adjusted trace depth (L1-L4) for capsule memory fidelity
        actions: Triggered DSL actions (partition, rollback, escalate, etc.)
    """
```

**Returns:**

| Output | Type | Description |
|--------|------|-------------|
| `trace_level` | `int` | Adjusted trace depth (1-4) for capsule memory fidelity |
| `actions` | `List[DSLAction]` | Triggered DSL actions |

---

### `get_lineage(capsule_state: CapsuleState) -> LineageGraph`

Builds the lineage graph from current capsule state.

```python
def get_lineage(self, capsule_state: CapsuleState) -> LineageGraph:
    """
    Constructs ancestry tree for hereditary drift analysis.
    
    Parameters:
        capsule_state: Current capsule state with parent_id reference
        
    Returns:
        LineageGraph with nodes and edges representing capsule ancestry
    """
```

---

### `execute_dsl_action(action: DSLAction, capsule_state: CapsuleState) -> None`

Applies a DSL action to a capsule state.

```python
def execute_dsl_action(self, action: DSLAction, capsule_state: CapsuleState) -> None:
    """
    Executes governance action on capsule.
    
    Parameters:
        action: The DSL action to execute
        capsule_state: Target capsule state
    """
```

---

### `performance_report() -> str`

Returns a summary of key performance metrics.

```python
def performance_report(self) -> str:
    """
    Generates performance summary including:
    - Ticks processed
    - Drift events detected
    - Actions executed
    - Average TTF
    - Current trace level distribution
    """
```

---

### `audit_proof() -> GlobalProof`

Generates cryptographic proof of system state.

```python
def audit_proof(self) -> GlobalProof:
    """
    Creates ZK-SP proof for external verification.
    
    Returns:
        GlobalProof containing merkle root and aggregated statistics
    """
```

---

## Data Types

### CapsuleState

```python
@dataclass
class CapsuleState:
    id: str                     # Unique capsule identifier
    phi: np.ndarray             # Φ vector (L2-normalized)
    entropy: float              # Current entropy H ∈ [0, 1]
    stability: float            # Current stability S ∈ [0, 1]
    parent_id: Optional[str]    # Parent capsule (None if root)
    metadata: Dict[str, Any]    # Additional context
    timestamp: float            # Unix timestamp
```

### DSLAction

```python
class DSLAction(Enum):
    PARTITION = "partition"     # Isolate capsule for recovery
    ROLLBACK = "rollback"       # Revert to last stable state
    ESCALATE = "escalate"       # Report to cluster-level node
    QUARANTINE = "quarantine"   # Suspend capsule operations
    HEAL = "heal"               # Attempt automatic repair
    PRUNE = "prune"             # Remove from active set
    MIGRATE = "migrate"         # Transfer to different node
    SNAPSHOT = "snapshot"       # Capture current state
    BOOST_SCD = "boost_scd"     # Increase cognitive density
    ISOLATE_LINEAGE = "isolate" # Quarantine entire lineage
```

### LineageGraph

```python
@dataclass
class LineageGraph:
    nodes: Dict[str, CapsuleState]  # All capsules in lineage
    edges: List[Tuple[str, str]]    # Parent-child relationships
    root_id: str                    # Original ancestor
    depth: int                      # Generations from root
```

### GlobalProof

```python
@dataclass
class GlobalProof:
    proof_id: str               # Unique proof identifier
    merkle_root: str            # Root hash of state tree
    swarm_size: int             # Number of agents covered
    timestamp: float            # Proof generation time
    public_inputs: Dict         # Verifiable public claims
    proof_data: bytes           # ZK proof bytes
```

---

## DSL Action Predicates

### Example Usage

```python
# High drift risk triggers partition
if capsule.DriftRisk > 0.85:
    capsule.partition(mode="safe")
    lineage.notify_drift(parent_id)

# Low stability triggers escalation
if capsule.stability < 0.3 and capsule.ttf < 10:
    efm.execute_dsl_action(DSLAction.ESCALATE, capsule)
    
# Hereditary drift triggers lineage isolation
if lineage.cascade_risk > 3.0:
    efm.execute_dsl_action(DSLAction.ISOLATE_LINEAGE, capsule)
```

### Supported DSL Actions

| Action | Description | Trigger Condition |
|--------|-------------|-------------------|
| `partition()` | Isolates capsule for recovery | DriftRisk > 0.85 |
| `rollback()` | Reverts to last stable symbolic state | Stability < 0.2 |
| `escalate()` | Reports hereditary risk to cluster node | TTF < 10 |
| `quarantine()` | Suspends all capsule operations | Severity ≥ CRITICAL |
| `heal()` | Attempts automatic repair | Severity = MILD, SCD > 0.5 |
| `prune()` | Removes capsule from active set | SCD < 0.15, Age > threshold |

---

## Subcomponents

### TPE: TemporalProjectionEngine

Forecasts system collapse (TTF) based on symbolic coherence degradation.

```python
class TPE:
    def compute_drift_velocity(self, phi_history: List[np.ndarray]) -> float:
        """Calculate rate of change in Φ space"""
        
    def predict_ttf(self, current_drift: float, threshold: float) -> float:
        """Estimate ticks until failure threshold breach"""
        
    def forecast_trajectory(self, horizon: int) -> List[float]:
        """Project future drift values"""
```

### CAC: CognitiveApertureController

Adjusts trace resolution using α_dyn based on TTF and symbolic pressure.

```python
class CAC:
    def compute_alpha_dyn(self, entropy: float, stability: float) -> float:
        """α_dyn = 0.1 + 0.9 × H × (1 - S)"""
        
    def determine_trace_level(self, ttf: float, sci: float) -> int:
        """Map metrics to L1-L4 trace depth"""
        
    def get_sampling_rate(self, trace_level: int) -> float:
        """Return appropriate sampling rate for level"""
```

### RPC: ReflectiveProjectionCheck

Performs self-assessment of symbolic lineage and triggers DSL actions.

```python
class RPC:
    def compute_sci(self, capsule: CapsuleState, lineage: LineageGraph) -> float:
        """Symbolic Coherence Index based on lineage similarity"""
        
    def validate_lineage(self, capsule: CapsuleState) -> bool:
        """Verify ancestry chain integrity"""
        
    def detect_orphan(self, capsule: CapsuleState) -> OrphanType:
        """Classify orphan status (SEVERED, CORRUPTED, ADOPTED, etc.)"""
```

### IPE: InternalProjectionEngine

Analyzes capsule ancestry to detect hereditary drift and calculate entropy.

```python
class IPE:
    def compute_entropy(self, phi: np.ndarray) -> float:
        """Calculate symbolic entropy H"""
        
    def detect_hereditary_drift(self, lineage: LineageGraph) -> List[str]:
        """Identify capsules with inherited drift"""
        
    def compute_fhd(self, capsule: CapsuleState) -> float:
        """Forensic Horizon Decay - time-decay based on LKC entropy"""
```

---

## Example Lifecycle

```python
from efm_core import EFMCore, CapsuleState
import numpy as np

# Initialize EFM with dependencies
bim = BehaviorIntegrityModule()
csl = CausalSymbolicLearner()
ctm = CognitiveTraceMatrix(dimensions=48)

efm = EFMCore(bim=bim, csl=csl, ctm=ctm)

# Create capsule state
phi = np.random.randn(48)
phi = phi / np.linalg.norm(phi)  # L2 normalize

capsule_state = CapsuleState(
    id="c.4041",
    phi=phi,
    entropy=0.93,
    stability=0.14,
    parent_id="c.4029",
    metadata={"source": "agent_007"},
    timestamp=time.time()
)

# Execute tick
trace_level, actions = efm.tick(capsule_state)

# Result: trace_level = 4 (L4 - Emergency)
#         actions = [DSLAction.PARTITION, DSLAction.ESCALATE]

# Get lineage for analysis
lineage = efm.get_lineage(capsule_state)
print(f"Lineage depth: {lineage.depth}")

# Generate audit proof
proof = efm.audit_proof()
print(f"Merkle root: {proof.merkle_root}")
```

---

## HTTP API (FastAPI)

For REST deployment, the EFMCore is wrapped in FastAPI:

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /ingest` | POST | Submit capsule state, returns governance policy |
| `GET /status` | GET | Current swarm status and statistics |
| `GET /agent/{id}` | GET | Individual agent state (TTF, SCI, drift) |
| `POST /vote` | POST | Submit d-CTM consensus vote |
| `GET /consensus/{id}` | GET | Get consensus result for proposal |
| `POST /escalate` | POST | Manual operator escalation |
| `GET /audit` | GET | Audit trail with filters |
| `GET /proof` | GET | Generate and return ZK-SP proof |
| `GET /health` | GET | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent_007",
    "phi_vector": [0.1, 0.2, ...],
    "parent_id": "agent_001",
    "metadata": {"source": "sensor_array"}
  }'
```

### Example Response

```json
{
  "trace_level": 2,
  "sampling_rate": 0.5,
  "tau_break": 30,
  "global_alert_active": false,
  "instructions": ["Continue normal operation", "Increase monitoring frequency"]
}
```

---

## Integration Notes

1. **Real-time or Batch**: EFMCore accepts both streaming and batch inputs
2. **Deployment Targets**: Robotic agents, financial AI, LLM monitoring, distributed governance
3. **ZK-SP Layer**: Best used with Plonky2-based proof system (see `zksp_stubs.py`)
4. **d-CTM Clustering**: Integrates with distributed consensus (see `distributed_efm.py`)

---

## Configuration

```python
# Default configuration
EFM_CONFIG = {
    "phi_dimensions": 48,
    "drift_threshold": 0.3,
    "stability_threshold": 0.5,
    "entropy_threshold": 0.7,
    "ttf_emergency": 10,
    "ttf_high_risk": 30,
    "ttf_elevated": 60,
    "sci_emergency": 0.6,
    "sci_high_risk": 0.75,
    "sci_elevated": 0.85,
    "tick_interval_ms": 100,
    "max_history_length": 100,
    "pruning_threshold_scd": 0.15,
    "cascade_risk_threshold": 3.0
}
```

---

## Error Handling

```python
class EFMError(Exception):
    """Base EFM exception"""

class LineageCorruptionError(EFMError):
    """Raised when lineage validation fails"""

class ConsensusTimeoutError(EFMError):
    """Raised when d-CTM consensus times out"""

class ProofGenerationError(EFMError):
    """Raised when ZK-SP proof generation fails"""
```

---

## License

MIT / Entropica Foundation 2025
