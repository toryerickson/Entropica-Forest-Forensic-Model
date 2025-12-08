# The Gardener Protocol: Human-Swarm Interface Specification

## Philosophy

> "If the system is a forest, the operator is a gardener, not a mechanic."

At 1,000,000 agents, direct control is impossible. The Gardener Protocol transforms the operator from **Controller** to **Curator** - someone who shapes the environment rather than commands individual units.

---

## The Four Scales of Observation

The interface provides **semantic zoom** - each scale reveals different information and enables different actions.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           THE GARDENER INTERFACE                             │
│                                                                              │
│  SCALE 1: CANOPY (1M agents)                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    🌳 GLOBAL HEALTH MAP 🌳                          │    │
│  │                                                                      │    │
│  │    ████████████░░░░░░░░████████████████░░░░░░████████████           │    │
│  │    ████████████░░░░░░░░████████████████░░░░░░████████████           │    │
│  │    ████████░░░░░░░░░░░░████████████████░░░░░░████████████           │    │
│  │    ████░░░░░░░░░░░░░░░░░░░░████████████░░░░░░████████████           │    │
│  │    ░░░░░░░░██████░░░░░░░░░░████████████░░░░░░████████████           │    │
│  │    ░░░░████████████░░░░░░░░████████████░░░░░░████████████           │    │
│  │                                                                      │    │
│  │    Legend: ████ Healthy (SCI > 0.8)                                 │    │
│  │            ░░░░ Stressed (SCI 0.5-0.8)                              │    │
│  │            ░░░░ Critical (SCI < 0.5)  [Red in actual UI]            │    │
│  │                                                                      │    │
│  │    Actions: [Set Global λ] [Designate Priority Sector] [Terraform]  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  SCALE 2: TRUNK (10K agents) - Click to zoom                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    🌿 PHYLOGENETIC TREE 🌿                          │    │
│  │                                                                      │    │
│  │                           [ROOT]                                     │    │
│  │                          /      \                                    │    │
│  │                     [T-001]    [T-002]                              │    │
│  │                    /   |   \      |                                  │    │
│  │               [B-01] [B-02] [B-03] [B-04]                           │    │
│  │                 ⚠️    ✓      ✓      ❌                               │    │
│  │                                                                      │    │
│  │    Actions: [Prune Branch] [Graft Heuristic] [Boost Resources]      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  SCALE 3: BRANCH (100 agents) - Click branch to zoom                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    🗳️ CONSENSUS VIEW 🗳️                             │    │
│  │                                                                      │    │
│  │    Proposal: ESCALATE agent_07234                                   │    │
│  │    ┌──────────────────────────────────────────┐                     │    │
│  │    │ FOR:  ████████████████████ 78%           │                     │    │
│  │    │ AGAINST: ██████ 22%                       │                     │    │
│  │    └──────────────────────────────────────────┘                     │    │
│  │                                                                      │    │
│  │    Drift Velocity: ▁▂▃▅▆▇█▇▆▅▃▂▁ (last 20 ticks)                   │    │
│  │                                                                      │    │
│  │    Actions: [Override Vote] [Investigate Dissent] [Force Quorum]    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  SCALE 4: LEAF (1 agent) - Click agent to zoom                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    🔬 FORENSIC DEBUG 🔬                              │    │
│  │                                                                      │    │
│  │    Agent ID: cap_07234                                              │    │
│  │    Status: QUARANTINED                                              │    │
│  │    Parent: cap_07201                                                │    │
│  │    Children: [cap_07245, cap_07246]                                 │    │
│  │                                                                      │    │
│  │    Φ Vector: [0.12, -0.34, 0.56, ...]                              │    │
│  │    Stability: 0.23 ⚠️                                               │    │
│  │    Entropy: 0.87 🔥                                                  │    │
│  │    TTF: 8 ticks ❌                                                   │    │
│  │                                                                      │    │
│  │    LKC Log: [View Full Trace] [Download JSON]                       │    │
│  │                                                                      │    │
│  │    Actions: [Release] [Terminate] [Rebuild from Snapshot]           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Scale Specifications

### Scale 1: The Canopy (Global View)

**Resolution**: 1,000,000 agents displayed as a density heatmap

**Visualization**: 
- 3D topological surface where height = knowledge density
- Color = health (green → yellow → red)
- Particles = active discoveries
- Storms = drift clusters

**Metrics Displayed**:
| Metric | Description |
|--------|-------------|
| Global SCI | Swarm-wide Symbolic Coherence Index |
| Discovery Rate (FDR) | Findings per tick across all clusters |
| Entropy Budget | Remaining growth capacity |
| Active Escalations | Number of human-review requests |

**Gardener Actions**:
| Action | Description | Effect |
|--------|-------------|--------|
| **Set Global λ** | Adjust risk tolerance | Changes drift thresholds swarm-wide |
| **Designate Priority** | Mark data sector as important | Increases resource allocation to region |
| **Terraform** | Reshape swarm topology | Redistributes agents across clusters |
| **Declare Season** | Signal environmental change | Triggers growth (Spring) or conservation (Winter) mode |

---

### Scale 2: The Trunk (Cluster View)

**Resolution**: 10,000 agents displayed as a phylogenetic tree

**Visualization**:
- Tree structure showing lineage relationships
- Branch thickness = resource allocation
- Branch color = health status
- Icons = alerts and warnings

**Metrics Displayed**:
| Metric | Description |
|--------|-------------|
| Cluster SCI | Coherence within this cluster |
| Lineage Depth | Generations from root |
| Branch Health | Aggregated child health |
| Consensus Ratio | Agreement level on decisions |

**Gardener Actions**:
| Action | Description | Effect |
|--------|-------------|--------|
| **Prune Branch** | Remove unhealthy lineage | Terminates branch and recovers resources |
| **Graft Heuristic** | Transfer successful pattern | Copies CSL-learned heuristic to branch |
| **Boost Resources** | Allocate extra entropy budget | Accelerates growth in promising areas |
| **Quarantine Branch** | Isolate without terminating | Prevents contagion while investigating |

---

### Scale 3: The Branch (Consensus View)

**Resolution**: 100 agents displayed as a voting panel

**Visualization**:
- Voting bars showing agreement/disagreement
- Timeline of drift velocity
- Network graph of agent relationships
- Alert queue

**Metrics Displayed**:
| Metric | Description |
|--------|-------------|
| Vote Distribution | FOR/AGAINST ratio on proposals |
| Drift Velocity | Rate of change in symbolic space |
| Dissent Clusters | Agents disagreeing with majority |
| Active Proposals | Pending governance decisions |

**Gardener Actions**:
| Action | Description | Effect |
|--------|-------------|--------|
| **Override Vote** | Force decision | Bypasses consensus for emergency |
| **Investigate Dissent** | Drill into disagreeing agents | Opens Scale 4 for each dissenter |
| **Force Quorum** | Require minimum participation | Prevents low-engagement decisions |
| **Replay History** | View decision timeline | Shows how consensus evolved |

---

### Scale 4: The Leaf (Forensic View)

**Resolution**: 1 agent displayed with full diagnostic data

**Visualization**:
- Complete state dump
- Phi vector visualization (radar chart)
- Lineage path (breadcrumb trail to root)
- Event log timeline

**Metrics Displayed**:
| Metric | Description |
|--------|-------------|
| Φ Vector | Raw symbolic state (48D) |
| Stability | Current S value |
| Entropy | Current H value |
| TTF | Time-to-failure prediction |
| Trace Level | Current L1-L4 fidelity |
| LKC | Last Known Configuration |

**Gardener Actions**:
| Action | Description | Effect |
|--------|-------------|--------|
| **Release** | Remove from quarantine | Returns agent to active pool |
| **Terminate** | Permanent removal | Triggers CDP resource recovery |
| **Rebuild** | Restore from snapshot | Resets to last stable state |
| **Inject Test** | Send synthetic signal | Tests agent response |

---

## The Gardener's Toolkit

### Climate Controls (Global)

```python
class ClimateController:
    """
    High-level controls that affect the entire swarm environment.
    These are the "weather" controls - broad, slow-acting, powerful.
    """
    
    def declare_season(self, season: str):
        """
        Signal environmental change to swarm.
        
        Seasons:
        - SPRING: High growth, low pruning, resource expansion
        - SUMMER: Balanced growth and consolidation
        - AUTUMN: Begin harvesting discoveries, reduce expansion
        - WINTER: Conservation mode, aggressive pruning, hunker down
        """
        pass
    
    def set_global_lambda(self, lambda_value: float):
        """
        Set swarm-wide risk tolerance.
        
        λ = 0.0: Extremely conservative (reject all uncertainty)
        λ = 0.5: Balanced (default)
        λ = 1.0: Aggressive exploration (accept high risk)
        """
        pass
    
    def terraform(self, source_region: str, target_region: str, percentage: float):
        """
        Redistribute agents between regions.
        
        Use when:
        - A region is exhausted (low FDR)
        - A new data sector needs attention
        - Balancing load across clusters
        """
        pass
```

### Cultivation Controls (Cluster)

```python
class CultivationController:
    """
    Mid-level controls for managing clusters and lineages.
    These are the "gardening" tools - targeted, medium-acting.
    """
    
    def prune(self, branch_id: str, reason: str):
        """
        Remove a branch and all descendants.
        
        Triggers:
        - CDP resource recovery
        - Orphan protocol for connected branches
        - Audit log entry
        """
        pass
    
    def graft(self, source_branch: str, target_branch: str, heuristic_id: str):
        """
        Copy a successful heuristic from one branch to another.
        
        The CSL-learned pattern is transferred, giving the target
        branch the benefit of the source's discoveries.
        """
        pass
    
    def fertilize(self, branch_id: str, resource_boost: float):
        """
        Allocate additional entropy budget to a branch.
        
        Use when a branch shows promise but needs resources
        to fully explore its discovery space.
        """
        pass
```

### Diagnostic Controls (Agent)

```python
class DiagnosticController:
    """
    Low-level controls for individual agent management.
    These are "surgical" tools - precise, immediate-acting.
    """
    
    def quarantine(self, agent_id: str, duration_ticks: int = None):
        """
        Isolate agent from swarm participation.
        
        The agent continues to run but cannot:
        - Vote on consensus
        - Propagate drift to children
        - Receive resource allocation
        """
        pass
    
    def release(self, agent_id: str):
        """
        Return agent to active pool from quarantine.
        
        Requires:
        - Stability > threshold
        - TTF > minimum
        - Manual approval if previously escalated
        """
        pass
    
    def terminate(self, agent_id: str, preserve_children: bool = True):
        """
        Permanently remove agent.
        
        Options:
        - preserve_children=True: Orphan protocol runs, children adopted
        - preserve_children=False: Entire lineage terminated
        """
        pass
    
    def rebuild(self, agent_id: str, snapshot_id: str):
        """
        Restore agent to a previous known-good state.
        
        Requires valid snapshot in audit log.
        """
        pass
```

---

## Alert System

### Alert Priorities

| Priority | Color | Sound | Action Required |
|----------|-------|-------|-----------------|
| P1 - Critical | 🔴 Red | Alarm | Immediate (< 1 min) |
| P2 - High | 🟠 Orange | Chime | Soon (< 5 min) |
| P3 - Medium | 🟡 Yellow | None | Today |
| P4 - Low | 🔵 Blue | None | This week |

### Alert Types

```python
ALERT_DEFINITIONS = {
    "SWARM_COHERENCE_LOSS": {
        "priority": "P1",
        "trigger": "Global SCI < 0.5",
        "suggested_action": "Declare WINTER season, investigate drift clusters"
    },
    "MASS_QUARANTINE": {
        "priority": "P1",
        "trigger": "> 10% of agents quarantined",
        "suggested_action": "Check for attack vector, review recent changes"
    },
    "CONSENSUS_DEADLOCK": {
        "priority": "P2",
        "trigger": "Proposal pending > 100 ticks without quorum",
        "suggested_action": "Force quorum or override vote"
    },
    "BRANCH_EXTINCTION": {
        "priority": "P2",
        "trigger": "Major branch (> 1000 agents) approaching termination",
        "suggested_action": "Fertilize or graft before loss"
    },
    "DISCOVERY_DROUGHT": {
        "priority": "P3",
        "trigger": "FDR < threshold for > 500 ticks",
        "suggested_action": "Terraform to new data sectors"
    },
    "LINEAGE_DEPTH_WARNING": {
        "priority": "P4",
        "trigger": "Branch exceeds 50 generations",
        "suggested_action": "Consider pruning or promoting to new trunk"
    }
}
```

---

## The Observer's Creed

The Gardener Protocol is built on three principles:

### 1. Observe, Don't Control

At scale, direct control creates more problems than it solves. The gardener's job is to:
- Watch for patterns
- Identify systemic issues
- Adjust the environment
- Trust the system's self-governance

### 2. Shape the Climate, Not the Weather

You cannot prevent individual failures. You can:
- Set global parameters that influence behavior
- Create conditions for healthy growth
- Remove obstacles to recovery

### 3. Harvest, Don't Extract

Discoveries are the fruit of the forest. The gardener:
- Waits for consensus-validated findings
- Promotes successful patterns via grafting
- Lets the CDP recycle failed experiments

---

## Technical Implementation

### Frontend Stack

```yaml
framework: React + Three.js
visualization:
  canopy: WebGL 3D terrain rendering
  trunk: D3.js force-directed tree
  branch: Chart.js for voting bars
  leaf: Custom diagnostic panels
  
real_time:
  protocol: WebSocket
  update_rate: 100ms for critical metrics
  batch_rate: 1s for aggregate stats
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/garden/canopy` | GET | Global health heatmap data |
| `/garden/trunk/{cluster_id}` | GET | Phylogenetic tree for cluster |
| `/garden/branch/{branch_id}` | GET | Consensus state for branch |
| `/garden/leaf/{agent_id}` | GET | Full diagnostic for agent |
| `/garden/action/terraform` | POST | Execute terraform operation |
| `/garden/action/prune` | POST | Execute prune operation |
| `/garden/action/season` | POST | Declare season change |
| `/garden/alerts` | GET | Current alert queue |
| `/garden/alerts/{id}/ack` | POST | Acknowledge alert |

---

## Summary

The Gardener Protocol transforms the impossible task of managing a million autonomous agents into the manageable art of cultivating a digital forest. 

By operating at the right scale:
- **Canopy**: Shape the environment
- **Trunk**: Guide the growth
- **Branch**: Resolve conflicts
- **Leaf**: Fix individual failures

The operator becomes a curator of emergence rather than a controller of behavior.

---

*"A gardener does not command each leaf to grow. They ensure the soil is rich, the water flows, and the sun reaches where it's needed. The forest does the rest."*
