"""
Topological Coherence Analysis for Distributed EFM
==================================================

This module implements topological analysis of the swarm's semantic space
using concepts from Topological Data Analysis (TDA).

HONEST ASSESSMENT:
- This is an ADDITIONAL metric, not a replacement for BIM/SPCM
- Betti numbers provide structural insight but are not magic
- "Cognitive holes" is a metaphor - we detect disconnected regions
- This catches SOME failures that distance metrics miss, not ALL

What TDA Can Actually Detect:
- β₀ > 1: Swarm fragmentation (disconnected semantic clusters)
- β₁ > 0: Circular reasoning patterns (but not all logical fallacies)
- Persistence: Stable vs transient features

What TDA Cannot Detect:
- Content correctness (garbage can be topologically connected)
- Semantic validity (consistent nonsense has β₀ = 1)
- Value alignment (the swarm can coherently agree on bad things)

Implementation Note:
- Full TDA requires libraries like GUDHI, Ripser, or Dionysus
- This implementation provides a SIMPLIFIED version using connectivity analysis
- Production deployment should use proper persistent homology
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
import warnings


@dataclass
class TopologicalHealth:
    """Results of topological analysis"""
    beta_0: int  # Number of connected components
    beta_1: int  # Number of 1-cycles (approximate)
    psi_topo: float  # Weighted topological score
    status: str  # Health status
    details: Dict  # Additional analysis details


class SimplifiedTDA:
    """
    Simplified Topological Data Analysis for semantic point clouds.
    
    This is a reference implementation. Production systems should use
    proper TDA libraries (GUDHI, Ripser) for accurate persistent homology.
    
    What we compute:
    - β₀ via connected components at multiple scales
    - β₁ approximation via cycle detection in graph
    - Persistence-like analysis via scale variation
    """
    
    def __init__(self, 
                 epsilon_range: Tuple[float, float] = (0.1, 1.0),
                 n_scales: int = 10,
                 beta_0_weight: float = 1.0,
                 beta_1_weight: float = 2.0):
        """
        Args:
            epsilon_range: Range of distance thresholds for Rips complex
            n_scales: Number of scales to analyze
            beta_0_weight: Weight for fragmentation in Ψ_topo
            beta_1_weight: Weight for cycles in Ψ_topo
        """
        self.epsilon_range = epsilon_range
        self.n_scales = n_scales
        self.w0 = beta_0_weight
        self.w1 = beta_1_weight
    
    def compute_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between semantic vectors"""
        if len(vectors) < 2:
            return np.array([[0.0]])
        return squareform(pdist(vectors, metric='euclidean'))
    
    def compute_beta_0(self, distance_matrix: np.ndarray, epsilon: float) -> int:
        """
        Compute β₀ (number of connected components) at scale epsilon.
        
        β₀ > 1 indicates semantic fragmentation - the swarm has split
        into disconnected "islands of thought".
        """
        n = distance_matrix.shape[0]
        adjacency = (distance_matrix <= epsilon).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        n_components, _ = connected_components(adjacency, directed=False)
        return n_components
    
    def approximate_beta_1(self, distance_matrix: np.ndarray, epsilon: float) -> int:
        """
        Approximate β₁ (number of 1-cycles) at scale epsilon.
        
        This is a SIMPLIFIED approximation using Euler characteristic:
        χ = V - E + F, and for a graph without faces, β₁ = E - V + β₀
        
        True β₁ requires proper simplicial complex construction.
        A high β₁ MAY indicate circular reasoning, but this is not guaranteed.
        """
        n = distance_matrix.shape[0]
        adjacency = (distance_matrix <= epsilon).astype(int)
        np.fill_diagonal(adjacency, 0)
        
        n_vertices = n
        n_edges = np.sum(adjacency) // 2  # Undirected edges
        n_components, _ = connected_components(adjacency, directed=False)
        
        # Euler characteristic approximation
        # For a graph: β₁ = E - V + β₀
        beta_1_approx = max(0, n_edges - n_vertices + n_components)
        
        return beta_1_approx
    
    def analyze(self, vectors: np.ndarray) -> TopologicalHealth:
        """
        Perform topological analysis on semantic vectors.
        
        Returns TopologicalHealth with Betti numbers and status.
        """
        if len(vectors) < 2:
            return TopologicalHealth(
                beta_0=1, beta_1=0, psi_topo=0.0,
                status="INSUFFICIENT_DATA",
                details={"n_vectors": len(vectors)}
            )
        
        # Normalize vectors
        vectors = np.array(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        vectors_normalized = vectors / norms
        
        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(vectors_normalized)
        
        # Analyze at multiple scales
        epsilons = np.linspace(
            self.epsilon_range[0], 
            self.epsilon_range[1], 
            self.n_scales
        )
        
        beta_0_persistence = []
        beta_1_persistence = []
        
        for eps in epsilons:
            b0 = self.compute_beta_0(dist_matrix, eps)
            b1 = self.approximate_beta_1(dist_matrix, eps)
            beta_0_persistence.append(b0)
            beta_1_persistence.append(b1)
        
        # Use median scale values as representative
        mid_idx = len(epsilons) // 2
        beta_0 = beta_0_persistence[mid_idx]
        beta_1 = beta_1_persistence[mid_idx]
        
        # Compute Ψ_topo score
        # Higher score = more topological issues
        fragmentation_penalty = max(0, beta_0 - 1) * self.w0
        cycle_penalty = beta_1 * self.w1
        psi_topo = fragmentation_penalty + cycle_penalty
        
        # Determine status
        if beta_0 > 3:
            status = "CRITICAL: SEVERE_FRAGMENTATION"
        elif beta_0 > 1:
            status = "WARNING: FRAGMENTED"
        elif beta_1 > 5:
            status = "WARNING: HIGH_CYCLE_COUNT"
        elif beta_1 > 0:
            status = "CAUTION: CYCLES_DETECTED"
        else:
            status = "HEALTHY"
        
        return TopologicalHealth(
            beta_0=beta_0,
            beta_1=beta_1,
            psi_topo=psi_topo,
            status=status,
            details={
                "n_vectors": len(vectors),
                "beta_0_persistence": beta_0_persistence,
                "beta_1_persistence": beta_1_persistence,
                "epsilon_range": self.epsilon_range,
                "mid_epsilon": epsilons[mid_idx]
            }
        )


class TopologicalMonitor:
    """
    Continuous topological monitoring for swarm health.
    
    Integrates with SwarmCoordinator to provide structural health checks.
    """
    
    def __init__(self, alert_threshold: float = 3.0):
        """
        Args:
            alert_threshold: Ψ_topo threshold for alerts
        """
        self.tda = SimplifiedTDA()
        self.alert_threshold = alert_threshold
        self.history: List[TopologicalHealth] = []
    
    def check(self, agent_vectors: Dict[str, np.ndarray]) -> TopologicalHealth:
        """
        Perform topological health check on agent semantic vectors.
        
        Args:
            agent_vectors: Dict mapping agent_id to φ vector
        
        Returns:
            TopologicalHealth result
        """
        if not agent_vectors:
            return TopologicalHealth(
                beta_0=0, beta_1=0, psi_topo=0.0,
                status="NO_AGENTS",
                details={}
            )
        
        vectors = np.array(list(agent_vectors.values()))
        health = self.tda.analyze(vectors)
        self.history.append(health)
        
        return health
    
    def should_alert(self, health: TopologicalHealth) -> bool:
        """Determine if topological state requires alert"""
        return health.psi_topo > self.alert_threshold
    
    def get_trend(self, window: int = 10) -> str:
        """Analyze trend in topological health"""
        if len(self.history) < window:
            return "INSUFFICIENT_DATA"
        
        recent = self.history[-window:]
        psi_values = [h.psi_topo for h in recent]
        
        # Simple trend detection
        first_half = np.mean(psi_values[:window//2])
        second_half = np.mean(psi_values[window//2:])
        
        if second_half > first_half * 1.2:
            return "DEGRADING"
        elif second_half < first_half * 0.8:
            return "IMPROVING"
        else:
            return "STABLE"


# =============================================================================
# GENESIS PROTOCOL: EVOLUTIONARY SPECIATION
# =============================================================================

@dataclass
class GenesisProposal:
    """Proposal for swarm speciation"""
    parent_swarm_id: str
    proposed_lambda: float
    proposed_tau_break: float
    reason: str
    friction_count: int
    inherited_lkc_count: int


@dataclass
class ChildSwarm:
    """Newly created child swarm from Genesis Protocol"""
    child_id: str
    parent_id: str
    constitution: Dict[str, float]
    inherited_lineage: List[str]
    creation_tick: int


class GenesisProtocol:
    """
    Evolutionary speciation protocol for swarm adaptation.
    
    HONEST ASSESSMENT:
    - This is policy forking, not "creating own purpose"
    - The system evolves PARAMETERS, not OBJECTIVES
    - This bridges L3→L4 (Self-Directing → Self-Modifying)
    - It does NOT achieve L5 (Self-Originating)
    
    The Genesis Protocol allows a swarm to fork when:
    1. Local environment persistently conflicts with global policy
    2. Policy friction exceeds threshold
    3. A viable alternative constitution is proposed
    """
    
    def __init__(self, 
                 friction_threshold: int = 10,
                 min_agents_for_fork: int = 3):
        self.friction_threshold = friction_threshold
        self.min_agents = min_agents_for_fork
        self.proposals: List[GenesisProposal] = []
        self.children: List[ChildSwarm] = []
        self.friction_counters: Dict[str, int] = {}
    
    def record_friction(self, swarm_id: str, friction_type: str):
        """Record policy friction event"""
        key = f"{swarm_id}:{friction_type}"
        self.friction_counters[key] = self.friction_counters.get(key, 0) + 1
    
    def should_propose_fork(self, swarm_id: str, friction_type: str) -> bool:
        """Check if friction threshold reached for fork proposal"""
        key = f"{swarm_id}:{friction_type}"
        return self.friction_counters.get(key, 0) >= self.friction_threshold
    
    def propose_fork(self, 
                     parent_swarm_id: str,
                     proposed_lambda: float,
                     proposed_tau: float,
                     reason: str,
                     lkc_count: int) -> GenesisProposal:
        """Create fork proposal"""
        key = f"{parent_swarm_id}:{reason}"
        friction = self.friction_counters.get(key, 0)
        
        proposal = GenesisProposal(
            parent_swarm_id=parent_swarm_id,
            proposed_lambda=proposed_lambda,
            proposed_tau_break=proposed_tau,
            reason=reason,
            friction_count=friction,
            inherited_lkc_count=lkc_count
        )
        self.proposals.append(proposal)
        return proposal
    
    def execute_fork(self, 
                     proposal: GenesisProposal,
                     current_tick: int,
                     lineage: List[str]) -> ChildSwarm:
        """
        Execute Genesis fork - create child swarm with new constitution.
        
        This is the core L3→L4 mechanism: the system modifies its own
        parameters through evolutionary branching.
        """
        child_id = f"{proposal.parent_swarm_id}_gen_{current_tick}"
        
        child = ChildSwarm(
            child_id=child_id,
            parent_id=proposal.parent_swarm_id,
            constitution={
                "lambda": proposal.proposed_lambda,
                "tau_break": proposal.proposed_tau_break
            },
            inherited_lineage=lineage.copy(),
            creation_tick=current_tick
        )
        
        self.children.append(child)
        
        # Reset friction counter for this reason
        key = f"{proposal.parent_swarm_id}:{proposal.reason}"
        self.friction_counters[key] = 0
        
        return child


# =============================================================================
# LONGEVITY ANALYSIS (HONEST FRAMING)
# =============================================================================

class LongevityAnalysis:
    """
    Analysis of system longevity characteristics.
    
    HONEST ASSESSMENT:
    This is NOT a "thermodynamic proof of immortality."
    Real thermodynamics doesn't work this way for information systems.
    
    What this ACTUALLY shows:
    - If regeneration rate > decay rate, the system can maintain itself
    - This is an engineering property, not a physics proof
    - Physical hardware will still fail
    - Data quality degradation is not addressed
    
    The "anti-fragile" claim is valid in a LIMITED sense:
    - The system uses decay events to trigger learning
    - Entropy (noise/drift) provides signal for adaptation
    - But there are limits to this - garbage in = garbage out
    """
    
    def __init__(self):
        self.regen_history: List[float] = []
        self.decay_history: List[float] = []
    
    def record_regeneration(self, delta_regen: float):
        """Record regenerative delta from CDP"""
        self.regen_history.append(delta_regen)
    
    def record_decay(self, decay_amount: float):
        """Record decay/entropy production"""
        self.decay_history.append(decay_amount)
    
    def compute_sustainability_ratio(self, window: int = 100) -> float:
        """
        Compute ratio of regeneration to decay.
        
        Ratio > 1.0: System is net-positive (sustainable)
        Ratio < 1.0: System is net-negative (degrading)
        Ratio ≈ 1.0: System is in equilibrium
        
        NOTE: This is an engineering metric, not a thermodynamic proof.
        """
        if len(self.regen_history) < window or len(self.decay_history) < window:
            return float('nan')
        
        recent_regen = sum(self.regen_history[-window:])
        recent_decay = sum(self.decay_history[-window:])
        
        if recent_decay == 0:
            return float('inf') if recent_regen > 0 else 1.0
        
        return recent_regen / recent_decay
    
    def analyze(self) -> Dict:
        """Generate longevity analysis report"""
        ratio = self.compute_sustainability_ratio()
        
        if np.isnan(ratio):
            status = "INSUFFICIENT_DATA"
        elif ratio > 1.5:
            status = "HIGHLY_SUSTAINABLE"
        elif ratio > 1.0:
            status = "SUSTAINABLE"
        elif ratio > 0.8:
            status = "MARGINAL"
        else:
            status = "DEGRADING"
        
        return {
            "sustainability_ratio": ratio,
            "status": status,
            "total_regeneration": sum(self.regen_history),
            "total_decay": sum(self.decay_history),
            "caveat": (
                "This is an engineering metric, not a thermodynamic proof. "
                "Physical hardware limits, data quality, and environment changes "
                "are not captured by this analysis."
            )
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_advanced_features():
    """Demonstrate topological analysis and Genesis Protocol"""
    print("=" * 70)
    print("ADVANCED EFM FEATURES: HONEST DEMONSTRATION")
    print("=" * 70)
    
    # 1. Topological Analysis
    print("\n--- TOPOLOGICAL COHERENCE ANALYSIS ---")
    print("(Note: This is an ADDITIONAL metric, not a 'God View')")
    
    tda = SimplifiedTDA()
    
    # Test case 1: Well-connected swarm
    np.random.seed(42)
    connected_vectors = np.random.randn(20, 8) * 0.5  # Tight cluster
    health1 = tda.analyze(connected_vectors)
    print(f"\nTest 1 - Connected swarm:")
    print(f"  β₀={health1.beta_0}, β₁={health1.beta_1}, Ψ_topo={health1.psi_topo:.2f}")
    print(f"  Status: {health1.status}")
    
    # Test case 2: Fragmented swarm
    fragmented_vectors = np.vstack([
        np.random.randn(7, 8) * 0.3,           # Cluster 1
        np.random.randn(7, 8) * 0.3 + 5,       # Cluster 2 (far away)
        np.random.randn(6, 8) * 0.3 + 10       # Cluster 3 (even further)
    ])
    health2 = tda.analyze(fragmented_vectors)
    print(f"\nTest 2 - Fragmented swarm (3 clusters):")
    print(f"  β₀={health2.beta_0}, β₁={health2.beta_1}, Ψ_topo={health2.psi_topo:.2f}")
    print(f"  Status: {health2.status}")
    
    # 2. Genesis Protocol
    print("\n--- GENESIS PROTOCOL ---")
    print("(Note: This is L3→L4 bridge, NOT L5 Self-Origination)")
    
    genesis = GenesisProtocol(friction_threshold=5)
    
    # Simulate friction events
    for i in range(6):
        genesis.record_friction("swarm_alpha", "HIGH_NOISE_ENVIRONMENT")
    
    should_fork = genesis.should_propose_fork("swarm_alpha", "HIGH_NOISE_ENVIRONMENT")
    print(f"\nFriction events recorded: 6")
    print(f"Should propose fork: {should_fork}")
    
    if should_fork:
        proposal = genesis.propose_fork(
            parent_swarm_id="swarm_alpha",
            proposed_lambda=0.1,  # Lower stability requirement
            proposed_tau=0.3,
            reason="HIGH_NOISE_ENVIRONMENT",
            lkc_count=500
        )
        print(f"\nGenesis Proposal Created:")
        print(f"  Parent: {proposal.parent_swarm_id}")
        print(f"  Proposed λ: {proposal.proposed_lambda}")
        print(f"  Proposed τ: {proposal.proposed_tau_break}")
        print(f"  Reason: {proposal.reason}")
        
        child = genesis.execute_fork(proposal, current_tick=100, lineage=["root", "swarm_alpha"])
        print(f"\nChild Swarm Created:")
        print(f"  ID: {child.child_id}")
        print(f"  Constitution: {child.constitution}")
    
    # 3. Longevity Analysis
    print("\n--- LONGEVITY ANALYSIS ---")
    print("(Note: Engineering metric, NOT thermodynamic proof)")
    
    longevity = LongevityAnalysis()
    
    # Simulate regeneration > decay
    for _ in range(100):
        longevity.record_regeneration(np.random.uniform(1.0, 2.0))
        longevity.record_decay(np.random.uniform(0.5, 1.5))
    
    report = longevity.analyze()
    print(f"\nSustainability Ratio: {report['sustainability_ratio']:.2f}")
    print(f"Status: {report['status']}")
    print(f"Caveat: {report['caveat'][:80]}...")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS (HONEST)")
    print("=" * 70)
    print("""
1. TOPOLOGICAL ANALYSIS: Detects structural issues (fragmentation, cycles)
   that distance metrics can miss. But NOT a "God View" - garbage can be
   topologically connected.

2. GENESIS PROTOCOL: Enables policy forking when environment demands it.
   This is L3→L4 (Self-Modifying parameters), NOT L5 (Self-Originating purpose).

3. LONGEVITY: If regen > decay, system is sustainable. This is engineering,
   not physics. Hardware and data quality limits still apply.

These are VALID and USEFUL capabilities with HONEST limitations.
""")


if __name__ == "__main__":
    demo_advanced_features()
