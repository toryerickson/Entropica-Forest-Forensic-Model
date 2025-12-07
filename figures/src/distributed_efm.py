"""
Entropica Forensic Model - Booklet 4: Cognitive Genealogy and Distributed Swarm Autonomy
========================================================================================

This module implements the distributed extension of EFM:
- d-CTM: Decentralized Cognitive Trace Memory
- IA-BIM: Inter-Agent Bridge Integrity Matrix
- Hierarchical ZK-SP: Recursive proof aggregation
- Orphan Protocol: Lost capsule detection and recovery
- Swarm Consensus: Byzantine fault tolerant semantic agreement

The central insight: Cognitive integrity is a distributed systems problem.
"""

import numpy as np
import hashlib
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum, auto
from collections import defaultdict
import json

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class AgentStatus(Enum):
    """Agent health status in swarm"""
    ACTIVE = auto()
    DEGRADED = auto()
    PARTITIONED = auto()
    ORPHANED = auto()
    BYZANTINE = auto()

class ConsensusState(Enum):
    """BFT consensus states"""
    PREPARING = auto()
    PREPARED = auto()
    COMMITTED = auto()
    FINALIZED = auto()

@dataclass
class SwarmCapsule:
    """Extended capsule with swarm metadata"""
    # Core capsule fields (from B1)
    capsule_id: str
    agent_id: str
    phi: np.ndarray  # Semantic state vector
    stability: float
    entropy: float
    timestamp: float
    
    # Genealogy fields (B4)
    parent_agent_id: Optional[str] = None
    parent_capsule_id: Optional[str] = None
    generation: int = 0
    spawn_reason: str = "root"
    
    # Swarm fields (B4)
    cluster_id: str = "default"
    local_sequence: int = 0
    global_sequence: Optional[int] = None
    consensus_hash: Optional[str] = None
    
    @property
    def drift_risk(self) -> float:
        return self.entropy * (1 - self.stability)
    
    @property
    def health_status(self) -> AgentStatus:
        dr = self.drift_risk
        if dr < 0.2:
            return AgentStatus.ACTIVE
        elif dr < 0.5:
            return AgentStatus.DEGRADED
        elif dr < 0.8:
            return AgentStatus.PARTITIONED
        else:
            return AgentStatus.BYZANTINE
    
    def fingerprint(self) -> str:
        """Cryptographic fingerprint for ZK-SP"""
        data = f"{self.capsule_id}:{self.agent_id}:{self.phi.tobytes().hex()}:{self.stability}:{self.entropy}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class Agent:
    """Autonomous agent in the swarm"""
    agent_id: str
    cluster_id: str
    parent_agent_id: Optional[str] = None
    generation: int = 0
    
    # Local state
    phi: np.ndarray = field(default_factory=lambda: np.random.randn(64))
    stability: float = 0.95
    entropy: float = 0.1
    
    # CAC state (local)
    trace_level: int = 2
    alpha_dyn: float = 0.2
    
    # Children
    children: List[str] = field(default_factory=list)
    
    # Local capsule sequence
    local_sequence: int = 0
    
    def spawn_child(self, reason: str = "task_delegation") -> 'Agent':
        """Create child agent inheriting cognitive state"""
        child_id = f"{self.agent_id}.{len(self.children)}"
        child = Agent(
            agent_id=child_id,
            cluster_id=self.cluster_id,
            parent_agent_id=self.agent_id,
            generation=self.generation + 1,
            phi=self.phi.copy() + np.random.randn(64) * 0.01,  # Small variation
            stability=self.stability * 0.98,  # Slight degradation
            entropy=self.entropy * 1.02,
        )
        self.children.append(child_id)
        return child
    
    def create_capsule(self) -> SwarmCapsule:
        """Generate capsule from current state"""
        self.local_sequence += 1
        return SwarmCapsule(
            capsule_id=f"{self.agent_id}:cap:{self.local_sequence}",
            agent_id=self.agent_id,
            phi=self.phi.copy(),
            stability=self.stability,
            entropy=self.entropy,
            timestamp=time.time(),
            parent_agent_id=self.parent_agent_id,
            generation=self.generation,
            cluster_id=self.cluster_id,
            local_sequence=self.local_sequence
        )


# =============================================================================
# INTER-AGENT BRIDGE INTEGRITY MATRIX (IA-BIM)
# =============================================================================

class InterAgentBIM:
    """
    Extends BIM from capsule-level to agent-level.
    Measures semantic coherence between agents in the swarm.
    
    Key insight: W_ij between agents tells us if the swarm has consensus.
    """
    
    def __init__(self, tau_break: float = 0.5, lambda_decay: float = 0.95):
        self.tau_break = tau_break
        self.lambda_decay = lambda_decay
        self.integrity_cache: Dict[Tuple[str, str], float] = {}
        self.history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    
    def compute_w(self, agent_i: Agent, agent_j: Agent) -> float:
        """
        Compute semantic bridge weight between two agents.
        
        W_ij = exp(-||φ_i - φ_j||² / (2σ²)) × min(S_i, S_j)
        
        where σ is scaled by combined entropy.
        """
        phi_diff = agent_i.phi - agent_j.phi
        distance_sq = np.dot(phi_diff, phi_diff)
        
        # Scale by combined entropy (higher entropy = more tolerance)
        sigma_sq = 2.0 * (1 + agent_i.entropy + agent_j.entropy)
        
        similarity = np.exp(-distance_sq / sigma_sq)
        stability_factor = min(agent_i.stability, agent_j.stability)
        
        w_ij = similarity * stability_factor
        
        # Cache and track history
        key = tuple(sorted([agent_i.agent_id, agent_j.agent_id]))
        self.integrity_cache[key] = w_ij
        self.history[key].append(w_ij)
        
        return w_ij
    
    def is_link_broken(self, agent_i: Agent, agent_j: Agent) -> bool:
        """Check if semantic link between agents is broken"""
        w = self.compute_w(agent_i, agent_j)
        return w < self.tau_break
    
    def compute_swarm_matrix(self, agents: List[Agent]) -> np.ndarray:
        """Compute full W matrix for swarm"""
        n = len(agents)
        W = np.eye(n)  # Self-links are 1.0
        
        for i in range(n):
            for j in range(i + 1, n):
                w_ij = self.compute_w(agents[i], agents[j])
                W[i, j] = w_ij
                W[j, i] = w_ij
        
        return W
    
    def detect_consensus_loss(self, agents: List[Agent], 
                               threshold: float = 0.5) -> List[Tuple[str, str]]:
        """Find all broken links in swarm (consensus loss)"""
        broken = []
        n = len(agents)
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.is_link_broken(agents[i], agents[j]):
                    broken.append((agents[i].agent_id, agents[j].agent_id))
        
        return broken
    
    def compute_spcm_swarm(self, agents: List[Agent]) -> float:
        """
        Systemic Pre-Collapse Metric for entire swarm.
        
        SPCM_swarm = 1 - (Σ W_ij / n(n-1)/2)
        
        High SPCM means swarm is losing consensus.
        """
        W = self.compute_swarm_matrix(agents)
        n = len(agents)
        
        if n < 2:
            return 0.0
        
        # Sum of off-diagonal elements
        total_w = (np.sum(W) - n) / 2  # Subtract diagonal, divide by 2 for symmetry
        max_links = n * (n - 1) / 2
        
        avg_w = total_w / max_links if max_links > 0 else 1.0
        
        return 1.0 - avg_w


# =============================================================================
# DECENTRALIZED CTM (d-CTM)
# =============================================================================

class LocalCTMNode:
    """
    Local CTM instance for a cluster of agents.
    Participates in BFT consensus for global parameters.
    """
    
    def __init__(self, node_id: str, cluster_id: str):
        self.node_id = node_id
        self.cluster_id = cluster_id
        
        # Local storage
        self.capsules: Dict[str, SwarmCapsule] = {}
        self.agents: Dict[str, Agent] = {}
        self.local_sequence: int = 0
        
        # Global state (from consensus)
        self.global_sequence: int = 0
        self.global_tau_break: float = 0.5
        self.global_lambda: float = 0.95
        
        # Consensus state
        self.pending_proposals: Dict[str, dict] = {}
        self.votes: Dict[str, Set[str]] = defaultdict(set)
        
        # CAC state (local)
        self.trace_level: int = 2
        self.alpha_dyn: float = 0.2
        
        # BIM instance
        self.ia_bim = InterAgentBIM(self.global_tau_break, self.global_lambda)
    
    def register_agent(self, agent: Agent):
        """Register agent with local CTM"""
        self.agents[agent.agent_id] = agent
    
    def store_capsule(self, capsule: SwarmCapsule) -> int:
        """Store capsule and assign local sequence number"""
        self.local_sequence += 1
        capsule.local_sequence = self.local_sequence
        self.capsules[capsule.capsule_id] = capsule
        return self.local_sequence
    
    def get_lineage(self, agent_id: str) -> List[SwarmCapsule]:
        """Get capsule lineage for agent"""
        lineage = []
        current_id = agent_id
        
        while current_id:
            agent_capsules = [c for c in self.capsules.values() 
                            if c.agent_id == current_id]
            lineage.extend(sorted(agent_capsules, key=lambda c: c.timestamp))
            
            agent = self.agents.get(current_id)
            current_id = agent.parent_agent_id if agent else None
        
        return lineage
    
    def propose_parameter_update(self, param: str, value: float, 
                                  proposer: str) -> str:
        """Propose global parameter update (BFT consensus)"""
        proposal_id = hashlib.sha256(
            f"{param}:{value}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        self.pending_proposals[proposal_id] = {
            "param": param,
            "value": value,
            "proposer": proposer,
            "state": ConsensusState.PREPARING,
            "timestamp": time.time()
        }
        
        return proposal_id
    
    def vote(self, proposal_id: str, voter_id: str, approve: bool):
        """Vote on parameter proposal"""
        if approve and proposal_id in self.pending_proposals:
            self.votes[proposal_id].add(voter_id)
    
    def check_consensus(self, proposal_id: str, total_nodes: int) -> bool:
        """Check if proposal has BFT consensus (2f+1 votes for n=3f+1)"""
        votes = len(self.votes.get(proposal_id, set()))
        required = (2 * total_nodes) // 3 + 1
        return votes >= required
    
    def apply_consensus(self, proposal_id: str):
        """Apply approved parameter update"""
        if proposal_id not in self.pending_proposals:
            return
        
        proposal = self.pending_proposals[proposal_id]
        param = proposal["param"]
        value = proposal["value"]
        
        if param == "tau_break":
            self.global_tau_break = value
            self.ia_bim.tau_break = value
        elif param == "lambda":
            self.global_lambda = value
            self.ia_bim.lambda_decay = value
        
        proposal["state"] = ConsensusState.FINALIZED
        self.global_sequence += 1


class DecentralizedCTM:
    """
    Distributed CTM coordinator.
    Manages multiple LocalCTMNode instances with BFT consensus.
    """
    
    def __init__(self):
        self.nodes: Dict[str, LocalCTMNode] = {}
        self.global_sequence: int = 0
        
    def add_node(self, node: LocalCTMNode):
        """Add local CTM node to network"""
        self.nodes[node.node_id] = node
    
    def broadcast_capsule(self, capsule: SwarmCapsule, source_node: str):
        """Broadcast capsule to all nodes (eventual consistency)"""
        for node_id, node in self.nodes.items():
            if node_id != source_node:
                node.store_capsule(capsule)
    
    def initiate_consensus(self, param: str, value: float, 
                           proposer_node: str) -> str:
        """Start BFT consensus for parameter update"""
        proposal_id = self.nodes[proposer_node].propose_parameter_update(
            param, value, proposer_node
        )
        
        # Broadcast proposal to all nodes
        for node_id, node in self.nodes.items():
            if node_id != proposer_node:
                node.pending_proposals[proposal_id] = \
                    self.nodes[proposer_node].pending_proposals[proposal_id].copy()
        
        return proposal_id
    
    def collect_votes(self, proposal_id: str) -> Dict[str, bool]:
        """Collect votes from all nodes (simplified - real BFT is more complex)"""
        votes = {}
        for node_id, node in self.nodes.items():
            # Simplified: each node votes based on local health
            local_agents = list(node.agents.values())
            if local_agents:
                avg_stability = np.mean([a.stability for a in local_agents])
                votes[node_id] = avg_stability > 0.5
            else:
                votes[node_id] = True
        return votes
    
    def finalize_consensus(self, proposal_id: str) -> bool:
        """Finalize consensus across all nodes"""
        votes = self.collect_votes(proposal_id)
        
        for node_id, node in self.nodes.items():
            for voter, approved in votes.items():
                node.vote(proposal_id, voter, approved)
        
        # Check if consensus achieved
        total = len(self.nodes)
        consensus = all(
            node.check_consensus(proposal_id, total) 
            for node in self.nodes.values()
        )
        
        if consensus:
            for node in self.nodes.values():
                node.apply_consensus(proposal_id)
            self.global_sequence += 1
            return True
        
        return False


# =============================================================================
# ORPHAN PROTOCOL
# =============================================================================

class OrphanClassification(Enum):
    """Types of orphan conditions"""
    PARENT_DEAD = auto()      # Parent agent terminated
    PARENT_CORRUPTED = auto() # Parent has high drift risk
    LINEAGE_BROKEN = auto()   # Cannot trace to root
    NETWORK_PARTITION = auto() # Temporarily unreachable

@dataclass
class OrphanRecord:
    """Record of detected orphan"""
    agent_id: str
    classification: OrphanClassification
    detected_at: float
    last_known_parent: Optional[str]
    recovery_attempts: int = 0
    adopted_by: Optional[str] = None

class OrphanProtocol:
    """
    Detect, quarantine, and recover orphaned agents.
    
    An agent is orphaned when:
    1. Its parent agent terminates or fails
    2. Its parent's drift risk exceeds threshold
    3. Its lineage chain is broken (missing ancestors)
    4. Network partition isolates it from cluster
    """
    
    def __init__(self, ctm: DecentralizedCTM, corruption_threshold: float = 0.7):
        self.ctm = ctm
        self.corruption_threshold = corruption_threshold
        self.orphans: Dict[str, OrphanRecord] = {}
        self.quarantine: Set[str] = set()
        self.adoption_queue: List[str] = []
    
    def detect_orphans(self, agents: Dict[str, Agent]) -> List[OrphanRecord]:
        """Scan for orphaned agents"""
        newly_detected = []
        
        for agent_id, agent in agents.items():
            if agent.parent_agent_id is None:
                continue  # Root agents can't be orphans
            
            classification = None
            
            # Check if parent exists
            if agent.parent_agent_id not in agents:
                classification = OrphanClassification.PARENT_DEAD
            else:
                parent = agents[agent.parent_agent_id]
                
                # Check parent corruption
                if parent.entropy * (1 - parent.stability) > self.corruption_threshold:
                    classification = OrphanClassification.PARENT_CORRUPTED
            
            # Check lineage integrity
            if classification is None:
                if not self._verify_lineage(agent, agents):
                    classification = OrphanClassification.LINEAGE_BROKEN
            
            if classification and agent_id not in self.orphans:
                record = OrphanRecord(
                    agent_id=agent_id,
                    classification=classification,
                    detected_at=time.time(),
                    last_known_parent=agent.parent_agent_id
                )
                self.orphans[agent_id] = record
                newly_detected.append(record)
        
        return newly_detected
    
    def _verify_lineage(self, agent: Agent, agents: Dict[str, Agent], 
                        max_depth: int = 10) -> bool:
        """Verify lineage chain to root"""
        current = agent
        depth = 0
        
        while current.parent_agent_id and depth < max_depth:
            if current.parent_agent_id not in agents:
                return False
            current = agents[current.parent_agent_id]
            depth += 1
        
        return current.parent_agent_id is None  # Reached root
    
    def quarantine_agent(self, agent_id: str):
        """Quarantine orphaned agent to prevent contagion"""
        self.quarantine.add(agent_id)
        if agent_id in self.orphans:
            self.orphans[agent_id].recovery_attempts += 1
    
    def attempt_adoption(self, orphan_id: str, adopter_id: str, 
                         agents: Dict[str, Agent]) -> bool:
        """
        Attempt to adopt orphan into healthy lineage.
        
        Requirements:
        1. Adopter must be healthy (drift risk < 0.3)
        2. Adopter must have semantic compatibility (W > 0.6)
        3. Orphan must not be Byzantine
        """
        if orphan_id not in agents or adopter_id not in agents:
            return False
        
        orphan = agents[orphan_id]
        adopter = agents[adopter_id]
        
        # Check adopter health
        adopter_risk = adopter.entropy * (1 - adopter.stability)
        if adopter_risk > 0.3:
            return False
        
        # Check semantic compatibility via IA-BIM
        ia_bim = InterAgentBIM()
        w = ia_bim.compute_w(orphan, adopter)
        if w < 0.6:
            return False
        
        # Check orphan isn't Byzantine
        if orphan_id in self.orphans:
            record = self.orphans[orphan_id]
            if record.classification == OrphanClassification.PARENT_CORRUPTED:
                orphan_risk = orphan.entropy * (1 - orphan.stability)
                if orphan_risk > 0.5:
                    return False
        
        # Execute adoption
        orphan.parent_agent_id = adopter_id
        adopter.children.append(orphan_id)
        
        if orphan_id in self.orphans:
            self.orphans[orphan_id].adopted_by = adopter_id
        if orphan_id in self.quarantine:
            self.quarantine.remove(orphan_id)
        
        return True


# =============================================================================
# HIERARCHICAL ZK-SP (Zero-Knowledge Symbolic Proof)
# =============================================================================

@dataclass
class AgentProof:
    """Proof of agent integrity"""
    agent_id: str
    capsule_count: int
    root_hash: str
    stability_commitment: str
    entropy_bound: str
    timestamp: float

@dataclass
class ClusterProof:
    """Aggregated proof for cluster"""
    cluster_id: str
    agent_proofs: List[AgentProof]
    merkle_root: str
    consensus_hash: str
    agent_count: int
    timestamp: float

@dataclass
class GlobalProof:
    """Global audit proof for entire swarm"""
    proof_id: str
    cluster_proofs: List[ClusterProof]
    global_merkle_root: str
    total_agents: int
    total_capsules: int
    swarm_spcm: float
    timestamp: float
    verification_key_hash: str

class HierarchicalZKSP:
    """
    Recursive ZK proof aggregation for swarm audit.
    
    Structure:
    - Agent Proofs: Individual agent integrity
    - Cluster Proofs: Aggregate agent proofs in cluster
    - Global Proof: Aggregate all cluster proofs
    
    Enables: "Audit 100,000 agents with ONE cryptographic check"
    """
    
    def __init__(self):
        self.verification_key = hashlib.sha256(b"efm-zksp-vk").hexdigest()
    
    def generate_agent_proof(self, agent: Agent, 
                             capsules: List[SwarmCapsule]) -> AgentProof:
        """Generate proof for single agent"""
        # Compute Merkle root of capsule fingerprints
        fingerprints = [c.fingerprint() for c in capsules]
        root_hash = self._merkle_root(fingerprints) if fingerprints else "empty"
        
        # Commitments (in real ZK, these would be Pedersen commitments)
        stability_commitment = hashlib.sha256(
            f"stability:{agent.stability}".encode()
        ).hexdigest()[:16]
        
        entropy_bound = hashlib.sha256(
            f"entropy_bound:{agent.entropy}".encode()
        ).hexdigest()[:16]
        
        return AgentProof(
            agent_id=agent.agent_id,
            capsule_count=len(capsules),
            root_hash=root_hash,
            stability_commitment=stability_commitment,
            entropy_bound=entropy_bound,
            timestamp=time.time()
        )
    
    def generate_cluster_proof(self, cluster_id: str, 
                                agent_proofs: List[AgentProof]) -> ClusterProof:
        """Aggregate agent proofs into cluster proof"""
        # Merkle root of agent proof hashes
        proof_hashes = [
            hashlib.sha256(
                f"{p.agent_id}:{p.root_hash}:{p.stability_commitment}".encode()
            ).hexdigest()
            for p in agent_proofs
        ]
        merkle_root = self._merkle_root(proof_hashes)
        
        # Consensus hash (simplified - would be BFT signature)
        consensus_hash = hashlib.sha256(
            f"{cluster_id}:{merkle_root}:{len(agent_proofs)}".encode()
        ).hexdigest()[:16]
        
        return ClusterProof(
            cluster_id=cluster_id,
            agent_proofs=agent_proofs,
            merkle_root=merkle_root,
            consensus_hash=consensus_hash,
            agent_count=len(agent_proofs),
            timestamp=time.time()
        )
    
    def generate_global_proof(self, cluster_proofs: List[ClusterProof],
                               swarm_spcm: float) -> GlobalProof:
        """Generate global audit proof from cluster proofs"""
        # Global Merkle root
        cluster_hashes = [
            hashlib.sha256(
                f"{p.cluster_id}:{p.merkle_root}:{p.consensus_hash}".encode()
            ).hexdigest()
            for p in cluster_proofs
        ]
        global_root = self._merkle_root(cluster_hashes)
        
        # Totals
        total_agents = sum(p.agent_count for p in cluster_proofs)
        total_capsules = sum(
            sum(ap.capsule_count for ap in p.agent_proofs)
            for p in cluster_proofs
        )
        
        proof_id = hashlib.sha256(
            f"global:{global_root}:{time.time()}".encode()
        ).hexdigest()[:16]
        
        return GlobalProof(
            proof_id=proof_id,
            cluster_proofs=cluster_proofs,
            global_merkle_root=global_root,
            total_agents=total_agents,
            total_capsules=total_capsules,
            swarm_spcm=swarm_spcm,
            timestamp=time.time(),
            verification_key_hash=self.verification_key[:16]
        )
    
    def verify_global_proof(self, proof: GlobalProof) -> bool:
        """
        Verify global proof in O(log n) time.
        
        In production: This would use recursive SNARK verification.
        Here: We verify the Merkle structure.
        """
        # Recompute global Merkle root
        cluster_hashes = [
            hashlib.sha256(
                f"{p.cluster_id}:{p.merkle_root}:{p.consensus_hash}".encode()
            ).hexdigest()
            for p in proof.cluster_proofs
        ]
        computed_root = self._merkle_root(cluster_hashes)
        
        return computed_root == proof.global_merkle_root
    
    def _merkle_root(self, items: List[str]) -> str:
        """Compute Merkle root of items"""
        if not items:
            return hashlib.sha256(b"empty").hexdigest()[:16]
        
        if len(items) == 1:
            return hashlib.sha256(items[0].encode()).hexdigest()[:16]
        
        # Pad to power of 2
        while len(items) & (len(items) - 1):
            items.append(items[-1])
        
        # Build tree bottom-up
        while len(items) > 1:
            items = [
                hashlib.sha256(
                    f"{items[i]}:{items[i+1]}".encode()
                ).hexdigest()[:16]
                for i in range(0, len(items), 2)
            ]
        
        return items[0]


# =============================================================================
# SWARM COGNITIVE CONTROLLER
# =============================================================================

class SwarmCAC:
    """
    Distributed Cognitive Aperture Controller.
    Coordinates trace levels across swarm with local autonomy.
    """
    
    def __init__(self, alpha_min: float = 0.1, alpha_max: float = 0.95):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.local_levels: Dict[str, int] = {}
        self.local_alphas: Dict[str, float] = {}
        
    def update_local(self, node_id: str, local_spcm: float, 
                     local_ttf: float) -> Tuple[int, float]:
        """Update local CAC for node"""
        # Velocity-based damping (from B3)
        if local_ttf > 0:
            alpha_dyn = self.alpha_max - (self.alpha_max - self.alpha_min) * \
                        (1 - np.exp(-10 / local_ttf))
        else:
            alpha_dyn = self.alpha_max
        
        # Level determination
        if local_spcm > 0.8:
            level = 4
        elif local_spcm > 0.5:
            level = 3
        elif local_spcm > 0.2:
            level = 2
        else:
            level = 1
        
        self.local_levels[node_id] = level
        self.local_alphas[node_id] = alpha_dyn
        
        return level, alpha_dyn
    
    def compute_global_level(self) -> int:
        """Compute global trace level from local levels"""
        if not self.local_levels:
            return 2
        return max(self.local_levels.values())
    
    def should_escalate(self, node_id: str, swarm_spcm: float) -> bool:
        """Determine if node should escalate based on swarm state"""
        local_level = self.local_levels.get(node_id, 2)
        
        # Escalate if swarm SPCM high but local level low
        if swarm_spcm > 0.5 and local_level < 3:
            return True
        
        return False


# =============================================================================
# INTEGRATED SWARM COORDINATOR
# =============================================================================

class SwarmCoordinator:
    """
    Main coordinator for distributed EFM swarm.
    Integrates d-CTM, IA-BIM, OrphanProtocol, ZK-SP, and SwarmCAC.
    """
    
    def __init__(self):
        self.d_ctm = DecentralizedCTM()
        self.ia_bim = InterAgentBIM()
        self.orphan_protocol = OrphanProtocol(self.d_ctm)
        self.zksp = HierarchicalZKSP()
        self.swarm_cac = SwarmCAC()
        
        self.all_agents: Dict[str, Agent] = {}
        self.clusters: Dict[str, List[str]] = defaultdict(list)
        self.tick_count: int = 0
        
    def register_cluster(self, cluster_id: str) -> LocalCTMNode:
        """Register new cluster with local CTM node"""
        node = LocalCTMNode(f"node_{cluster_id}", cluster_id)
        self.d_ctm.add_node(node)
        return node
    
    def register_agent(self, agent: Agent, node: LocalCTMNode):
        """Register agent with cluster"""
        self.all_agents[agent.agent_id] = agent
        self.clusters[agent.cluster_id].append(agent.agent_id)
        node.register_agent(agent)
    
    def tick(self) -> Dict:
        """
        Single swarm tick - the heartbeat of distributed cognition.
        
        Phases:
        1. Each agent generates capsule
        2. Local CTM stores with local sequence
        3. IA-BIM computes swarm coherence
        4. Orphan detection
        5. Swarm CAC update
        6. ZK-SP proof generation (periodic)
        """
        self.tick_count += 1
        results = {
            "tick": self.tick_count,
            "agents": len(self.all_agents),
            "clusters": len(self.clusters),
            "events": []
        }
        
        # Phase 1-2: Generate and store capsules
        all_capsules = []
        for node_id, node in self.d_ctm.nodes.items():
            for agent_id, agent in node.agents.items():
                capsule = agent.create_capsule()
                node.store_capsule(capsule)
                all_capsules.append(capsule)
        
        results["capsules_generated"] = len(all_capsules)
        
        # Phase 3: Compute swarm coherence
        agents_list = list(self.all_agents.values())
        if len(agents_list) >= 2:
            W = self.ia_bim.compute_swarm_matrix(agents_list)
            swarm_spcm = self.ia_bim.compute_spcm_swarm(agents_list)
            broken_links = self.ia_bim.detect_consensus_loss(agents_list)
            
            results["swarm_spcm"] = swarm_spcm
            results["broken_links"] = len(broken_links)
            
            if broken_links:
                results["events"].append({
                    "type": "CONSENSUS_LOSS",
                    "links": broken_links[:5]  # First 5
                })
        else:
            swarm_spcm = 0.0
            results["swarm_spcm"] = 0.0
            results["broken_links"] = 0
        
        # Phase 4: Orphan detection
        new_orphans = self.orphan_protocol.detect_orphans(self.all_agents)
        if new_orphans:
            results["events"].append({
                "type": "ORPHANS_DETECTED",
                "count": len(new_orphans),
                "ids": [o.agent_id for o in new_orphans]
            })
        
        results["total_orphans"] = len(self.orphan_protocol.orphans)
        results["quarantined"] = len(self.orphan_protocol.quarantine)
        
        # Phase 5: Swarm CAC update
        for node_id, node in self.d_ctm.nodes.items():
            # Compute local SPCM for node's agents
            local_agents = list(node.agents.values())
            if len(local_agents) >= 2:
                local_spcm = self.ia_bim.compute_spcm_swarm(local_agents)
            else:
                local_spcm = 0.0
            
            # Simplified TTF (would use TPE in full implementation)
            local_ttf = 100 * (1 - local_spcm) if local_spcm < 1.0 else 0
            
            level, alpha = self.swarm_cac.update_local(node_id, local_spcm, local_ttf)
            
            # Check for escalation
            if self.swarm_cac.should_escalate(node_id, swarm_spcm):
                results["events"].append({
                    "type": "CAC_ESCALATION",
                    "node": node_id,
                    "new_level": level + 1
                })
        
        results["global_trace_level"] = self.swarm_cac.compute_global_level()
        
        # Phase 6: ZK-SP proof (every 10 ticks)
        if self.tick_count % 10 == 0:
            proof = self._generate_global_proof()
            if proof:
                results["events"].append({
                    "type": "GLOBAL_PROOF_GENERATED",
                    "proof_id": proof.proof_id,
                    "total_agents": proof.total_agents,
                    "total_capsules": proof.total_capsules
                })
        
        return results
    
    def _generate_global_proof(self) -> Optional[GlobalProof]:
        """Generate hierarchical ZK proof for swarm"""
        cluster_proofs = []
        
        for cluster_id, agent_ids in self.clusters.items():
            agent_proofs = []
            
            for agent_id in agent_ids:
                if agent_id not in self.all_agents:
                    continue
                
                agent = self.all_agents[agent_id]
                
                # Get capsules for agent
                capsules = []
                for node in self.d_ctm.nodes.values():
                    capsules.extend([
                        c for c in node.capsules.values()
                        if c.agent_id == agent_id
                    ])
                
                proof = self.zksp.generate_agent_proof(agent, capsules)
                agent_proofs.append(proof)
            
            if agent_proofs:
                cluster_proof = self.zksp.generate_cluster_proof(
                    cluster_id, agent_proofs
                )
                cluster_proofs.append(cluster_proof)
        
        if not cluster_proofs:
            return None
        
        # Get swarm SPCM
        agents_list = list(self.all_agents.values())
        swarm_spcm = self.ia_bim.compute_spcm_swarm(agents_list) \
                     if len(agents_list) >= 2 else 0.0
        
        return self.zksp.generate_global_proof(cluster_proofs, swarm_spcm)
    
    def inject_corruption(self, agent_id: str, severity: float = 0.5):
        """Inject corruption into agent for testing"""
        if agent_id in self.all_agents:
            agent = self.all_agents[agent_id]
            agent.stability *= (1 - severity)
            agent.entropy = min(1.0, agent.entropy + severity)
            agent.phi += np.random.randn(64) * severity
    
    def spawn_child_agent(self, parent_id: str, reason: str = "task") -> Optional[Agent]:
        """Spawn child agent from parent"""
        if parent_id not in self.all_agents:
            return None
        
        parent = self.all_agents[parent_id]
        child = parent.spawn_child(reason)
        
        # Find parent's node
        for node in self.d_ctm.nodes.values():
            if parent_id in node.agents:
                self.register_agent(child, node)
                return child
        
        return None


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_distributed_efm():
    """Demonstrate distributed EFM capabilities"""
    print("=" * 70)
    print("DISTRIBUTED EFM DEMONSTRATION")
    print("Booklet 4: Cognitive Genealogy and Distributed Swarm Autonomy")
    print("=" * 70)
    
    # Initialize coordinator
    coordinator = SwarmCoordinator()
    
    # Create 3 clusters
    clusters = {}
    for i in range(3):
        cluster_id = f"cluster_{i}"
        node = coordinator.register_cluster(cluster_id)
        clusters[cluster_id] = node
    
    # Create agents (3 per cluster, with parent-child relationships)
    print("\n[Phase 1] Creating swarm with 3 clusters, 9 agents")
    agent_count = 0
    for cluster_id, node in clusters.items():
        # Root agent
        root = Agent(
            agent_id=f"agent_{agent_count}",
            cluster_id=cluster_id,
            phi=np.random.randn(64)
        )
        coordinator.register_agent(root, node)
        agent_count += 1
        
        # Two children
        for _ in range(2):
            child = root.spawn_child("initialization")
            coordinator.register_agent(child, node)
            agent_count += 1
    
    print(f"  Created {len(coordinator.all_agents)} agents across {len(clusters)} clusters")
    
    # Run stable ticks
    print("\n[Phase 2] Running 20 stable ticks")
    for t in range(20):
        results = coordinator.tick()
    
    print(f"  Tick {results['tick']}: SPCM={results['swarm_spcm']:.3f}, "
          f"Level={results['global_trace_level']}, Orphans={results['total_orphans']}")
    
    # Inject corruption into one agent
    print("\n[Phase 3] Injecting corruption into agent_1")
    coordinator.inject_corruption("agent_1", severity=0.8)
    
    # Run more ticks to observe propagation
    print("\n[Phase 4] Running 30 ticks with corruption")
    for t in range(30):
        results = coordinator.tick()
        
        if results["events"]:
            for event in results["events"]:
                if event["type"] == "CONSENSUS_LOSS":
                    print(f"  Tick {results['tick']}: CONSENSUS LOSS detected - "
                          f"{len(event['links'])} broken links")
                elif event["type"] == "ORPHANS_DETECTED":
                    print(f"  Tick {results['tick']}: ORPHANS detected - "
                          f"{event['count']} agents: {event['ids']}")
                elif event["type"] == "GLOBAL_PROOF_GENERATED":
                    print(f"  Tick {results['tick']}: Global proof generated - "
                          f"{event['total_agents']} agents, {event['total_capsules']} capsules")
    
    print(f"\n  Final: SPCM={results['swarm_spcm']:.3f}, "
          f"Level={results['global_trace_level']}, Orphans={results['total_orphans']}")
    
    # Demonstrate adoption
    print("\n[Phase 5] Attempting orphan adoption")
    for orphan_id in list(coordinator.orphan_protocol.orphans.keys()):
        # Try to find healthy adopter
        for agent_id, agent in coordinator.all_agents.items():
            if agent_id != orphan_id:
                success = coordinator.orphan_protocol.attempt_adoption(
                    orphan_id, agent_id, coordinator.all_agents
                )
                if success:
                    print(f"  Agent {orphan_id} adopted by {agent_id}")
                    break
    
    # Final proof
    print("\n[Phase 6] Generating final global audit proof")
    proof = coordinator._generate_global_proof()
    if proof:
        print(f"  Proof ID: {proof.proof_id}")
        print(f"  Total Agents: {proof.total_agents}")
        print(f"  Total Capsules: {proof.total_capsules}")
        print(f"  Swarm SPCM: {proof.swarm_spcm:.3f}")
        print(f"  Verification: {'✓ VALID' if coordinator.zksp.verify_global_proof(proof) else '✗ INVALID'}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return coordinator


if __name__ == "__main__":
    demonstrate_distributed_efm()
