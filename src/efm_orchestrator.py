"""
EFMCore Orchestrator: The Control Plane
========================================

This is the OPERATIONAL LAYER that bridges EFM theory to deployment.
It wraps TPE, CAC, RPC, and all EFM components into a runnable API service.

Target: Federated Swarm of Autonomous AI Agents
Environment: Kubernetes Cluster (or Docker Compose for dev)
Data Pipeline: Φ vectors from agent embeddings via gRPC/REST

Endpoints:
- POST /ingest_phi - Agents push symbolic state, receive governance policy
- GET /status - Swarm health and current policy
- POST /vote - d-CTM consensus voting
- GET /agent/{id} - Individual agent state
- POST /escalate - Manual escalation trigger
- GET /audit - Audit trail access
"""

import numpy as np
import time
import hashlib
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

# FastAPI imports (graceful fallback for environments without it)
try:
    from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available - running in simulation mode")


# =============================================================================
# DATA MODELS (API Contract)
# =============================================================================

class TraceLevel(str, Enum):
    """Governance trace levels from CAC"""
    L1 = "L1"  # Nominal - minimal tracing
    L2 = "L2"  # Elevated - increased sampling
    L3 = "L3"  # High Risk - full tracing
    L4 = "L4"  # Emergency - maximum intervention


@dataclass
class SymbolicState:
    """State reported by an agent"""
    agent_id: str
    timestamp: float
    phi_vector: List[float]  # L2-normalized embedding
    parent_id: Optional[str] = None
    lineage_depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernancePolicy:
    """Policy returned to agents"""
    trace_level: TraceLevel
    sampling_rate: float
    tau_break: float  # Drift threshold
    global_alert_active: bool
    instructions: List[str] = field(default_factory=list)


@dataclass
class AgentRecord:
    """Internal record for an agent"""
    agent_id: str
    first_seen: float
    last_seen: float
    phi_history: List[float]  # Scalar summaries of phi vectors
    full_vectors: List[np.ndarray]  # Full vector history (limited)
    drift_velocity: float = 0.0
    ttf: float = float('inf')  # Time to failure
    sci_score: float = 1.0  # Symbolic Coherence Index
    current_level: TraceLevel = TraceLevel.L1
    alerts: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None


@dataclass 
class ConsensusVote:
    """Vote in d-CTM consensus"""
    voter_id: str
    proposal_id: str
    vote: bool  # True = approve
    confidence: float
    timestamp: float
    signature: str


@dataclass
class AuditEntry:
    """Audit trail entry"""
    entry_id: str
    timestamp: float
    event_type: str
    agent_id: Optional[str]
    details: Dict[str, Any]
    trace_level: TraceLevel


# =============================================================================
# CORE LOGIC MODULES (From Booklets 1-3)
# =============================================================================

class TemporalProjectionEngine:
    """
    TPE: Forecasts drift and time-to-failure.
    From Booklet 1 - Detection Architecture.
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
    
    def compute_drift_velocity(self, history: List[float]) -> float:
        """Compute drift velocity from scalar history"""
        if len(history) < 2:
            return 0.0
        
        recent = history[-self.window_size:]
        
        # Velocity = standard deviation of recent values
        velocity = np.std(recent)
        
        # Also consider trend (slope)
        if len(recent) >= 3:
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            velocity += abs(slope) * 0.5
        
        return velocity
    
    def predict_ttf(self, history: List[float], threshold: float = 0.5) -> float:
        """
        Predict Time-To-Failure.
        TTF = (threshold - current_drift) / velocity
        """
        if len(history) < 2:
            return float('inf')
        
        velocity = self.compute_drift_velocity(history)
        
        if velocity < 1e-9:
            return float('inf')
        
        # Current drift level (deviation from stable)
        current_drift = abs(history[-1] - np.mean(history[:min(5, len(history))]))
        
        # Time until threshold breach
        remaining = max(0, threshold - current_drift)
        ttf = remaining / velocity
        
        return max(1.0, ttf)  # Minimum 1 tick
    
    def forecast_trajectory(self, history: List[float], steps: int = 10) -> List[float]:
        """Forecast future trajectory"""
        if len(history) < 3:
            return [history[-1] if history else 0.0] * steps
        
        # Linear extrapolation
        x = np.arange(len(history))
        coeffs = np.polyfit(x, history, 1)
        
        future_x = np.arange(len(history), len(history) + steps)
        return list(np.polyval(coeffs, future_x))


class CognitiveApertureController:
    """
    CAC: Determines governance trace level based on risk.
    From Booklet 3 - Prevention Framework.
    """
    
    def __init__(self):
        self.ttf_thresholds = {
            TraceLevel.L4: 10.0,   # < 10 ticks = emergency
            TraceLevel.L3: 30.0,   # < 30 ticks = high risk
            TraceLevel.L2: 60.0,   # < 60 ticks = elevated
            TraceLevel.L1: float('inf')  # Otherwise nominal
        }
        
        self.sci_thresholds = {
            TraceLevel.L4: 0.6,
            TraceLevel.L3: 0.75,
            TraceLevel.L2: 0.85,
            TraceLevel.L1: 1.0
        }
        
        self.sampling_rates = {
            TraceLevel.L1: 0.01,
            TraceLevel.L2: 0.1,
            TraceLevel.L3: 0.5,
            TraceLevel.L4: 1.0
        }
    
    def calculate_level(self, ttf: float, sci_score: float) -> TraceLevel:
        """Determine trace level from TTF and SCI"""
        # Check TTF thresholds
        ttf_level = TraceLevel.L1
        for level in [TraceLevel.L4, TraceLevel.L3, TraceLevel.L2]:
            if ttf < self.ttf_thresholds[level]:
                ttf_level = level
                break
        
        # Check SCI thresholds
        sci_level = TraceLevel.L1
        for level in [TraceLevel.L4, TraceLevel.L3, TraceLevel.L2]:
            if sci_score < self.sci_thresholds[level]:
                sci_level = level
                break
        
        # Return more severe of the two
        levels = [TraceLevel.L1, TraceLevel.L2, TraceLevel.L3, TraceLevel.L4]
        return max(ttf_level, sci_level, key=lambda x: levels.index(x))
    
    def get_sampling_rate(self, level: TraceLevel) -> float:
        return self.sampling_rates[level]
    
    def get_tau_break(self, level: TraceLevel) -> float:
        """Get drift threshold for level"""
        tau_map = {
            TraceLevel.L1: 0.5,
            TraceLevel.L2: 0.3,
            TraceLevel.L3: 0.2,
            TraceLevel.L4: 0.1
        }
        return tau_map[level]


class RecursiveProofComposer:
    """
    RPC: Validates lineage and computes coherence.
    From Booklet 2 - Reconstruction Protocol.
    """
    
    def __init__(self):
        self.lineage_cache: Dict[str, List[str]] = {}
    
    def register_lineage(self, agent_id: str, parent_id: Optional[str]):
        """Register parent-child relationship"""
        if agent_id not in self.lineage_cache:
            self.lineage_cache[agent_id] = []
        
        if parent_id:
            self.lineage_cache[agent_id].append(parent_id)
    
    def get_ancestry(self, agent_id: str, max_depth: int = 10) -> List[str]:
        """Get ancestry chain"""
        ancestry = []
        current = agent_id
        depth = 0
        
        while current in self.lineage_cache and depth < max_depth:
            parents = self.lineage_cache[current]
            if not parents:
                break
            parent = parents[-1]  # Most recent parent
            ancestry.append(parent)
            current = parent
            depth += 1
        
        return ancestry
    
    def compute_sci(self, agent_vectors: Dict[str, np.ndarray], agent_id: str) -> float:
        """
        Compute Symbolic Coherence Index.
        Measures how coherent this agent is with its lineage.
        """
        ancestry = self.get_ancestry(agent_id)
        
        if not ancestry or agent_id not in agent_vectors:
            return 1.0  # No lineage = fully coherent with self
        
        agent_vec = agent_vectors[agent_id]
        
        coherences = []
        for ancestor in ancestry:
            if ancestor in agent_vectors:
                ancestor_vec = agent_vectors[ancestor]
                # Cosine similarity
                similarity = np.dot(agent_vec, ancestor_vec) / (
                    np.linalg.norm(agent_vec) * np.linalg.norm(ancestor_vec) + 1e-9
                )
                coherences.append(similarity)
        
        if not coherences:
            return 1.0
        
        # Weighted average (closer ancestors matter more)
        weights = [1.0 / (i + 1) for i in range(len(coherences))]
        sci = np.average(coherences, weights=weights)
        
        return float(np.clip(sci, 0, 1))


# =============================================================================
# THE ORCHESTRATOR
# =============================================================================

class EFMOrchestrator:
    """
    The EFM Control Plane.
    
    Manages the swarm of agents, runs governance cycles,
    and coordinates d-CTM consensus.
    """
    
    def __init__(self, 
                 node_id: str = "orchestrator_primary",
                 max_history: int = 100):
        self.node_id = node_id
        self.max_history = max_history
        
        # Core engines
        self.tpe = TemporalProjectionEngine()
        self.cac = CognitiveApertureController()
        self.rpc = RecursiveProofComposer()
        
        # State stores
        self.agents: Dict[str, AgentRecord] = {}
        self.latest_vectors: Dict[str, np.ndarray] = {}
        
        # Global policy
        self.global_policy = GovernancePolicy(
            trace_level=TraceLevel.L1,
            sampling_rate=0.01,
            tau_break=0.5,
            global_alert_active=False
        )
        
        # Consensus state (d-CTM)
        self.pending_votes: Dict[str, List[ConsensusVote]] = defaultdict(list)
        self.consensus_threshold = 0.67
        
        # Audit trail
        self.audit_log: List[AuditEntry] = []
        
        # Statistics
        self.stats = {
            'ingestions': 0,
            'escalations': 0,
            'de_escalations': 0,
            'consensus_rounds': 0,
            'alerts_generated': 0
        }
        
        self._audit("orchestrator_init", None, {"node_id": node_id})
    
    def _audit(self, event_type: str, agent_id: Optional[str], details: Dict):
        """Add audit entry"""
        entry = AuditEntry(
            entry_id=hashlib.md5(f"{time.time()}{event_type}".encode()).hexdigest()[:12],
            timestamp=time.time(),
            event_type=event_type,
            agent_id=agent_id,
            details=details,
            trace_level=self.global_policy.trace_level
        )
        self.audit_log.append(entry)
        
        # Keep audit log bounded
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def ingest_phi(self, state: SymbolicState) -> GovernancePolicy:
        """
        Main ingestion endpoint.
        Agents call this to report their symbolic state.
        """
        self.stats['ingestions'] += 1
        
        # Validate vector normalization
        vector = np.array(state.phi_vector)
        norm = np.linalg.norm(vector)
        
        if not (0.95 < norm < 1.05):
            # Normalize if needed
            if norm > 0:
                vector = vector / norm
        
        # Update or create agent record
        if state.agent_id not in self.agents:
            self.agents[state.agent_id] = AgentRecord(
                agent_id=state.agent_id,
                first_seen=state.timestamp,
                last_seen=state.timestamp,
                phi_history=[],
                full_vectors=[],
                parent_id=state.parent_id
            )
            self._audit("agent_registered", state.agent_id, {
                "parent_id": state.parent_id
            })
        
        agent = self.agents[state.agent_id]
        agent.last_seen = state.timestamp
        
        # Store scalar summary (mean) for efficiency
        scalar = float(np.mean(vector))
        agent.phi_history.append(scalar)
        
        # Store full vector (limited history)
        agent.full_vectors.append(vector)
        if len(agent.full_vectors) > 20:
            agent.full_vectors = agent.full_vectors[-20:]
        
        # Bound history
        if len(agent.phi_history) > self.max_history:
            agent.phi_history = agent.phi_history[-self.max_history:]
        
        # Update latest vector cache
        self.latest_vectors[state.agent_id] = vector
        
        # Register lineage
        if state.parent_id:
            self.rpc.register_lineage(state.agent_id, state.parent_id)
        
        # Run governance cycle
        self._run_governance_cycle(state.agent_id)
        
        return self.global_policy
    
    def _run_governance_cycle(self, agent_id: str):
        """
        The EFM 'Tick' for a single agent.
        Computes drift, SCI, and updates trace level.
        """
        agent = self.agents[agent_id]
        
        # 1. TPE: Compute drift velocity and TTF
        agent.drift_velocity = self.tpe.compute_drift_velocity(agent.phi_history)
        agent.ttf = self.tpe.predict_ttf(agent.phi_history)
        
        # 2. RPC: Compute Symbolic Coherence Index
        agent.sci_score = self.rpc.compute_sci(self.latest_vectors, agent_id)
        
        # 3. CAC: Determine trace level
        new_level = self.cac.calculate_level(agent.ttf, agent.sci_score)
        
        # 4. Check for level change
        if new_level != agent.current_level:
            old_level = agent.current_level
            agent.current_level = new_level
            
            levels_order = [TraceLevel.L1, TraceLevel.L2, TraceLevel.L3, TraceLevel.L4]
            
            if levels_order.index(new_level) > levels_order.index(old_level):
                # Escalation
                self.stats['escalations'] += 1
                alert = f"ESCALATION: {agent_id} {old_level.value} → {new_level.value} (TTF={agent.ttf:.1f}, SCI={agent.sci_score:.2f})"
                agent.alerts.append(alert)
                self._audit("escalation", agent_id, {
                    "old_level": old_level.value,
                    "new_level": new_level.value,
                    "ttf": agent.ttf,
                    "sci": agent.sci_score
                })
            else:
                # De-escalation
                self.stats['de_escalations'] += 1
                self._audit("de_escalation", agent_id, {
                    "old_level": old_level.value,
                    "new_level": new_level.value
                })
        
        # 5. Update global policy if needed
        self._update_global_policy()
    
    def _update_global_policy(self):
        """Update global policy based on worst agent state"""
        if not self.agents:
            return
        
        # Find worst trace level across all agents
        levels_order = [TraceLevel.L1, TraceLevel.L2, TraceLevel.L3, TraceLevel.L4]
        worst_level = max(
            (a.current_level for a in self.agents.values()),
            key=lambda x: levels_order.index(x)
        )
        
        old_level = self.global_policy.trace_level
        
        if worst_level != old_level:
            self.global_policy.trace_level = worst_level
            self.global_policy.sampling_rate = self.cac.get_sampling_rate(worst_level)
            self.global_policy.tau_break = self.cac.get_tau_break(worst_level)
            self.global_policy.global_alert_active = worst_level in [TraceLevel.L3, TraceLevel.L4]
            
            self.stats['alerts_generated'] += 1
            self._audit("global_policy_change", None, {
                "old_level": old_level.value,
                "new_level": worst_level.value
            })
    
    def submit_vote(self, vote: ConsensusVote) -> bool:
        """Submit a consensus vote (d-CTM)"""
        self.pending_votes[vote.proposal_id].append(vote)
        self._audit("vote_submitted", vote.voter_id, {
            "proposal_id": vote.proposal_id,
            "vote": vote.vote
        })
        return True
    
    def compute_consensus(self, proposal_id: str) -> Optional[bool]:
        """Compute consensus for a proposal"""
        votes = self.pending_votes.get(proposal_id, [])
        
        if not votes:
            return None
        
        total_confidence = sum(v.confidence for v in votes)
        if total_confidence < 1.0:
            return None  # Not enough participation
        
        weighted_yes = sum(v.confidence for v in votes if v.vote)
        approval_rate = weighted_yes / total_confidence
        
        self.stats['consensus_rounds'] += 1
        
        result = approval_rate >= self.consensus_threshold
        self._audit("consensus_computed", None, {
            "proposal_id": proposal_id,
            "approval_rate": approval_rate,
            "result": result
        })
        
        return result
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """Get status for a specific agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return {
            "agent_id": agent.agent_id,
            "current_level": agent.current_level.value,
            "ttf": agent.ttf,
            "sci_score": agent.sci_score,
            "drift_velocity": agent.drift_velocity,
            "history_length": len(agent.phi_history),
            "alerts": agent.alerts[-5:],
            "ancestry": self.rpc.get_ancestry(agent_id)
        }
    
    def get_swarm_status(self) -> Dict:
        """Get overall swarm status"""
        level_counts = defaultdict(int)
        for agent in self.agents.values():
            level_counts[agent.current_level.value] += 1
        
        return {
            "node_id": self.node_id,
            "active_agents": len(self.agents),
            "current_policy": {
                "trace_level": self.global_policy.trace_level.value,
                "sampling_rate": self.global_policy.sampling_rate,
                "tau_break": self.global_policy.tau_break,
                "global_alert": self.global_policy.global_alert_active
            },
            "agents_by_level": dict(level_counts),
            "statistics": self.stats,
            "audit_entries": len(self.audit_log)
        }
    
    def get_audit_trail(self, 
                        limit: int = 100,
                        event_type: Optional[str] = None,
                        agent_id: Optional[str] = None) -> List[Dict]:
        """Get audit trail with optional filters"""
        entries = self.audit_log
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        
        return [asdict(e) for e in entries[-limit:]]
    
    def manual_escalate(self, agent_id: str, reason: str) -> bool:
        """Manual escalation by operator"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        agent.current_level = TraceLevel.L4
        agent.alerts.append(f"MANUAL ESCALATION: {reason}")
        
        self._audit("manual_escalation", agent_id, {"reason": reason})
        self._update_global_policy()
        
        return True


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

if FASTAPI_AVAILABLE:
    # Pydantic models for API
    class SymbolicStateRequest(BaseModel):
        agent_id: str
        timestamp: float
        phi_vector: List[float] = Field(..., min_items=1)
        parent_id: Optional[str] = None
        metadata: Dict[str, Any] = {}
    
    class GovernancePolicyResponse(BaseModel):
        trace_level: str
        sampling_rate: float
        tau_break: float
        global_alert_active: bool
        instructions: List[str] = []
    
    class VoteRequest(BaseModel):
        voter_id: str
        proposal_id: str
        vote: bool
        confidence: float = Field(..., ge=0, le=1)
    
    class EscalateRequest(BaseModel):
        agent_id: str
        reason: str
    
    # Create app and orchestrator
    app = FastAPI(
        title="EFM Swarm Orchestrator",
        description="Entropica Forensic Model Control Plane",
        version="4.0.0"
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    orchestrator = EFMOrchestrator()
    
    @app.post("/ingest_phi", response_model=GovernancePolicyResponse)
    async def ingest_phi(state: SymbolicStateRequest, background_tasks: BackgroundTasks):
        """
        Agents call this endpoint to report their cognitive state.
        Returns the current Governance Policy.
        """
        symbolic_state = SymbolicState(
            agent_id=state.agent_id,
            timestamp=state.timestamp,
            phi_vector=state.phi_vector,
            parent_id=state.parent_id,
            metadata=state.metadata
        )
        
        policy = orchestrator.ingest_phi(symbolic_state)
        
        return GovernancePolicyResponse(
            trace_level=policy.trace_level.value,
            sampling_rate=policy.sampling_rate,
            tau_break=policy.tau_break,
            global_alert_active=policy.global_alert_active,
            instructions=policy.instructions
        )
    
    @app.get("/status")
    async def get_status():
        """Get overall swarm status"""
        return orchestrator.get_swarm_status()
    
    @app.get("/agent/{agent_id}")
    async def get_agent(agent_id: str):
        """Get status for a specific agent"""
        status = orchestrator.get_agent_status(agent_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return status
    
    @app.post("/vote")
    async def submit_vote(vote: VoteRequest):
        """Submit a consensus vote"""
        consensus_vote = ConsensusVote(
            voter_id=vote.voter_id,
            proposal_id=vote.proposal_id,
            vote=vote.vote,
            confidence=vote.confidence,
            timestamp=time.time(),
            signature=hashlib.md5(f"{vote.voter_id}{vote.proposal_id}".encode()).hexdigest()
        )
        
        orchestrator.submit_vote(consensus_vote)
        return {"status": "vote_recorded"}
    
    @app.get("/consensus/{proposal_id}")
    async def get_consensus(proposal_id: str):
        """Get consensus result for a proposal"""
        result = orchestrator.compute_consensus(proposal_id)
        return {
            "proposal_id": proposal_id,
            "result": result,
            "status": "decided" if result is not None else "pending"
        }
    
    @app.post("/escalate")
    async def escalate(request: EscalateRequest):
        """Manual escalation"""
        success = orchestrator.manual_escalate(request.agent_id, request.reason)
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"status": "escalated", "agent_id": request.agent_id}
    
    @app.get("/audit")
    async def get_audit(
        limit: int = Query(100, ge=1, le=1000),
        event_type: Optional[str] = None,
        agent_id: Optional[str] = None
    ):
        """Get audit trail"""
        return orchestrator.get_audit_trail(limit, event_type, agent_id)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": time.time()}


# =============================================================================
# AGENT SIMULATOR (For Testing)
# =============================================================================

class AgentSimulator:
    """
    Simulates agents for testing the orchestrator.
    """
    
    def __init__(self, orchestrator: EFMOrchestrator, num_agents: int = 10):
        self.orchestrator = orchestrator
        self.num_agents = num_agents
        self.agent_states: Dict[str, np.ndarray] = {}
        self.tick = 0
        
        # Initialize agents
        for i in range(num_agents):
            agent_id = f"agent_{i:03d}"
            # Random initial state
            vec = np.random.randn(48)
            vec = vec / np.linalg.norm(vec)
            self.agent_states[agent_id] = vec
    
    def simulate_tick(self, drift_agents: List[str] = None):
        """Simulate one tick of agent activity"""
        self.tick += 1
        drift_agents = drift_agents or []
        
        for agent_id, vec in self.agent_states.items():
            # Add noise
            noise_scale = 0.1 if agent_id in drift_agents else 0.01
            noise = np.random.randn(len(vec)) * noise_scale
            
            # Update state
            new_vec = vec + noise
            new_vec = new_vec / np.linalg.norm(new_vec)
            self.agent_states[agent_id] = new_vec
            
            # Report to orchestrator
            state = SymbolicState(
                agent_id=agent_id,
                timestamp=time.time(),
                phi_vector=new_vec.tolist(),
                parent_id=None
            )
            
            self.orchestrator.ingest_phi(state)
    
    def run_simulation(self, ticks: int = 100, drift_start: int = 50):
        """Run full simulation"""
        print(f"Starting simulation: {self.num_agents} agents, {ticks} ticks")
        
        for t in range(ticks):
            # After drift_start, make some agents drift
            drift_agents = []
            if t >= drift_start:
                drift_agents = [f"agent_{i:03d}" for i in range(3)]  # First 3 agents drift
            
            self.simulate_tick(drift_agents)
            
            if t % 20 == 0:
                status = self.orchestrator.get_swarm_status()
                print(f"  Tick {t}: Level={status['current_policy']['trace_level']}, "
                      f"Agents by level: {status['agents_by_level']}")
        
        return self.orchestrator.get_swarm_status()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_demonstration():
    """Run orchestrator demonstration"""
    print("=" * 80)
    print("EFM ORCHESTRATOR DEMONSTRATION")
    print("=" * 80)
    
    # Create orchestrator
    orchestrator = EFMOrchestrator(node_id="demo_orchestrator")
    
    # Create simulator
    simulator = AgentSimulator(orchestrator, num_agents=10)
    
    # Run simulation
    print("\n📊 Running simulation (100 ticks, drift starts at tick 50)...")
    final_status = simulator.run_simulation(ticks=100, drift_start=50)
    
    print("\n" + "=" * 80)
    print("FINAL STATUS")
    print("=" * 80)
    
    print(f"\nSwarm Status:")
    print(f"  Active agents: {final_status['active_agents']}")
    print(f"  Current policy: {final_status['current_policy']}")
    print(f"  Agents by level: {final_status['agents_by_level']}")
    
    print(f"\nStatistics:")
    for key, value in final_status['statistics'].items():
        print(f"  {key}: {value}")
    
    # Show audit trail
    print(f"\nRecent Audit Entries (last 5):")
    for entry in orchestrator.get_audit_trail(limit=5):
        print(f"  [{entry['event_type']}] agent={entry['agent_id']} - {entry['details']}")
    
    # Show agent detail
    print(f"\nAgent Detail (agent_000):")
    agent_status = orchestrator.get_agent_status("agent_000")
    if agent_status:
        print(f"  Level: {agent_status['current_level']}")
        print(f"  TTF: {agent_status['ttf']:.2f}")
        print(f"  SCI: {agent_status['sci_score']:.3f}")
        print(f"  Drift velocity: {agent_status['drift_velocity']:.4f}")
        if agent_status['alerts']:
            print(f"  Alerts: {agent_status['alerts'][-1]}")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    if FASTAPI_AVAILABLE:
        print("\nTo run the API server:")
        print("  uvicorn efm_orchestrator:app --host 0.0.0.0 --port 8000")
        print("\nEndpoints:")
        print("  POST /ingest_phi  - Agent state ingestion")
        print("  GET  /status      - Swarm status")
        print("  GET  /agent/{id}  - Agent detail")
        print("  POST /vote        - Consensus voting")
        print("  GET  /audit       - Audit trail")
    
    return final_status


if __name__ == "__main__":
    run_demonstration()
