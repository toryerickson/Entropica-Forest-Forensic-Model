"""
Biosphere Command Interface (BCI) Implementation
=================================================

Integrates Appendix G concepts with EFMCore:
- Semantic Zoom (Canopy → Trunk → Branch → Leaf)
- Cognitive Sonar (5D query with echo coefficients)
- Hereditary Lighting (Lineage traversal)
- Swarm Weather (Environmental Volatility Index)
- Genesis Event Console (Deployment simulation)

This module provides the backend API and data structures
for the BCI visualization layer (Unity/Three.js frontend).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from datetime import datetime
import json


# =============================================================================
# SEMANTIC ZOOM LEVELS
# =============================================================================

class ZoomLevel(Enum):
    """The four semantic zoom levels from Appendix G.2"""
    CANOPY = "canopy"   # 1M agents - Global FDR/SCI fields
    TRUNK = "trunk"     # 10K clusters - Lineage trees
    BRANCH = "branch"   # 100 agents - Consensus/dissent zones
    LEAF = "leaf"       # 1 agent - Full forensic detail


@dataclass
class CanopyView:
    """Global swarm view (1M agents aggregated)"""
    total_agents: int
    active_agents: int
    quarantined_agents: int
    global_sci: float              # Swarm-wide Symbolic Coherence Index
    global_fdr: float              # Findings Discovery Rate
    entropy_budget_remaining: float
    drift_hotspots: List[Dict]     # Regions with high drift
    growth_zones: List[Dict]       # Regions with active expansion
    weather_condition: str         # "CLEAR", "DRIFT_STORM", "FOG", "DROUGHT"
    evi: float                     # Environmental Volatility Index


@dataclass
class TrunkView:
    """Cluster view (10K agents in lineage trees)"""
    cluster_id: str
    agent_count: int
    lineage_depth: int
    root_agent_id: str
    branch_health: Dict[str, float]  # branch_id → health score
    heuristic_signatures: List[str]  # Active CSL heuristics
    graft_candidates: List[str]      # Branches eligible for grafting
    prune_candidates: List[str]      # Branches marked for pruning


@dataclass
class BranchView:
    """Local consensus view (100 agents)"""
    branch_id: str
    agent_ids: List[str]
    consensus_ratio: float          # Agreement level
    dissent_agents: List[str]       # Agents in disagreement
    drift_velocity: float           # Average drift in branch
    active_proposals: List[Dict]    # Pending governance decisions
    stability_distribution: Dict[str, float]  # agent_id → stability


@dataclass
class LeafView:
    """Full forensic view (1 agent)"""
    agent_id: str
    phi_vector: np.ndarray
    stability: float
    entropy: float
    ttf: float
    trace_level: int
    status: str
    parent_id: Optional[str]
    children_ids: List[str]
    lkc_log: List[Dict]            # Last Known Configuration log
    event_history: List[Dict]       # Full event timeline


# =============================================================================
# COGNITIVE SONAR (G.3)
# =============================================================================

@dataclass
class SonarEcho:
    """Single echo point from cognitive sonar ping"""
    agent_id: str
    coordinates: np.ndarray        # 3D projection of 5D position
    intensity: float               # Echo strength (similarity score)
    temperature: float             # Local entropy density (S_ψ)
    distance: float                # Semantic distance from query


@dataclass
class SonarResult:
    """Result of a cognitive sonar ping"""
    query_vector: np.ndarray
    threshold: float
    echoes: List[SonarEcho]
    total_scanned: int
    response_time_ms: float


class CognitiveSonar:
    """
    Active query system for navigating 5D Knowledge Matrices.
    
    From Appendix G.3:
    - Input: Semantic query vector
    - Output: Echo points with coordinates, intensity, temperature
    - Visualization: UMAP projection with thermal overlays
    """
    
    def __init__(self, swarm_state: Dict[str, np.ndarray]):
        self.swarm_state = swarm_state  # agent_id → phi_vector
        
    def ping(self, query_vector: np.ndarray, threshold: float = 0.8) -> SonarResult:
        """
        Ping the swarm with a semantic query.
        
        Echo coefficient formula from whiteboard:
        ε_co = {sim(φ_t, g) | ≥ τ}
        
        Returns all agents where cosine similarity ≥ threshold.
        """
        import time
        start = time.time()
        
        echoes = []
        query_norm = query_vector / np.linalg.norm(query_vector)
        
        for agent_id, phi in self.swarm_state.items():
            phi_norm = phi / np.linalg.norm(phi)
            
            # Cosine similarity
            similarity = float(np.dot(query_norm, phi_norm))
            
            if similarity >= threshold:
                # Calculate temperature (entropy density)
                # ψ_i = Σ σ_i e^(-Δ_t IT_i) / T
                temperature = self._calculate_temperature(phi)
                
                # Project to 3D for visualization
                coords = self._project_to_3d(phi)
                
                echoes.append(SonarEcho(
                    agent_id=agent_id,
                    coordinates=coords,
                    intensity=similarity,
                    temperature=temperature,
                    distance=1.0 - similarity
                ))
        
        # Sort by intensity (highest first)
        echoes.sort(key=lambda e: e.intensity, reverse=True)
        
        elapsed_ms = (time.time() - start) * 1000
        
        return SonarResult(
            query_vector=query_vector,
            threshold=threshold,
            echoes=echoes,
            total_scanned=len(self.swarm_state),
            response_time_ms=elapsed_ms
        )
    
    def _calculate_temperature(self, phi: np.ndarray) -> float:
        """
        Cell temperature formula from G.3:
        ψ_i = Σ σ_i e^(-Δ_t IT_i) / T
        
        Simplified: Use entropy of phi distribution
        """
        # Normalize to probability distribution
        p = np.abs(phi) / np.sum(np.abs(phi))
        p = p[p > 0]  # Remove zeros for log
        entropy = -np.sum(p * np.log2(p))
        
        # Normalize to 0-1 range
        max_entropy = np.log2(len(phi))
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _project_to_3d(self, phi: np.ndarray) -> np.ndarray:
        """
        Project high-dimensional phi to 3D for visualization.
        
        In production: Use UMAP or t-SNE
        Here: Simple PCA-style projection
        """
        # Take first 3 principal components as approximation
        if len(phi) >= 3:
            return phi[:3] / np.linalg.norm(phi[:3])
        return np.array([phi[0], 0, 0])


# =============================================================================
# HEREDITARY LIGHTING (G.4)
# =============================================================================

class TraceDirection(Enum):
    UPSTREAM = "upstream"      # Trace ancestors
    DOWNSTREAM = "downstream"  # Trace descendants
    LATERAL = "lateral"        # Trace siblings


@dataclass
class LitAgent:
    """Agent illuminated by hereditary lighting"""
    agent_id: str
    luminosity: float          # Brightness (scales with proximity)
    trace_distance: int        # Steps from origin
    relationship: str          # "ancestor", "descendant", "sibling"


class HereditaryLighting:
    """
    Lineage traversal engine from Appendix G.4.
    
    Illuminates the DAG of symbolic ancestry to:
    - Diagnose root causes (upstream)
    - Track mutation propagation (downstream)
    - Map contagion risks (lateral)
    """
    
    def __init__(self, lineage_graph: Dict[str, Dict]):
        """
        lineage_graph format:
        {
            "agent_id": {
                "parent_id": "parent_agent_id" or None,
                "children_ids": ["child1", "child2"],
                "stability": 0.85,
                "heuristics": ["h1", "h2"]
            }
        }
        """
        self.graph = lineage_graph
    
    def trace(self, origin_id: str, direction: TraceDirection, 
              max_depth: int = 10) -> List[LitAgent]:
        """
        Trace lineage in specified direction.
        
        Luminosity formula:
        L = 1.0 / (1 + distance)  # Inverse distance scaling
        """
        if origin_id not in self.graph:
            return []
        
        lit_agents = []
        
        if direction == TraceDirection.UPSTREAM:
            lit_agents = self._trace_upstream(origin_id, max_depth)
        elif direction == TraceDirection.DOWNSTREAM:
            lit_agents = self._trace_downstream(origin_id, max_depth)
        elif direction == TraceDirection.LATERAL:
            lit_agents = self._trace_lateral(origin_id)
        
        return lit_agents
    
    def _trace_upstream(self, origin_id: str, max_depth: int) -> List[LitAgent]:
        """Illuminate ancestors"""
        lit = []
        current = origin_id
        depth = 0
        
        while depth < max_depth:
            parent_id = self.graph.get(current, {}).get("parent_id")
            if not parent_id or parent_id not in self.graph:
                break
            
            depth += 1
            luminosity = 1.0 / (1 + depth)
            
            lit.append(LitAgent(
                agent_id=parent_id,
                luminosity=luminosity,
                trace_distance=depth,
                relationship="ancestor"
            ))
            
            current = parent_id
        
        return lit
    
    def _trace_downstream(self, origin_id: str, max_depth: int) -> List[LitAgent]:
        """Illuminate descendants (BFS)"""
        lit = []
        queue = [(origin_id, 0)]
        visited = {origin_id}
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            children = self.graph.get(current, {}).get("children_ids", [])
            
            for child_id in children:
                if child_id not in visited and child_id in self.graph:
                    visited.add(child_id)
                    child_depth = depth + 1
                    luminosity = 1.0 / (1 + child_depth)
                    
                    lit.append(LitAgent(
                        agent_id=child_id,
                        luminosity=luminosity,
                        trace_distance=child_depth,
                        relationship="descendant"
                    ))
                    
                    queue.append((child_id, child_depth))
        
        return lit
    
    def _trace_lateral(self, origin_id: str) -> List[LitAgent]:
        """Illuminate siblings (agents sharing same parent)"""
        lit = []
        parent_id = self.graph.get(origin_id, {}).get("parent_id")
        
        if not parent_id or parent_id not in self.graph:
            return lit
        
        siblings = self.graph.get(parent_id, {}).get("children_ids", [])
        
        for sibling_id in siblings:
            if sibling_id != origin_id and sibling_id in self.graph:
                lit.append(LitAgent(
                    agent_id=sibling_id,
                    luminosity=0.8,  # High luminosity for direct siblings
                    trace_distance=1,
                    relationship="sibling"
                ))
        
        return lit


# =============================================================================
# SWARM WEATHER (G.5)
# =============================================================================

class WeatherCondition(Enum):
    CLEAR_SKIES = "clear_skies"    # High SCI, Low Risk
    DRIFT_STORM = "drift_storm"    # High drift velocity
    FOG = "fog"                    # High entropy (S_ψ → 1.0)
    DROUGHT = "drought"            # Low activity/growth


@dataclass
class SwarmWeather:
    """Weather report for swarm health visualization"""
    condition: WeatherCondition
    evi: float                     # Environmental Volatility Index (0-1)
    drift_rate: float              # Global drift velocity
    entropy_level: float           # Average entropy
    activity_level: float          # Growth/discovery rate
    description: str
    recommended_action: str


class SwarmWeatherStation:
    """
    Generates swarm weather reports from Appendix G.5.
    
    Conditions:
    - Drift Storm: High Rate_drift → Raise CAC to Level 4
    - Fog: High S_ψ → Lower autonomy, wait for clarity
    - Drought: Low Activity → Activate CDP pruning
    - Clear Skies: High SCI, Low Risk → Authorize AGM expansion
    """
    
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "drift_storm": 0.5,     # Drift velocity threshold
            "fog": 0.8,             # Entropy threshold
            "drought_activity": 0.2, # Activity threshold
            "clear_sci": 0.8        # SCI threshold for clear
        }
    
    def calculate_evi(self, sci: float, drift_rate: float, 
                      entropy: float, activity: float) -> float:
        """
        Environmental Volatility Index.
        
        EVI = w1*(1-SCI) + w2*drift + w3*entropy + w4*(1-activity)
        
        Low EVI = Stable swarm
        High EVI = Volatile swarm
        """
        w1, w2, w3, w4 = 0.3, 0.3, 0.2, 0.2
        
        evi = (
            w1 * (1 - sci) +
            w2 * min(drift_rate, 1.0) +
            w3 * entropy +
            w4 * (1 - activity)
        )
        
        return min(max(evi, 0.0), 1.0)
    
    def get_weather(self, sci: float, drift_rate: float,
                    entropy: float, activity: float) -> SwarmWeather:
        """
        Generate weather report from current swarm metrics.
        """
        evi = self.calculate_evi(sci, drift_rate, entropy, activity)
        
        # Determine dominant condition
        if drift_rate > self.thresholds["drift_storm"]:
            condition = WeatherCondition.DRIFT_STORM
            description = f"Drift Storm: High drift velocity ({drift_rate:.2f})"
            action = "Raise CAC to Level 4. Investigate drift clusters."
            
        elif entropy > self.thresholds["fog"]:
            condition = WeatherCondition.FOG
            description = f"Fog: High entropy density ({entropy:.2f})"
            action = "Lower autonomy. Wait for signal clarity."
            
        elif activity < self.thresholds["drought_activity"]:
            condition = WeatherCondition.DROUGHT
            description = f"Drought: Low activity ({activity:.2f})"
            action = "Activate CDP pruning to recover resources."
            
        else:
            condition = WeatherCondition.CLEAR_SKIES
            description = f"Clear Skies: Stable swarm (SCI={sci:.2f})"
            action = "Authorize AGM expansion. Growth conditions favorable."
        
        return SwarmWeather(
            condition=condition,
            evi=evi,
            drift_rate=drift_rate,
            entropy_level=entropy,
            activity_level=activity,
            description=description,
            recommended_action=action
        )


# =============================================================================
# GENESIS EVENT CONSOLE (G.6)
# =============================================================================

class SeasonType(Enum):
    SPRING = "spring"   # High growth, resource expansion
    SUMMER = "summer"   # Balanced growth and consolidation
    AUTUMN = "autumn"   # Harvest discoveries, reduce expansion
    WINTER = "winter"   # Conservation, aggressive pruning


@dataclass
class GenesisEvent:
    """A deployment simulation event"""
    event_id: str
    timestamp: datetime
    event_type: str              # "STEM_DEPLOY", "WINTER_START", "BIFURCATION", etc.
    affected_agents: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    success: bool
    notes: str


class GenesisEventConsole:
    """
    Deployment simulation dashboard from Appendix G.6.
    
    Capabilities:
    - Deploy stem agents with high entropy budgets
    - Simulate "data winters" (resource scarcity)
    - Monitor lineage bifurcations and survival
    - Track real-time FDR, SCI, budget consumption
    """
    
    def __init__(self):
        self.events: List[GenesisEvent] = []
        self.current_season = SeasonType.SUMMER
        self.stem_agents: Set[str] = set()
        
    def deploy_stem_agents(self, count: int, entropy_budget: float) -> GenesisEvent:
        """Deploy new stem agents with high entropy budgets"""
        new_agents = [f"stem_{i:04d}" for i in range(count)]
        self.stem_agents.update(new_agents)
        
        event = GenesisEvent(
            event_id=f"genesis_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            event_type="STEM_DEPLOY",
            affected_agents=new_agents,
            metrics_before={"stem_count": len(self.stem_agents) - count},
            metrics_after={"stem_count": len(self.stem_agents), "budget_each": entropy_budget},
            success=True,
            notes=f"Deployed {count} stem agents with budget {entropy_budget}"
        )
        
        self.events.append(event)
        return event
    
    def declare_season(self, season: SeasonType) -> GenesisEvent:
        """Change global season (affects all growth/pruning behavior)"""
        old_season = self.current_season
        self.current_season = season
        
        event = GenesisEvent(
            event_id=f"season_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            event_type="SEASON_CHANGE",
            affected_agents=list(self.stem_agents),
            metrics_before={"season": old_season.value},
            metrics_after={"season": season.value},
            success=True,
            notes=f"Season changed: {old_season.value} → {season.value}"
        )
        
        self.events.append(event)
        return event
    
    def simulate_data_winter(self, duration_ticks: int, 
                             resource_reduction: float) -> GenesisEvent:
        """
        Simulate hostile data environment.
        
        Tests swarm resilience:
        - Can it survive resource scarcity?
        - Does CDP activate correctly?
        - Do strong lineages persist?
        """
        event = GenesisEvent(
            event_id=f"winter_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            event_type="DATA_WINTER",
            affected_agents=list(self.stem_agents),
            metrics_before={"resource_level": 1.0},
            metrics_after={
                "resource_level": 1.0 - resource_reduction,
                "duration_ticks": duration_ticks
            },
            success=True,
            notes=f"Data winter: {resource_reduction*100:.0f}% resource reduction for {duration_ticks} ticks"
        )
        
        self.events.append(event)
        return event


# =============================================================================
# BIOSPHERE COMMAND INTERFACE (MAIN CLASS)
# =============================================================================

class BiosphereCommandInterface:
    """
    Main BCI class integrating all components from Appendix G.
    
    Provides unified API for:
    - Semantic zoom navigation
    - Cognitive sonar queries
    - Hereditary lighting
    - Swarm weather monitoring
    - Genesis event simulation
    """
    
    def __init__(self, swarm_state: Dict, lineage_graph: Dict):
        """
        Initialize BCI with swarm state and lineage data.
        
        Args:
            swarm_state: Dict mapping agent_id → phi_vector
            lineage_graph: Dict mapping agent_id → {parent_id, children_ids, ...}
        """
        self.swarm_state = swarm_state
        self.lineage_graph = lineage_graph
        
        # Initialize subsystems
        self.sonar = CognitiveSonar(swarm_state)
        self.lighting = HereditaryLighting(lineage_graph)
        self.weather_station = SwarmWeatherStation()
        self.genesis_console = GenesisEventConsole()
        
        # Current zoom state
        self.current_zoom = ZoomLevel.CANOPY
        self.selected_cluster = None
        self.selected_branch = None
        self.selected_agent = None
    
    # -------------------------------------------------------------------------
    # SEMANTIC ZOOM (G.2)
    # -------------------------------------------------------------------------
    
    def zoom_to(self, level: ZoomLevel, target_id: str = None):
        """Navigate to a specific zoom level"""
        self.current_zoom = level
        
        if level == ZoomLevel.TRUNK:
            self.selected_cluster = target_id
        elif level == ZoomLevel.BRANCH:
            self.selected_branch = target_id
        elif level == ZoomLevel.LEAF:
            self.selected_agent = target_id
    
    def get_canopy_view(self) -> CanopyView:
        """Get global swarm view"""
        total = len(self.swarm_state)
        active = sum(1 for aid in self.lineage_graph 
                    if self.lineage_graph[aid].get("status") == "ACTIVE")
        quarantined = total - active
        
        # Calculate global metrics
        sci = self._calculate_global_sci()
        fdr = self._calculate_fdr()
        
        # Get weather
        weather = self.weather_station.get_weather(
            sci=sci,
            drift_rate=self._calculate_drift_rate(),
            entropy=self._calculate_entropy(),
            activity=fdr
        )
        
        return CanopyView(
            total_agents=total,
            active_agents=active,
            quarantined_agents=quarantined,
            global_sci=sci,
            global_fdr=fdr,
            entropy_budget_remaining=1.0,  # Placeholder
            drift_hotspots=[],  # Would be computed from cluster analysis
            growth_zones=[],
            weather_condition=weather.condition.value,
            evi=weather.evi
        )
    
    def get_leaf_view(self, agent_id: str) -> Optional[LeafView]:
        """Get full forensic view of single agent"""
        if agent_id not in self.swarm_state:
            return None
        
        phi = self.swarm_state[agent_id]
        info = self.lineage_graph.get(agent_id, {})
        
        return LeafView(
            agent_id=agent_id,
            phi_vector=phi,
            stability=info.get("stability", 1.0),
            entropy=self.sonar._calculate_temperature(phi),
            ttf=info.get("ttf", 1000.0),
            trace_level=info.get("trace_level", 1),
            status=info.get("status", "UNKNOWN"),
            parent_id=info.get("parent_id"),
            children_ids=info.get("children_ids", []),
            lkc_log=info.get("lkc_log", []),
            event_history=info.get("events", [])
        )
    
    # -------------------------------------------------------------------------
    # COGNITIVE SONAR (G.3)
    # -------------------------------------------------------------------------
    
    def ping(self, query_vector: np.ndarray, threshold: float = 0.8) -> SonarResult:
        """Execute cognitive sonar ping"""
        return self.sonar.ping(query_vector, threshold)
    
    def ping_by_pattern(self, pattern_name: str) -> SonarResult:
        """Ping using predefined pattern queries"""
        patterns = {
            "high_drift": np.array([0.8, 0.1, 0.1] + [0.0] * 45),
            "low_stability": np.array([0.1, 0.8, 0.1] + [0.0] * 45),
            "consensus_loss": np.array([0.1, 0.1, 0.8] + [0.0] * 45),
        }
        
        if pattern_name in patterns:
            return self.sonar.ping(patterns[pattern_name], threshold=0.6)
        
        return SonarResult(
            query_vector=np.zeros(48),
            threshold=0.0,
            echoes=[],
            total_scanned=0,
            response_time_ms=0
        )
    
    # -------------------------------------------------------------------------
    # HEREDITARY LIGHTING (G.4)
    # -------------------------------------------------------------------------
    
    def illuminate(self, agent_id: str, direction: TraceDirection,
                   max_depth: int = 10) -> List[LitAgent]:
        """Illuminate lineage from specified agent"""
        return self.lighting.trace(agent_id, direction, max_depth)
    
    def trace_root_cause(self, agent_id: str) -> List[LitAgent]:
        """Trace upstream to find root cause of failure"""
        return self.illuminate(agent_id, TraceDirection.UPSTREAM)
    
    def trace_contagion(self, agent_id: str) -> List[LitAgent]:
        """Trace downstream and lateral to map contagion risk"""
        downstream = self.illuminate(agent_id, TraceDirection.DOWNSTREAM)
        lateral = self.illuminate(agent_id, TraceDirection.LATERAL)
        return downstream + lateral
    
    # -------------------------------------------------------------------------
    # SWARM WEATHER (G.5)
    # -------------------------------------------------------------------------
    
    def get_weather(self) -> SwarmWeather:
        """Get current swarm weather report"""
        return self.weather_station.get_weather(
            sci=self._calculate_global_sci(),
            drift_rate=self._calculate_drift_rate(),
            entropy=self._calculate_entropy(),
            activity=self._calculate_fdr()
        )
    
    # -------------------------------------------------------------------------
    # GENESIS CONSOLE (G.6)
    # -------------------------------------------------------------------------
    
    def deploy_stem(self, count: int, budget: float) -> GenesisEvent:
        """Deploy new stem agents"""
        return self.genesis_console.deploy_stem_agents(count, budget)
    
    def set_season(self, season: SeasonType) -> GenesisEvent:
        """Change global season"""
        return self.genesis_console.declare_season(season)
    
    def simulate_winter(self, duration: int, severity: float) -> GenesisEvent:
        """Simulate data winter"""
        return self.genesis_console.simulate_data_winter(duration, severity)
    
    # -------------------------------------------------------------------------
    # HELPER METHODS
    # -------------------------------------------------------------------------
    
    def _calculate_global_sci(self) -> float:
        """Calculate swarm-wide Symbolic Coherence Index"""
        if not self.swarm_state:
            return 0.0
        
        stabilities = [
            self.lineage_graph.get(aid, {}).get("stability", 1.0)
            for aid in self.swarm_state
        ]
        return np.mean(stabilities)
    
    def _calculate_drift_rate(self) -> float:
        """Calculate global drift velocity"""
        if not self.lineage_graph:
            return 0.0
        
        drifts = [
            info.get("drift_velocity", 0.0)
            for info in self.lineage_graph.values()
        ]
        return np.mean(drifts) if drifts else 0.0
    
    def _calculate_entropy(self) -> float:
        """Calculate average entropy across swarm"""
        if not self.swarm_state:
            return 0.0
        
        entropies = [
            self.sonar._calculate_temperature(phi)
            for phi in self.swarm_state.values()
        ]
        return np.mean(entropies)
    
    def _calculate_fdr(self) -> float:
        """Calculate Findings Discovery Rate"""
        # Placeholder - would track actual discoveries
        return 0.5


# =============================================================================
# API ENDPOINTS FOR FRONTEND
# =============================================================================

"""
WebSocket/REST API endpoints for BCI frontend:

GET  /bci/canopy              → CanopyView
GET  /bci/trunk/{cluster_id}  → TrunkView
GET  /bci/branch/{branch_id}  → BranchView
GET  /bci/leaf/{agent_id}     → LeafView

POST /bci/sonar/ping          → SonarResult
     Body: {query_vector: [...], threshold: 0.8}

POST /bci/illuminate          → List[LitAgent]
     Body: {agent_id: "...", direction: "upstream|downstream|lateral"}

GET  /bci/weather             → SwarmWeather

POST /bci/genesis/deploy      → GenesisEvent
     Body: {count: 10, budget: 100.0}

POST /bci/genesis/season      → GenesisEvent
     Body: {season: "winter"}

WebSocket /bci/stream         → Real-time updates
     Pushes: weather changes, drift alerts, consensus events
"""


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BIOSPHERE COMMAND INTERFACE - DEMO")
    print("=" * 80)
    
    # Create mock swarm
    np.random.seed(42)
    n_agents = 100
    phi_dim = 48
    
    swarm_state = {}
    lineage_graph = {}
    
    for i in range(n_agents):
        agent_id = f"agent_{i:04d}"
        phi = np.random.randn(phi_dim)
        phi = phi / np.linalg.norm(phi)
        swarm_state[agent_id] = phi
        
        parent_id = f"agent_{i-1:04d}" if i > 0 else None
        lineage_graph[agent_id] = {
            "parent_id": parent_id,
            "children_ids": [f"agent_{i+1:04d}"] if i < n_agents - 1 else [],
            "stability": 0.5 + np.random.random() * 0.5,
            "drift_velocity": np.random.random() * 0.3,
            "status": "ACTIVE" if np.random.random() > 0.1 else "QUARANTINED"
        }
    
    # Initialize BCI
    bci = BiosphereCommandInterface(swarm_state, lineage_graph)
    
    # Demo: Canopy View
    print("\n📊 CANOPY VIEW (Global)")
    canopy = bci.get_canopy_view()
    print(f"   Total Agents: {canopy.total_agents}")
    print(f"   Active: {canopy.active_agents}")
    print(f"   Global SCI: {canopy.global_sci:.3f}")
    print(f"   Weather: {canopy.weather_condition}")
    print(f"   EVI: {canopy.evi:.3f}")
    
    # Demo: Cognitive Sonar
    print("\n🔊 COGNITIVE SONAR PING")
    query = np.random.randn(phi_dim)
    result = bci.ping(query, threshold=0.5)
    print(f"   Query threshold: 0.5")
    print(f"   Echoes found: {len(result.echoes)}")
    if result.echoes:
        top = result.echoes[0]
        print(f"   Top echo: {top.agent_id} (intensity={top.intensity:.3f}, temp={top.temperature:.3f})")
    
    # Demo: Hereditary Lighting
    print("\n💡 HEREDITARY LIGHTING")
    lit = bci.trace_root_cause("agent_0050")
    print(f"   Tracing upstream from agent_0050")
    print(f"   Ancestors illuminated: {len(lit)}")
    if lit:
        for a in lit[:3]:
            print(f"     → {a.agent_id} (luminosity={a.luminosity:.2f}, distance={a.trace_distance})")
    
    # Demo: Weather
    print("\n🌤️ SWARM WEATHER")
    weather = bci.get_weather()
    print(f"   Condition: {weather.condition.value}")
    print(f"   EVI: {weather.evi:.3f}")
    print(f"   Description: {weather.description}")
    print(f"   Action: {weather.recommended_action}")
    
    # Demo: Genesis Console
    print("\n🌱 GENESIS CONSOLE")
    event = bci.deploy_stem(5, 200.0)
    print(f"   Event: {event.event_type}")
    print(f"   Notes: {event.notes}")
    
    winter = bci.simulate_winter(100, 0.6)
    print(f"   Simulation: {winter.notes}")
    
    print("\n" + "=" * 80)
    print("BCI DEMO COMPLETE")
    print("=" * 80)
