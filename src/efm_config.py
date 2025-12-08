# efm_config.py
"""
Centralized Configuration for the EFM Biosphere System
======================================================

This module defines ALL swarm-wide constants, thresholds, policies,
and system-level control variables in ONE place.

TERMINOLOGY ALIGNMENT (Booklet 4 ↔ Code):
    B_entropy     = INIT_BUDGET (entropy budget)
    τ_FDR         = MIN_FDR (findings discovery rate threshold)
    τ_BIM         = PRUNE_THRESHOLD (behavioral integrity threshold)
    Δ_Regen       = Computed from growth metrics
    SCI           = MIN_SCI (symbolic coherence index)
    ∇Ψ            = growth_vector (gradient of purpose field)

Usage:
    from efm_config import EFMConfig as cfg
    
    if branch.resource_budget > cfg.GENESIS_THRESHOLD:
        # Reproduce
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class LogLevel(Enum):
    """Logging verbosity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    QUIET = "QUIET"


class BenchmarkPreset(Enum):
    """Pre-configured benchmark scenarios."""
    GENESIS = "genesis"           # Standard 50-agent spawn
    HOSTILE_MATRIX = "hostile"    # Adversarial drift injection
    DATA_WINTER = "winter"        # Resource scarcity test
    HYPERSCALE = "hyperscale"     # 1M agent stress test


@dataclass
class EFMConfig:
    """
    Centralized configuration for the EFM Biosphere System.
    
    These parameters govern entropy budgets, swarm behavior, thresholds,
    and system-level policies for benchmarking and mission control.
    
    All values are documented with their Booklet 4 terminology.
    """
    
    # === SYSTEM-WIDE CONTROLS ===
    TICK_RATE: float = 1.0              # Seconds per tick
    MAX_TICKS: int = 100                # For benchmarks / demos
    LOG_LEVEL: str = "INFO"             # DEBUG / INFO / QUIET
    RANDOM_SEED: Optional[int] = None   # Set for reproducibility
    
    # === AGENT INITIALIZATION ===
    INIT_BUDGET: float = 500.0          # B_entropy starting value
    GENESIS_THRESHOLD: float = 150.0    # Reproduction cutoff
    PRUNE_THRESHOLD: float = 0.1        # τ_BIM: Min health before pruning
    
    # === GROWTH PHYSICS ===
    GROWTH_COST_FACTOR: float = 0.1     # Cost multiplier per drift unit
    RESOURCE_RECOVERY_RATE: float = 0.5 # % resource recovered on pruning (compost)
    
    # === METRIC THRESHOLDS ===
    MIN_SCI: float = 0.80               # Stability threshold (SCI)
    MIN_FDR: float = 0.90               # τ_FDR: Success classification
    DRIFT_ALERT_THRESHOLD: float = 0.8  # Triggers Drift Storm status
    
    # === CONSENSUS PARAMETERS ===
    CONSENSUS_QUORUM: float = 0.75      # % needed for action agreement
    CONSENSUS_TIMEOUT: int = 5          # Ticks before retry
    BYZANTINE_TOLERANCE: float = 0.33   # Max fraction of byzantine nodes
    
    # === VISUALIZATION (BCI) ===
    COLOR_HOT: tuple = (255, 0, 0)      # High activity (Red)
    COLOR_COLD: tuple = (0, 0, 255)     # Low activity (Blue)
    DIM_LUMINOSITY: float = 0.1
    GLOW_LUMINOSITY: float = 0.8
    MAX_LUMINOSITY: float = 1.0
    
    # === MISSION SCALING ===
    MISSION_SCALE: Dict[str, int] = field(default_factory=lambda: {
        'seed': 50,           # Genesis Event scale
        'small': 500,
        'medium': 10000,
        'large': 100000,
        'hyperscale': 1000000,
    })
    
    # === INVARIANT BOUNDS (for assertions) ===
    ENTROPY_MIN: float = 0.0            # B_entropy >= 0 always
    ENTROPY_MAX: float = 10000.0        # Upper sanity bound
    DRIFT_MIN: float = 0.0              # Drift ∈ [0, 1]
    DRIFT_MAX: float = 1.0
    SCI_MIN: float = 0.0                # SCI ∈ [0, 1]
    SCI_MAX: float = 1.0
    FDR_MIN: float = 0.0                # FDR ∈ [0, 1]
    FDR_MAX: float = 1.0
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        assert 0 < self.CONSENSUS_QUORUM <= 1, "Quorum must be in (0, 1]"
        assert 0 < self.BYZANTINE_TOLERANCE < 0.5, "Byzantine tolerance must be < 0.5"
        assert self.MIN_SCI >= 0 and self.MIN_SCI <= 1, "MIN_SCI must be in [0, 1]"
        assert self.MIN_FDR >= 0 and self.MIN_FDR <= 1, "MIN_FDR must be in [0, 1]"
        assert self.INIT_BUDGET > 0, "Initial budget must be positive"
        assert self.GENESIS_THRESHOLD > 0, "Genesis threshold must be positive"
        return True
    
    @classmethod
    def for_preset(cls, preset: BenchmarkPreset) -> 'EFMConfig':
        """Create config for a benchmark preset."""
        if preset == BenchmarkPreset.GENESIS:
            return cls(
                MAX_TICKS=100,
                INIT_BUDGET=500.0,
                GENESIS_THRESHOLD=150.0,
                LOG_LEVEL="INFO"
            )
        elif preset == BenchmarkPreset.HOSTILE_MATRIX:
            return cls(
                MAX_TICKS=200,
                INIT_BUDGET=300.0,  # Less resources
                DRIFT_ALERT_THRESHOLD=0.5,  # More sensitive
                LOG_LEVEL="DEBUG"
            )
        elif preset == BenchmarkPreset.DATA_WINTER:
            return cls(
                MAX_TICKS=150,
                INIT_BUDGET=200.0,  # Scarce
                RESOURCE_RECOVERY_RATE=0.3,  # Less recovery
                LOG_LEVEL="INFO"
            )
        elif preset == BenchmarkPreset.HYPERSCALE:
            return cls(
                MAX_TICKS=50,
                INIT_BUDGET=1000.0,
                LOG_LEVEL="QUIET"  # Reduce noise
            )
        return cls()


# === INVARIANT CHECKER ===

class InvariantError(Exception):
    """Raised when a system invariant is violated."""
    pass


def check_invariants(cfg: EFMConfig, **metrics):
    """
    Check system invariants against current metrics.
    
    Args:
        cfg: EFMConfig instance
        **metrics: Current metric values (B_entropy, drift, SCI, FDR, etc.)
    
    Raises:
        InvariantError: If any invariant is violated
    """
    violations = []
    
    # Entropy budget bounds
    if 'B_entropy' in metrics:
        if metrics['B_entropy'] < cfg.ENTROPY_MIN:
            violations.append(f"B_entropy={metrics['B_entropy']} < {cfg.ENTROPY_MIN}")
        if metrics['B_entropy'] > cfg.ENTROPY_MAX:
            violations.append(f"B_entropy={metrics['B_entropy']} > {cfg.ENTROPY_MAX}")
    
    # Drift bounds
    if 'drift' in metrics:
        if not (cfg.DRIFT_MIN <= metrics['drift'] <= cfg.DRIFT_MAX):
            violations.append(f"drift={metrics['drift']} not in [{cfg.DRIFT_MIN}, {cfg.DRIFT_MAX}]")
    
    # SCI bounds
    if 'SCI' in metrics:
        if not (cfg.SCI_MIN <= metrics['SCI'] <= cfg.SCI_MAX):
            violations.append(f"SCI={metrics['SCI']} not in [{cfg.SCI_MIN}, {cfg.SCI_MAX}]")
    
    # FDR bounds
    if 'FDR' in metrics:
        if not (cfg.FDR_MIN <= metrics['FDR'] <= cfg.FDR_MAX):
            violations.append(f"FDR={metrics['FDR']} not in [{cfg.FDR_MIN}, {cfg.FDR_MAX}]")
    
    if violations:
        raise InvariantError(f"Invariant violations: {'; '.join(violations)}")


# === TICK LOG SCHEMA ===

@dataclass
class TickLogEntry:
    """
    Standardized log entry for each tick.
    
    This schema enables:
    - Reproduction of diagrams from logs alone
    - Benchmark auditability
    - Cross-run comparison
    """
    tick: int
    agent_id: str
    position: tuple  # (x, y, z) or higher-dimensional
    FDR: float       # Findings Discovery Rate
    SCI: float       # Symbolic Coherence Index
    B_entropy: float # Entropy budget
    health: float    # Agent health [0, 1]
    growth_vector: tuple  # ∇Ψ direction
    status: str      # ACTIVE, DORMANT, QUARANTINED, PRUNED
    action: str      # GROW, EXPLORE, HEAL, PRUNE, IDLE
    branch_id: str   # Parent branch identifier
    mission_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "tick": self.tick,
            "agent_id": self.agent_id,
            "position": list(self.position),
            "FDR": self.FDR,
            "SCI": self.SCI,
            "B_entropy": self.B_entropy,
            "health": self.health,
            "growth_vector": list(self.growth_vector),
            "status": self.status,
            "action": self.action,
            "branch_id": self.branch_id,
            "mission_id": self.mission_id
        }


# === DEMO ===

if __name__ == "__main__":
    print("=" * 60)
    print("EFM CONFIGURATION MODULE")
    print("=" * 60)
    
    # Default config
    cfg = EFMConfig()
    cfg.validate()
    
    print("\n📋 DEFAULT CONFIG:")
    print(f"   B_entropy (INIT_BUDGET): {cfg.INIT_BUDGET}")
    print(f"   τ_FDR (MIN_FDR): {cfg.MIN_FDR}")
    print(f"   τ_BIM (PRUNE_THRESHOLD): {cfg.PRUNE_THRESHOLD}")
    print(f"   MIN_SCI: {cfg.MIN_SCI}")
    print(f"   CONSENSUS_QUORUM: {cfg.CONSENSUS_QUORUM}")
    print(f"   GENESIS_THRESHOLD: {cfg.GENESIS_THRESHOLD}")
    
    # Preset configs
    print("\n🧪 BENCHMARK PRESETS:")
    for preset in BenchmarkPreset:
        p_cfg = EFMConfig.for_preset(preset)
        print(f"   {preset.value}: B_entropy={p_cfg.INIT_BUDGET}, ticks={p_cfg.MAX_TICKS}")
    
    # Invariant check demo
    print("\n🔒 INVARIANT CHECK:")
    try:
        check_invariants(cfg, B_entropy=100, drift=0.5, SCI=0.85, FDR=0.95)
        print("   ✅ All invariants passed")
    except InvariantError as e:
        print(f"   ❌ {e}")
    
    try:
        check_invariants(cfg, B_entropy=-50)  # Should fail
    except InvariantError as e:
        print(f"   ✅ Caught violation: {e}")
    
    # Log entry demo
    print("\n📊 TICK LOG SCHEMA:")
    entry = TickLogEntry(
        tick=21,
        agent_id="ROOT.sub.4",
        position=(54, 53, 51),
        FDR=1.0,
        SCI=0.91,
        B_entropy=231.5,
        health=1.0,
        growth_vector=(1, 0, -1),
        status="ACTIVE",
        action="GROW",
        branch_id="ROOT.sub",
        mission_id="mission_042"
    )
    import json
    print(f"   {json.dumps(entry.to_dict(), indent=2)}")
    
    print("\n" + "=" * 60)
    print("✅ Config module ready")
    print("=" * 60)
