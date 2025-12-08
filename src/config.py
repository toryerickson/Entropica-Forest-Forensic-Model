"""
EFM Configuration & Invariant Enforcement
==========================================

This module centralizes ALL hyperparameters and provides runtime invariant
checking to harden the EFM from "research prototype" to "auditable system."

IMPORTANT: This is a RESEARCH-GRADE implementation. While it implements
the concepts from Booklet 4 with algorithmic fidelity, it is NOT yet
production-hardened for deployment. Specifically:

- Cryptographic signatures are PLACEHOLDERS (not real crypto)
- Consensus is BYZANTINE-INSPIRED (not formal PBFT)
- ZK proofs are API STUBS (not actual circuits)

See individual modules for specific limitations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import logging
from functools import wraps


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(name)s | %(message)s'
)

logger = logging.getLogger("EFM")


# =============================================================================
# INVARIANT VIOLATION EXCEPTIONS
# =============================================================================

class InvariantViolation(Exception):
    """Raised when a core system invariant is violated."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class BudgetExhausted(Exception):
    """Raised when entropy budget cannot fulfill request."""
    pass


# =============================================================================
# CENTRALIZED CONFIGURATION
# =============================================================================

@dataclass
class RegenerativeConfig:
    """
    Configuration for Regenerative Architecture (CDP, Budget, AGM, RET, AES).
    
    All thresholds are documented with their rationale.
    Terminology aligned with Booklet 4:
        - B_entropy = entropy_budget
        - τ_FDR = tau_fdr  
        - Δ_Regen = delta_regen
    """
    
    # Context Decay Pruning (CDP)
    tau_fdr: float = 0.15          # τ_FDR: prune if below (15% discovery rate)
    tau_bim: float = 0.20          # τ_BIM: prune if below (20% integrity)
    min_capsule_age_ticks: int = 10  # Minimum age before eligible for pruning
    
    # Entropy Budget (B_entropy)
    B_entropy_init: float = 500.0   # B_entropy starting value (was 1000)
    min_reserve_ratio: float = 0.20 # Reserve 20% - never spend below this
    max_allocation_per_purpose: float = 0.15  # Max 15% per single purpose
    
    # Regenerative Efficiency Test (RET)
    ret_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "SURPLUS": 0.5,    # Fast rebuild when budget is high
        "HEALTHY": 0.75,   # Normal rebuild
        "LOW": 1.0,        # Slower rebuild
        "CRITICAL": 1.5    # Much slower when budget exhausted
    })
    ret_improvement_target: float = 0.45  # 45% improvement required to pass
    
    # Autonomous Growth Module (AGM)
    genesis_threshold: float = 150.0      # Reproduction cutoff for branching
    growth_stability_threshold: float = 0.7  # Min stability to allow branching
    growth_cost_factor: float = 0.1       # Cost multiplier per drift unit
    resource_recovery_rate: float = 0.5   # % resource recovered on pruning
    
    # Adaptive Energy Scaling (AES)
    risk_cost_base: float = 10.0     # Base cost per unit risk
    risk_cost_exponent: float = 1.5  # Superlinear scaling with risk


@dataclass
class SystemConfig:
    """
    System-wide controls for tick rate, logging, and benchmarks.
    """
    tick_rate: float = 1.0           # Seconds per tick
    max_ticks: int = 100             # For benchmarks/demos
    log_level: str = "INFO"          # DEBUG / INFO / QUIET
    
    # Metric Thresholds
    min_sci: float = 0.80            # Stability threshold (SCI)
    min_fdr: float = 0.90            # Success classification threshold
    drift_alert_threshold: float = 0.8  # Triggers Drift Storm status
    
    # Mission Scaling Presets
    mission_scales: Dict[str, int] = field(default_factory=lambda: {
        'seed': 50,           # Genesis Event scale
        'small': 500,
        'medium': 10000,
        'large': 100000,
        'hyperscale': 1000000,
    })


@dataclass
class VisualizationConfig:
    """
    Configuration for BCI visualization colors and luminosity.
    """
    color_hot: tuple = (255, 0, 0)      # High activity (Red)
    color_cold: tuple = (0, 0, 255)     # Low activity (Blue)
    dim_luminosity: float = 0.1
    glow_luminosity: float = 0.8
    max_luminosity: float = 1.0


@dataclass
class ForestConfig:
    """
    Configuration for Forest Architecture (Anomaly Detection, Missions, Branches).
    
    NOTE: Several components use STOCHASTIC selection (np.random.choice).
    Results will vary across runs. Report averages over multiple seeds.
    """
    
    # Anomaly Detection
    knn_neighbors: int = 5              # k for density estimation
    density_percentile_threshold: float = 0.1  # Bottom 10% = anomalous
    min_cluster_size: int = 3           # Min points for pattern cluster
    
    # Mission Generation (STOCHASTIC - uses random selection)
    max_concurrent_missions: int = 10
    mission_budget_ratio: float = 0.1   # Allocate 10% of budget per mission
    
    # Branch Management
    max_branch_depth: int = 50          # Warn if lineage exceeds this
    trunk_stability_min: float = 0.8    # Trunk must stay above this
    
    # Compost Cycle
    compost_recovery_ratio: float = 0.5 # Recover 50% of pruned resources
    
    # Scalability (O(n²) warning threshold)
    n_squared_warning_threshold: int = 10000  # Warn if n > this


@dataclass
class ProductionConfig:
    """
    Configuration for Production Core (Embeddings, Consensus, Validation).
    
    IMPORTANT LIMITATIONS:
    - Signatures use MD5 PLACEHOLDERS (not cryptographically secure)
    - Consensus is BYZANTINE-INSPIRED (simplified from formal PBFT)
    - No persistent storage or distributed runtime
    """
    
    # Semantic Embeddings
    embedding_dimensions: int = 48
    similarity_threshold: float = 0.7   # Min cosine similarity for match
    
    # Byzantine-Inspired Consensus (NOT formal BFT)
    min_quorum_ratio: float = 0.67      # 2/3 for consensus
    max_byzantine_ratio: float = 0.33   # Tolerate up to 1/3 byzantine
    reputation_decay: float = 0.95      # Per-round reputation decay
    initial_reputation: float = 1.0
    
    # Validation
    min_confidence_for_production: float = 0.8
    human_review_threshold: float = 0.6  # Below this triggers review
    
    # API
    max_patterns_per_ingest: int = 100
    pattern_ttl_ticks: int = 1000


@dataclass
class BCIConfig:
    """
    Configuration for Biosphere Command Interface.
    """
    
    # Swarm Weather Thresholds
    drift_storm_threshold: float = 0.5
    fog_entropy_threshold: float = 0.8
    drought_activity_threshold: float = 0.2
    clear_sci_threshold: float = 0.8
    
    # EVI Weights (must sum to 1.0)
    evi_weight_sci: float = 0.3
    evi_weight_drift: float = 0.3
    evi_weight_entropy: float = 0.2
    evi_weight_activity: float = 0.2
    
    # Hereditary Lighting
    max_trace_depth: int = 10
    luminosity_decay_factor: float = 1.0  # L = 1 / (1 + distance * factor)


@dataclass 
class EFMConfig:
    """
    Master configuration aggregating all subsystem configs.
    
    Usage:
        config = EFMConfig()
        budget = EntropyBudget(config.regenerative)
        forest = CognitiveForest(config.forest)
    """
    
    regenerative: RegenerativeConfig = field(default_factory=RegenerativeConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)
    production: ProductionConfig = field(default_factory=ProductionConfig)
    bci: BCIConfig = field(default_factory=BCIConfig)
    
    # Global settings
    random_seed: Optional[int] = None   # Set for reproducibility
    log_level: str = "INFO"
    strict_invariants: bool = True      # Raise on violation vs warn
    
    def validate(self):
        """Validate configuration consistency."""
        # EVI weights must sum to 1.0
        evi_sum = (self.bci.evi_weight_sci + self.bci.evi_weight_drift +
                   self.bci.evi_weight_entropy + self.bci.evi_weight_activity)
        if abs(evi_sum - 1.0) > 0.001:
            raise ConfigurationError(f"EVI weights must sum to 1.0, got {evi_sum}")
        
        # Reserve ratio must be valid
        if not 0 < self.regenerative.min_reserve_ratio < 1:
            raise ConfigurationError("Reserve ratio must be between 0 and 1")
        
        # Byzantine tolerance must be valid
        if self.production.max_byzantine_ratio >= 0.5:
            raise ConfigurationError("Byzantine ratio must be < 0.5 for consensus")
        
        logger.info("Configuration validated successfully")
        return True


# =============================================================================
# INVARIANT ENFORCEMENT
# =============================================================================

class InvariantChecker:
    """
    Runtime invariant checking system.
    
    Invariants are conditions that must ALWAYS be true. Violations indicate
    bugs in the implementation, not user errors.
    """
    
    def __init__(self, strict: bool = True):
        self.strict = strict
        self.violations: List[Dict] = []
        self.checks_passed: int = 0
        self.checks_failed: int = 0
    
    def check(self, condition: bool, message: str, context: Dict = None):
        """
        Check an invariant condition.
        
        Args:
            condition: Must be True
            message: Description if violated
            context: Additional debug info
        """
        if condition:
            self.checks_passed += 1
            return True
        
        self.checks_failed += 1
        violation = {
            "message": message,
            "context": context or {},
            "timestamp": __import__('time').time()
        }
        self.violations.append(violation)
        
        if self.strict:
            raise InvariantViolation(f"INVARIANT VIOLATED: {message} | Context: {context}")
        else:
            logger.warning(f"INVARIANT VIOLATED (non-strict): {message}")
            return False
    
    def report(self) -> str:
        """Generate invariant check report."""
        total = self.checks_passed + self.checks_failed
        lines = [
            "=" * 60,
            "INVARIANT CHECK REPORT",
            "=" * 60,
            f"Total checks: {total}",
            f"Passed: {self.checks_passed}",
            f"Failed: {self.checks_failed}",
        ]
        
        if self.violations:
            lines.append("\nViolations:")
            for v in self.violations[-10:]:  # Last 10
                lines.append(f"  - {v['message']}")
        
        return "\n".join(lines)


# Global invariant checker instance
_invariant_checker = InvariantChecker(strict=True)


def check_invariant(condition: bool, message: str, context: Dict = None):
    """Convenience function for invariant checking."""
    return _invariant_checker.check(condition, message, context)


def invariant_report() -> str:
    """Get global invariant report."""
    return _invariant_checker.report()


# =============================================================================
# INVARIANT DECORATOR
# =============================================================================

def enforce_invariants(*invariant_checks: Callable):
    """
    Decorator to enforce invariants before and after method execution.
    
    Usage:
        @enforce_invariants(
            lambda self: self.budget >= 0,
            lambda self: self.knowledge >= 0
        )
        def process_tick(self):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Pre-conditions
            for check in invariant_checks:
                result = check(self)
                check_invariant(
                    result,
                    f"Pre-condition failed for {func.__name__}: {check.__doc__ or 'unnamed'}",
                    {"method": func.__name__, "class": type(self).__name__}
                )
            
            # Execute
            result = func(self, *args, **kwargs)
            
            # Post-conditions
            for check in invariant_checks:
                result_check = check(self)
                check_invariant(
                    result_check,
                    f"Post-condition failed for {func.__name__}: {check.__doc__ or 'unnamed'}",
                    {"method": func.__name__, "class": type(self).__name__}
                )
            
            return result
        return wrapper
    return decorator


# =============================================================================
# CORE INVARIANTS (Referenced by all modules)
# =============================================================================

class CoreInvariants:
    """
    Central definition of system-wide invariants.
    
    These are the "laws" of the EFM system that must never be violated.
    """
    
    @staticmethod
    def budget_non_negative(budget_value: float) -> bool:
        """Entropy budget must never be negative."""
        return budget_value >= 0
    
    @staticmethod
    def budget_above_reserve(current: float, reserve: float) -> bool:
        """Current budget must stay above minimum reserve."""
        return current >= reserve
    
    @staticmethod
    def knowledge_monotonic_unless_pruned(old: float, new: float, pruned: float) -> bool:
        """Knowledge can only decrease through explicit pruning."""
        return new >= old - pruned
    
    @staticmethod
    def confidence_bounded(confidence: float) -> bool:
        """Confidence scores must be in [0, 1]."""
        return 0.0 <= confidence <= 1.0
    
    @staticmethod
    def reputation_bounded(reputation: float) -> bool:
        """Reputation scores must be in [0, 1]."""
        return 0.0 <= reputation <= 1.0
    
    @staticmethod
    def stability_bounded(stability: float) -> bool:
        """Stability scores must be in [0, 1]."""
        return 0.0 <= stability <= 1.0
    
    @staticmethod
    def tick_monotonic(old_tick: int, new_tick: int) -> bool:
        """Tick counter must only increase."""
        return new_tick >= old_tick
    
    @staticmethod
    def lineage_acyclic(parent_id: str, child_id: str) -> bool:
        """Parent cannot equal child (basic cycle check)."""
        return parent_id != child_id


# =============================================================================
# STOCHASTICITY DOCUMENTATION
# =============================================================================

class StochasticComponent:
    """
    Marker class to document stochastic behavior.
    
    Components using random selection should inherit from this
    or be marked with @stochastic decorator.
    """
    
    STOCHASTIC_NOTE = """
    NOTE: This component uses STOCHASTIC selection.
    
    Results will vary across runs with different random seeds.
    For reproducible results:
    1. Set random_seed in EFMConfig
    2. Report results as averages over multiple runs
    3. Include standard deviation in metrics
    """


def stochastic(description: str = "Uses random selection"):
    """
    Decorator to mark stochastic methods.
    
    Usage:
        @stochastic("Mission selection uses np.random.choice")
        def select_mission(self, candidates):
            ...
    """
    def decorator(func):
        func._stochastic = True
        func._stochastic_description = description
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._stochastic = True
        wrapper._stochastic_description = description
        return wrapper
    return decorator


# =============================================================================
# METRICS EXPORT
# =============================================================================

class MetricsExporter:
    """
    Unified metrics collection and export.
    
    Ensures booklet tables can be generated consistently from code output.
    """
    
    def __init__(self):
        self.metrics: List[Dict] = []
    
    def record(self, category: str, name: str, value: Any, 
               tick: int = None, context: Dict = None):
        """Record a metric."""
        self.metrics.append({
            "category": category,
            "name": name,
            "value": value,
            "tick": tick,
            "context": context or {},
            "timestamp": __import__('time').time()
        })
    
    def export_csv(self, filepath: str):
        """Export metrics to CSV."""
        import csv
        
        if not self.metrics:
            return
        
        fieldnames = ["timestamp", "tick", "category", "name", "value"]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.metrics)
        
        logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
    
    def export_json(self, filepath: str):
        """Export metrics to JSON."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
    
    def summary(self) -> Dict:
        """Generate summary statistics."""
        from collections import defaultdict
        import numpy as np
        
        by_name = defaultdict(list)
        for m in self.metrics:
            if isinstance(m['value'], (int, float)):
                by_name[f"{m['category']}.{m['name']}"].append(m['value'])
        
        summary = {}
        for name, values in by_name.items():
            arr = np.array(values)
            summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "count": len(values)
            }
        
        return summary


# Global metrics exporter
metrics = MetricsExporter()


# =============================================================================
# TERMINOLOGY MAPPING (Booklet ↔ Code alignment)
# =============================================================================

TERMINOLOGY_MAP = {
    # Booklet Term → Code Attribute
    "B_entropy": "entropy_budget",
    "Δ_Regen": "delta_regen", 
    "τ_FDR": "tau_fdr",
    "τ_BIM": "tau_bim",
    "S_ψ": "entropy_density",
    "Φ": "phi_vector",
    "A_s": "stability",
    "SCI": "symbolic_coherence_index",
    "FDR": "findings_discovery_rate",
    "TTF": "time_to_failure",
    "LKC": "last_known_configuration",
    "CDP": "context_decay_pruning",
    "AGM": "autonomous_growth_module",
    "RET": "regenerative_efficiency_test",
    "AES": "adaptive_energy_scaling",
    "BIM": "behavioral_integrity_matrix",
    "CSL": "capsule_synthesis_layer",
    "CAC": "cognitive_aperture_controller",
    "TPE": "temporal_projection_engine",
    "RPC": "reflective_projection_check",
    "EVC": "epistemic_vector_compass",
    "d-CTM": "distributed_cognitive_trust_matrix",
    "ZK-SP": "zero_knowledge_swarm_proof",
    "BCI": "biosphere_command_interface",
    "EVI": "environmental_volatility_index",
}


def get_code_name(booklet_term: str) -> str:
    """Convert booklet terminology to code attribute name."""
    return TERMINOLOGY_MAP.get(booklet_term, booklet_term.lower().replace("-", "_"))


# =============================================================================
# DEMO / VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EFM CONFIGURATION & INVARIANT SYSTEM")
    print("=" * 70)
    
    # Create and validate config
    config = EFMConfig()
    config.validate()
    
    print("\n📋 REGENERATIVE CONFIG:")
    print(f"   τ_FDR (tau_fdr): {config.regenerative.tau_fdr}")
    print(f"   τ_BIM (tau_bim): {config.regenerative.tau_bim}")
    print(f"   Initial Budget: {config.regenerative.initial_budget}")
    print(f"   Reserve Ratio: {config.regenerative.min_reserve_ratio}")
    
    print("\n🌳 FOREST CONFIG:")
    print(f"   kNN Neighbors: {config.forest.knn_neighbors}")
    print(f"   Max Missions: {config.forest.max_concurrent_missions}")
    print(f"   O(n²) Warning: n > {config.forest.n_squared_warning_threshold}")
    
    print("\n⚙️ PRODUCTION CONFIG:")
    print(f"   Embedding Dims: {config.production.embedding_dimensions}")
    print(f"   Quorum Ratio: {config.production.min_quorum_ratio}")
    print(f"   Byzantine Tolerance: {config.production.max_byzantine_ratio}")
    
    print("\n🔍 INVARIANT TESTS:")
    
    # Test invariant checking
    checker = InvariantChecker(strict=False)
    checker.check(CoreInvariants.budget_non_negative(100), "Budget positive")
    checker.check(CoreInvariants.budget_non_negative(-1), "Budget negative (should fail)")
    checker.check(CoreInvariants.confidence_bounded(0.5), "Confidence in range")
    checker.check(CoreInvariants.confidence_bounded(1.5), "Confidence out of range (should fail)")
    
    print(checker.report())
    
    print("\n📊 METRICS EXAMPLE:")
    metrics.record("regenerative", "delta_regen", 0.15, tick=100)
    metrics.record("regenerative", "delta_regen", 0.18, tick=101)
    metrics.record("forest", "missions_completed", 5, tick=100)
    
    print(f"   Recorded: {len(metrics.metrics)} metrics")
    summary = metrics.summary()
    for name, stats in summary.items():
        print(f"   {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    
    print("\n📖 TERMINOLOGY MAPPING:")
    for booklet, code in list(TERMINOLOGY_MAP.items())[:5]:
        print(f"   {booklet} → {code}")
    
    print("\n" + "=" * 70)
    print("✅ Configuration system ready")
    print("=" * 70)
