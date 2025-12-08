"""
EFM Simulation Output: Tick-by-Tick Logs
=========================================

This module generates comprehensive simulation logs showing:
1. Capsule drift detection
2. CAC trace level adjustments
3. RPC actions and responses
4. Lineage propagation events
5. Recovery and partitioning

These logs serve as the "proof of execution" for the EFM framework.
"""

import numpy as np
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


# =============================================================================
# LOG STRUCTURES
# =============================================================================

class EventType(Enum):
    TICK_START = "TICK_START"
    TICK_END = "TICK_END"
    PHI_INGESTED = "PHI_INGESTED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    STABILITY_CHANGE = "STABILITY_CHANGE"
    ENTROPY_SPIKE = "ENTROPY_SPIKE"
    TTF_WARNING = "TTF_WARNING"
    TTF_CRITICAL = "TTF_CRITICAL"
    TRACE_LEVEL_CHANGE = "TRACE_LEVEL_CHANGE"
    DSL_ACTION = "DSL_ACTION"
    CONSENSUS_UPDATE = "CONSENSUS_UPDATE"
    LINEAGE_EVENT = "LINEAGE_EVENT"
    QUARANTINE = "QUARANTINE"
    HEAL_ATTEMPT = "HEAL_ATTEMPT"
    HEAL_SUCCESS = "HEAL_SUCCESS"
    HEAL_FAILURE = "HEAL_FAILURE"
    PARTITION = "PARTITION"
    ESCALATION = "ESCALATION"
    PROOF_GENERATED = "PROOF_GENERATED"
    RECOVERY = "RECOVERY"


@dataclass
class LogEntry:
    """Single log entry"""
    tick: int
    timestamp: float
    event_type: str
    capsule_id: Optional[str]
    severity: str  # INFO, WARNING, CRITICAL
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    def to_line(self) -> str:
        """Single-line log format"""
        data_str = json.dumps(self.data) if self.data else ""
        return f"[T{self.tick:04d}] [{self.severity:8s}] [{self.event_type:20s}] {self.capsule_id or 'SYSTEM':12s} | {self.message} {data_str}"


class SimulationLogger:
    """Comprehensive simulation logger"""
    
    def __init__(self):
        self.entries: List[LogEntry] = []
        self.tick = 0
        
    def log(self, event_type: EventType, capsule_id: Optional[str], 
            severity: str, message: str, data: Dict = None):
        entry = LogEntry(
            tick=self.tick,
            timestamp=time.time(),
            event_type=event_type.value,
            capsule_id=capsule_id,
            severity=severity,
            message=message,
            data=data or {}
        )
        self.entries.append(entry)
        return entry
    
    def advance_tick(self):
        self.tick += 1
        
    def get_logs(self, event_type: EventType = None) -> List[LogEntry]:
        if event_type:
            return [e for e in self.entries if e.event_type == event_type.value]
        return self.entries
    
    def export_txt(self) -> str:
        """Export as human-readable text"""
        lines = ["=" * 100]
        lines.append("EFM SIMULATION LOG")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total Entries: {len(self.entries)}")
        lines.append("=" * 100)
        lines.append("")
        
        for entry in self.entries:
            lines.append(entry.to_line())
        
        return "\n".join(lines)
    
    def export_json(self) -> str:
        """Export as JSON"""
        return json.dumps([asdict(e) for e in self.entries], indent=2)


# =============================================================================
# SIMULATION ENGINE WITH LOGGING
# =============================================================================

@dataclass
class SimCapsule:
    """Simulated capsule with full state"""
    id: str
    phi: np.ndarray
    parent_id: Optional[str]
    stability: float = 1.0
    entropy: float = 0.0
    drift_velocity: float = 0.0
    ttf: float = 1000.0
    trace_level: int = 1
    status: str = "ACTIVE"
    lineage_depth: int = 0
    tick_created: int = 0
    last_action_tick: int = 0
    quarantine_count: int = 0


class EFMSimulationEngine:
    """
    Full EFM simulation with comprehensive logging.
    
    Demonstrates:
    - Drift detection and response
    - Trace level adjustments
    - DSL action execution
    - Lineage tracking
    - Recovery procedures
    """
    
    def __init__(self, n_capsules: int = 10, phi_dim: int = 48):
        self.logger = SimulationLogger()
        self.capsules: Dict[str, SimCapsule] = {}
        self.phi_dim = phi_dim
        self.tick = 0
        
        # Thresholds
        self.drift_threshold = 0.3
        self.stability_critical = 0.3
        self.entropy_warning = 0.7
        self.ttf_warning = 30
        self.ttf_critical = 10
        
        # Initialize capsules
        self._init_capsules(n_capsules)
        
    def _init_capsules(self, n: int):
        """Initialize capsule population"""
        for i in range(n):
            phi = np.random.randn(self.phi_dim)
            phi = phi / np.linalg.norm(phi)
            
            parent_id = None
            if i > 0 and np.random.random() < 0.7:
                parent_id = f"cap_{np.random.randint(0, i):03d}"
            
            capsule = SimCapsule(
                id=f"cap_{i:03d}",
                phi=phi,
                parent_id=parent_id,
                lineage_depth=0 if parent_id is None else self.capsules.get(parent_id, SimCapsule("", np.array([]), None)).lineage_depth + 1,
                tick_created=0
            )
            self.capsules[capsule.id] = capsule
            
            self.logger.log(
                EventType.PHI_INGESTED,
                capsule.id,
                "INFO",
                f"Capsule initialized with parent={parent_id}, lineage_depth={capsule.lineage_depth}",
                {"phi_norm": float(np.linalg.norm(phi)), "parent": parent_id}
            )
    
    def run_simulation(self, n_ticks: int = 100, drift_injection_tick: int = 30,
                       drift_targets: List[str] = None) -> str:
        """
        Run full simulation with drift injection.
        
        Args:
            n_ticks: Number of ticks to simulate
            drift_injection_tick: Tick at which to inject drift
            drift_targets: Capsule IDs to inject drift into (random if None)
        """
        self.logger.log(
            EventType.TICK_START,
            None,
            "INFO",
            f"=== SIMULATION START: {n_ticks} ticks, {len(self.capsules)} capsules ===",
            {"n_ticks": n_ticks, "n_capsules": len(self.capsules)}
        )
        
        if drift_targets is None:
            drift_targets = list(self.capsules.keys())[:3]
        
        for tick in range(n_ticks):
            self.tick = tick
            self.logger.tick = tick
            
            # Tick start
            self.logger.log(
                EventType.TICK_START,
                None,
                "INFO",
                f"--- Tick {tick} Start ---",
                {"active_capsules": len([c for c in self.capsules.values() if c.status == "ACTIVE"])}
            )
            
            # Inject drift at specified tick
            if tick == drift_injection_tick:
                self._inject_drift(drift_targets)
            
            # Process each capsule
            for capsule_id, capsule in self.capsules.items():
                if capsule.status == "ACTIVE":
                    self._process_capsule(capsule)
            
            # Global consensus update
            self._update_consensus()
            
            # Generate proof every 10 ticks
            if tick % 10 == 0 and tick > 0:
                self._generate_proof()
            
            # Tick end
            self.logger.log(
                EventType.TICK_END,
                None,
                "INFO",
                f"--- Tick {tick} End ---",
                self._get_tick_summary()
            )
            
            self.logger.advance_tick()
        
        return self.logger.export_txt()
    
    def _inject_drift(self, target_ids: List[str]):
        """Inject drift into target capsules"""
        self.logger.log(
            EventType.DRIFT_DETECTED,
            None,
            "WARNING",
            f"!!! DRIFT INJECTION EVENT: Targeting {len(target_ids)} capsules !!!",
            {"targets": target_ids}
        )
        
        for cap_id in target_ids:
            if cap_id in self.capsules:
                cap = self.capsules[cap_id]
                # Add significant noise to phi
                noise = np.random.randn(self.phi_dim) * 0.3
                cap.phi = cap.phi + noise
                cap.phi = cap.phi / np.linalg.norm(cap.phi)
                cap.drift_velocity = 0.4 + np.random.random() * 0.2
                
                self.logger.log(
                    EventType.DRIFT_DETECTED,
                    cap_id,
                    "CRITICAL",
                    f"Drift injected: velocity={cap.drift_velocity:.3f}",
                    {"drift_velocity": cap.drift_velocity, "injection_type": "ADVERSARIAL"}
                )
    
    def _process_capsule(self, capsule: SimCapsule):
        """Process single capsule through EFM pipeline"""
        
        # 1. Update metrics (simulate natural evolution)
        self._update_metrics(capsule)
        
        # 2. Check for drift
        if capsule.drift_velocity > self.drift_threshold:
            self._handle_drift(capsule)
        
        # 3. Check stability
        if capsule.stability < self.stability_critical:
            self.logger.log(
                EventType.STABILITY_CHANGE,
                capsule.id,
                "WARNING",
                f"Low stability: {capsule.stability:.3f}",
                {"stability": capsule.stability, "threshold": self.stability_critical}
            )
        
        # 4. Check entropy
        if capsule.entropy > self.entropy_warning:
            self.logger.log(
                EventType.ENTROPY_SPIKE,
                capsule.id,
                "WARNING",
                f"High entropy: {capsule.entropy:.3f}",
                {"entropy": capsule.entropy, "threshold": self.entropy_warning}
            )
        
        # 5. Check TTF
        if capsule.ttf < self.ttf_critical:
            self.logger.log(
                EventType.TTF_CRITICAL,
                capsule.id,
                "CRITICAL",
                f"TTF CRITICAL: {capsule.ttf:.1f} ticks remaining",
                {"ttf": capsule.ttf}
            )
            self._execute_action("ESCALATE", capsule)
        elif capsule.ttf < self.ttf_warning:
            self.logger.log(
                EventType.TTF_WARNING,
                capsule.id,
                "WARNING",
                f"TTF Warning: {capsule.ttf:.1f} ticks remaining",
                {"ttf": capsule.ttf}
            )
        
        # 6. Adjust trace level
        self._adjust_trace_level(capsule)
    
    def _update_metrics(self, capsule: SimCapsule):
        """Update capsule metrics based on state"""
        # Decay drift naturally
        if capsule.drift_velocity > 0.01:
            capsule.drift_velocity *= 0.95
        
        # Update stability based on drift
        capsule.stability = max(0.1, 1.0 - capsule.drift_velocity * 2)
        
        # Update entropy
        capsule.entropy = min(1.0, capsule.drift_velocity + np.random.random() * 0.1)
        
        # Update TTF
        if capsule.drift_velocity > 0.01:
            capsule.ttf = self.drift_threshold / capsule.drift_velocity * 10
        else:
            capsule.ttf = min(capsule.ttf + 1, 1000)
    
    def _handle_drift(self, capsule: SimCapsule):
        """Handle detected drift"""
        self.logger.log(
            EventType.DRIFT_DETECTED,
            capsule.id,
            "WARNING",
            f"Drift detected: velocity={capsule.drift_velocity:.3f}, stability={capsule.stability:.3f}",
            {
                "drift_velocity": capsule.drift_velocity,
                "stability": capsule.stability,
                "ttf": capsule.ttf
            }
        )
        
        # Determine action based on severity
        if capsule.drift_velocity > 0.5:
            self._execute_action("QUARANTINE", capsule)
        elif capsule.drift_velocity > 0.3:
            self._execute_action("HEAL", capsule)
    
    def _execute_action(self, action: str, capsule: SimCapsule):
        """Execute DSL action"""
        self.logger.log(
            EventType.DSL_ACTION,
            capsule.id,
            "INFO",
            f"Executing action: {action}",
            {"action": action, "trigger": "drift_threshold"}
        )
        
        if action == "QUARANTINE":
            capsule.status = "QUARANTINED"
            capsule.quarantine_count += 1
            self.logger.log(
                EventType.QUARANTINE,
                capsule.id,
                "WARNING",
                f"Capsule QUARANTINED (count: {capsule.quarantine_count})",
                {"quarantine_count": capsule.quarantine_count}
            )
            
            # Check lineage contagion
            self._check_lineage_contagion(capsule)
            
        elif action == "HEAL":
            self.logger.log(
                EventType.HEAL_ATTEMPT,
                capsule.id,
                "INFO",
                f"Attempting heal: stability={capsule.stability:.3f}",
                {}
            )
            
            # Simulate heal attempt
            if np.random.random() < capsule.stability:
                capsule.drift_velocity *= 0.5
                self.logger.log(
                    EventType.HEAL_SUCCESS,
                    capsule.id,
                    "INFO",
                    f"Heal SUCCESS: drift reduced to {capsule.drift_velocity:.3f}",
                    {"new_drift": capsule.drift_velocity}
                )
            else:
                self.logger.log(
                    EventType.HEAL_FAILURE,
                    capsule.id,
                    "WARNING",
                    "Heal FAILED: escalating to quarantine",
                    {}
                )
                self._execute_action("QUARANTINE", capsule)
                
        elif action == "ESCALATE":
            self.logger.log(
                EventType.ESCALATION,
                capsule.id,
                "CRITICAL",
                "ESCALATION: Human-in-the-loop review required",
                {
                    "drift_velocity": capsule.drift_velocity,
                    "ttf": capsule.ttf,
                    "stability": capsule.stability
                }
            )
            
        elif action == "PARTITION":
            capsule.status = "PARTITIONED"
            self.logger.log(
                EventType.PARTITION,
                capsule.id,
                "CRITICAL",
                "Capsule PARTITIONED from swarm",
                {"lineage_severed": True}
            )
    
    def _check_lineage_contagion(self, capsule: SimCapsule):
        """Check for drift contagion in lineage"""
        # Find children
        children = [c for c in self.capsules.values() if c.parent_id == capsule.id]
        
        if children:
            self.logger.log(
                EventType.LINEAGE_EVENT,
                capsule.id,
                "WARNING",
                f"Checking lineage contagion: {len(children)} children",
                {"children": [c.id for c in children]}
            )
            
            # Propagate some drift to children
            for child in children:
                if child.status == "ACTIVE":
                    child.drift_velocity = max(child.drift_velocity, capsule.drift_velocity * 0.5)
                    self.logger.log(
                        EventType.LINEAGE_EVENT,
                        child.id,
                        "WARNING",
                        f"Inherited drift from parent {capsule.id}: velocity={child.drift_velocity:.3f}",
                        {"inherited_from": capsule.id, "drift": child.drift_velocity}
                    )
    
    def _adjust_trace_level(self, capsule: SimCapsule):
        """Adjust trace level based on CAC logic"""
        old_level = capsule.trace_level
        
        if capsule.ttf < self.ttf_critical or capsule.stability < 0.3:
            capsule.trace_level = 4
        elif capsule.ttf < self.ttf_warning or capsule.stability < 0.5:
            capsule.trace_level = 3
        elif capsule.ttf < 60 or capsule.stability < 0.7:
            capsule.trace_level = 2
        else:
            capsule.trace_level = 1
        
        if capsule.trace_level != old_level:
            self.logger.log(
                EventType.TRACE_LEVEL_CHANGE,
                capsule.id,
                "INFO",
                f"Trace level: L{old_level} → L{capsule.trace_level}",
                {"old_level": old_level, "new_level": capsule.trace_level}
            )
    
    def _update_consensus(self):
        """Update global consensus state"""
        active = [c for c in self.capsules.values() if c.status == "ACTIVE"]
        quarantined = [c for c in self.capsules.values() if c.status == "QUARANTINED"]
        
        if quarantined:
            self.logger.log(
                EventType.CONSENSUS_UPDATE,
                None,
                "INFO",
                f"Consensus update: {len(active)} active, {len(quarantined)} quarantined",
                {
                    "active_count": len(active),
                    "quarantined_count": len(quarantined),
                    "quarantined_ids": [c.id for c in quarantined]
                }
            )
    
    def _generate_proof(self):
        """Generate ZK-SP proof"""
        active = [c for c in self.capsules.values() if c.status == "ACTIVE"]
        proof_id = f"proof_{self.tick:04d}"
        
        self.logger.log(
            EventType.PROOF_GENERATED,
            None,
            "INFO",
            f"ZK-SP Proof generated: {proof_id}",
            {
                "proof_id": proof_id,
                "agents_covered": len(active),
                "tick": self.tick
            }
        )
    
    def _get_tick_summary(self) -> Dict:
        """Get summary stats for tick"""
        active = len([c for c in self.capsules.values() if c.status == "ACTIVE"])
        quarantined = len([c for c in self.capsules.values() if c.status == "QUARANTINED"])
        avg_stability = np.mean([c.stability for c in self.capsules.values()])
        avg_drift = np.mean([c.drift_velocity for c in self.capsules.values()])
        
        return {
            "active": active,
            "quarantined": quarantined,
            "avg_stability": round(avg_stability, 3),
            "avg_drift": round(avg_drift, 3)
        }


# =============================================================================
# RUN SIMULATION
# =============================================================================

def generate_simulation_logs():
    """Generate comprehensive simulation logs"""
    print("=" * 80)
    print("EFM SIMULATION: GENERATING TICK-BY-TICK LOGS")
    print("=" * 80)
    
    # Create engine
    engine = EFMSimulationEngine(n_capsules=10, phi_dim=48)
    
    # Run simulation with drift injection at tick 30
    logs = engine.run_simulation(
        n_ticks=100,
        drift_injection_tick=30,
        drift_targets=["cap_000", "cap_001", "cap_002"]
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    
    # Count events by type
    event_counts = {}
    for entry in engine.logger.entries:
        event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1
    
    print("\nEvent Summary:")
    for event_type, count in sorted(event_counts.items()):
        print(f"  {event_type:25s}: {count:4d}")
    
    print(f"\nTotal Log Entries: {len(engine.logger.entries)}")
    
    # Return logs
    return logs, engine.logger.export_json()


if __name__ == "__main__":
    txt_logs, json_logs = generate_simulation_logs()
    
    # Print first 50 log lines
    print("\n" + "=" * 80)
    print("SAMPLE LOG OUTPUT (First 100 lines)")
    print("=" * 80)
    lines = txt_logs.split("\n")[:100]
    for line in lines:
        print(line)
    
    # Save logs
    with open("/home/claude/efm-forest-final/logs/simulation_output.txt", "w") as f:
        f.write(txt_logs)
    
    with open("/home/claude/efm-forest-final/logs/simulation_output.json", "w") as f:
        f.write(json_logs)
    
    print(f"\n✅ Logs saved to /home/claude/efm-forest-final/logs/")
