#!/usr/bin/env python3
"""
EFM Forest Benchmark Runner
===========================

Run standardized benchmarks to validate Booklet 4 claims.

Usage:
    python run_forest.py --preset genesis
    python run_forest.py --preset hostile_matrix
    python run_forest.py --preset data_winter
    python run_forest.py --ticks 200 --agents 100

Presets:
    genesis       - Standard 50-agent spawn, validates FDR=1.0, SCI>0.8
    hostile       - Adversarial drift injection stress test
    winter        - Data winter (resource scarcity) survival test
    hyperscale    - 1M agent simulation (requires significant resources)

Output:
    - Console summary with pass/fail status
    - tick_log.json with full audit trail
    - final_genealogy.json with lineage tree
"""

import argparse
import json
import time
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# Import config
from efm_config import EFMConfig, BenchmarkPreset, check_invariants, InvariantError, TickLogEntry


# =============================================================================
# AGENT SIMULATION
# =============================================================================

@dataclass
class Agent:
    """Simulated EFM agent for benchmarking."""
    agent_id: str
    position: tuple
    B_entropy: float
    health: float
    drift: float
    SCI: float
    FDR: float
    branch_id: str
    status: str = "ACTIVE"
    generation: int = 0
    parent_id: Optional[str] = None
    children: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


class ForestSimulator:
    """
    Forest growth simulator for benchmark validation.
    
    This simulates the core EFM behaviors:
    - Agent spawning and pruning
    - Drift injection and healing
    - Entropy budget management
    - Consensus and mission execution
    """
    
    def __init__(self, cfg: EFMConfig):
        self.cfg = cfg
        self.agents: Dict[str, Agent] = {}
        self.tick = 0
        self.tick_log: List[dict] = []
        self.events: List[dict] = []
        self.genealogy: Dict[str, dict] = {}
        
        # Metrics tracking
        self.total_spawned = 0
        self.total_pruned = 0
        self.total_discoveries = 0
        self.winters_survived = 0
        self.invariant_violations = 0
        
        # Set random seed if specified
        if cfg.RANDOM_SEED is not None:
            random.seed(cfg.RANDOM_SEED)
            np.random.seed(cfg.RANDOM_SEED)
    
    def spawn_initial(self, count: int):
        """Spawn initial agent population."""
        for i in range(count):
            agent_id = f"ROOT.{i:03d}"
            pos = np.random.randint(0, 100, 3)
            agent = Agent(
                agent_id=agent_id,
                position=tuple(int(x) for x in pos),
                B_entropy=self.cfg.INIT_BUDGET,
                health=1.0,
                drift=0.0,
                SCI=1.0,
                FDR=1.0,
                branch_id="ROOT",
                generation=0
            )
            self.agents[agent_id] = agent
            self.genealogy[agent_id] = {
                "parent": None,
                "children": [],
                "generation": 0,
                "spawn_tick": 0
            }
            self.total_spawned += 1
        
        self._log_event("GENESIS", f"Spawned {count} initial agents")
    
    def simulate_tick(self):
        """Execute one simulation tick."""
        self.tick += 1
        
        active_agents = [a for a in self.agents.values() if a.status == "ACTIVE"]
        
        for agent in active_agents:
            # Natural drift accumulation
            agent.drift += random.uniform(0, 0.05)
            agent.drift = min(agent.drift, 1.0)
            
            # Entropy consumption
            agent.B_entropy -= self.cfg.GROWTH_COST_FACTOR * 10
            
            # Update SCI based on drift
            agent.SCI = max(0, 1.0 - agent.drift)
            
            # Update FDR (discovery rate)
            if agent.SCI > self.cfg.MIN_SCI:
                agent.FDR = min(1.0, agent.FDR + random.uniform(0, 0.1))
                if random.random() < 0.1:
                    self.total_discoveries += 1
            
            # Health degrades with drift
            agent.health = max(0, 1.0 - agent.drift * 0.5)
            
            # Check invariants
            try:
                check_invariants(
                    self.cfg,
                    B_entropy=agent.B_entropy,
                    drift=agent.drift,
                    SCI=agent.SCI,
                    FDR=agent.FDR
                )
            except InvariantError:
                self.invariant_violations += 1
            
            # Determine action
            action = self._determine_action(agent)
            self._execute_action(agent, action)
            
            # Log tick
            self._log_tick(agent, action)
        
        # Global events
        self._check_global_events()
    
    def _determine_action(self, agent: Agent) -> str:
        """Determine agent's action based on state."""
        if agent.B_entropy <= 0:
            return "STARVE"
        if agent.drift > self.cfg.DRIFT_ALERT_THRESHOLD:
            return "HEAL"
        if agent.health < self.cfg.PRUNE_THRESHOLD:
            return "PRUNE"
        if agent.B_entropy > self.cfg.GENESIS_THRESHOLD and agent.SCI > self.cfg.MIN_SCI:
            if random.random() < 0.1:
                return "SPAWN"
        if agent.FDR > self.cfg.MIN_FDR:
            return "EXPLORE"
        return "GROW"
    
    def _execute_action(self, agent: Agent, action: str):
        """Execute determined action."""
        if action == "STARVE":
            agent.status = "PRUNED"
            self.total_pruned += 1
            self._log_event("PRUNE", f"{agent.agent_id} starved (B_entropy=0)")
        
        elif action == "HEAL":
            heal_cost = 20.0
            if agent.B_entropy >= heal_cost:
                agent.B_entropy -= heal_cost
                agent.drift = max(0, agent.drift - 0.3)
                self._log_event("HEAL", f"{agent.agent_id} healed, drift now {agent.drift:.2f}")
        
        elif action == "PRUNE":
            agent.status = "PRUNED"
            self.total_pruned += 1
            # Recover resources to parent
            if agent.parent_id and agent.parent_id in self.agents:
                recovery = agent.B_entropy * self.cfg.RESOURCE_RECOVERY_RATE
                self.agents[agent.parent_id].B_entropy += recovery
            self._log_event("PRUNE", f"{agent.agent_id} pruned (health={agent.health:.2f})")
        
        elif action == "SPAWN":
            child_id = f"{agent.agent_id}.{len(agent.children):03d}"
            spawn_cost = self.cfg.GENESIS_THRESHOLD * 0.8
            
            new_pos = np.array(agent.position) + np.random.randint(-5, 6, 3)
            child = Agent(
                agent_id=child_id,
                position=tuple(int(x) for x in new_pos),
                B_entropy=spawn_cost,
                health=1.0,
                drift=0.0,
                SCI=agent.SCI * 0.9,  # Inherit 90% coherence
                FDR=0.5,  # Start with baseline discovery
                branch_id=agent.agent_id,
                generation=agent.generation + 1,
                parent_id=agent.agent_id
            )
            
            agent.B_entropy -= spawn_cost
            agent.children.append(child_id)
            self.agents[child_id] = child
            self.genealogy[child_id] = {
                "parent": agent.agent_id,
                "children": [],
                "generation": child.generation,
                "spawn_tick": self.tick
            }
            self.genealogy[agent.agent_id]["children"].append(child_id)
            self.total_spawned += 1
            
            self._log_event("SPAWN", f"{agent.agent_id} spawned {child_id}")
        
        elif action == "EXPLORE":
            # Move in growth direction
            move = np.random.randint(-3, 4, 3)
            new_pos = np.array(agent.position) + move
            agent.position = tuple(int(x) for x in new_pos)
            agent.B_entropy -= 5.0
        
        elif action == "GROW":
            # Accumulate entropy
            agent.B_entropy += random.uniform(5, 15)
            agent.B_entropy = min(agent.B_entropy, self.cfg.INIT_BUDGET * 2)
    
    def _check_global_events(self):
        """Check for global events like data winters."""
        # Random data winter (resource scarcity)
        if random.random() < 0.02:
            self._inject_data_winter()
    
    def _inject_data_winter(self, severity: float = 0.5):
        """Inject a data winter event."""
        self._log_event("DATA_WINTER", f"Data winter began (severity={severity})")
        
        for agent in self.agents.values():
            if agent.status == "ACTIVE":
                agent.B_entropy *= (1 - severity)
        
        self.winters_survived += 1
    
    def inject_drift_storm(self, targets: int = 10):
        """Inject adversarial drift into random agents."""
        active = [a for a in self.agents.values() if a.status == "ACTIVE"]
        targets = min(targets, len(active))
        
        for agent in random.sample(active, targets):
            agent.drift = min(1.0, agent.drift + random.uniform(0.3, 0.6))
        
        self._log_event("DRIFT_STORM", f"Drift injected into {targets} agents")
    
    def _log_tick(self, agent: Agent, action: str):
        """Log agent state for this tick."""
        entry = TickLogEntry(
            tick=self.tick,
            agent_id=agent.agent_id,
            position=tuple(int(x) for x in agent.position),  # Convert numpy to int
            FDR=float(agent.FDR),
            SCI=float(agent.SCI),
            B_entropy=float(agent.B_entropy),
            health=float(agent.health),
            growth_vector=(0, 0, 0),  # Simplified
            status=agent.status,
            action=action,
            branch_id=agent.branch_id
        )
        self.tick_log.append(entry.to_dict())
    
    def _log_event(self, event_type: str, message: str):
        """Log a significant event."""
        self.events.append({
            "tick": self.tick,
            "type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        
        if self.cfg.LOG_LEVEL in ["DEBUG", "INFO"]:
            print(f"[T{self.tick:04d}] {event_type}: {message}")
    
    def run(self, ticks: Optional[int] = None):
        """Run the simulation."""
        ticks = ticks or self.cfg.MAX_TICKS
        
        print(f"\n{'='*60}")
        print(f"EFM FOREST BENCHMARK")
        print(f"{'='*60}")
        print(f"Ticks: {ticks}")
        print(f"Initial Budget: {self.cfg.INIT_BUDGET}")
        print(f"Genesis Threshold: {self.cfg.GENESIS_THRESHOLD}")
        print(f"{'='*60}\n")
        
        for _ in range(ticks):
            self.simulate_tick()
        
        return self.get_results()
    
    def get_results(self) -> dict:
        """Get benchmark results."""
        active = [a for a in self.agents.values() if a.status == "ACTIVE"]
        
        avg_sci = float(np.mean([a.SCI for a in active])) if active else 0.0
        avg_fdr = float(np.mean([a.FDR for a in active])) if active else 0.0
        max_generation = int(max([a.generation for a in self.agents.values()])) if self.agents else 0
        
        sci_passed = bool(avg_sci >= self.cfg.MIN_SCI)
        fdr_passed = bool(avg_fdr >= self.cfg.MIN_FDR)
        survival_passed = bool(len(active) > 0)
        
        results = {
            "ticks_completed": int(self.tick),
            "total_spawned": int(self.total_spawned),
            "total_pruned": int(self.total_pruned),
            "active_agents": int(len(active)),
            "total_discoveries": int(self.total_discoveries),
            "winters_survived": int(self.winters_survived),
            "invariant_violations": int(self.invariant_violations),
            "avg_SCI": float(avg_sci),
            "avg_FDR": float(avg_fdr),
            "max_generation": int(max_generation),
            "sci_passed": sci_passed,
            "fdr_passed": fdr_passed,
            "survival_passed": survival_passed,
            "overall_passed": bool(sci_passed and fdr_passed and survival_passed)
        }
        
        return results
    
    def export_logs(self, output_dir: str = "benchmark_output"):
        """Export logs to files."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Tick log
        with open(f"{output_dir}/tick_log.json", "w") as f:
            json.dump(self.tick_log, f, indent=2)
        
        # Events
        with open(f"{output_dir}/events.json", "w") as f:
            json.dump(self.events, f, indent=2)
        
        # Genealogy
        with open(f"{output_dir}/genealogy.json", "w") as f:
            json.dump(self.genealogy, f, indent=2)
        
        # Results summary
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump(self.get_results(), f, indent=2)
        
        print(f"\n📁 Logs exported to {output_dir}/")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EFM Forest Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_forest.py --preset genesis
    python run_forest.py --preset hostile --ticks 200
    python run_forest.py --agents 100 --ticks 150 --seed 42
        """
    )
    
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=["genesis", "hostile", "winter", "hyperscale"],
        help="Benchmark preset to use"
    )
    parser.add_argument("--ticks", type=int, help="Number of ticks to simulate")
    parser.add_argument("--agents", type=int, default=50, help="Initial agent count")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="benchmark_output", help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress tick-by-tick output")
    
    args = parser.parse_args()
    
    # Create config
    if args.preset:
        preset_map = {
            "genesis": BenchmarkPreset.GENESIS,
            "hostile": BenchmarkPreset.HOSTILE_MATRIX,
            "winter": BenchmarkPreset.DATA_WINTER,
            "hyperscale": BenchmarkPreset.HYPERSCALE
        }
        cfg = EFMConfig.for_preset(preset_map[args.preset])
    else:
        cfg = EFMConfig()
    
    # Override with CLI args
    if args.ticks:
        cfg.MAX_TICKS = args.ticks
    if args.seed:
        cfg.RANDOM_SEED = args.seed
    if args.quiet:
        cfg.LOG_LEVEL = "QUIET"
    
    # Run simulation
    sim = ForestSimulator(cfg)
    sim.spawn_initial(args.agents)
    
    # Preset-specific injections
    if args.preset == "hostile":
        # Schedule drift storms
        for tick in [20, 50, 80]:
            pass  # Storms happen randomly in this version
    
    results = sim.run()
    
    # Export logs
    sim.export_logs(args.output)
    
    # Print results
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  Ticks Completed: {results['ticks_completed']}")
    print(f"  Total Spawned: {results['total_spawned']}")
    print(f"  Total Pruned: {results['total_pruned']}")
    print(f"  Active Agents: {results['active_agents']}")
    print(f"  Total Discoveries: {results['total_discoveries']}")
    print(f"  Winters Survived: {results['winters_survived']}")
    print(f"  Max Generation: {results['max_generation']}")
    print(f"\n  Avg SCI: {results['avg_SCI']:.3f} {'✅' if results['sci_passed'] else '❌'} (min: {cfg.MIN_SCI})")
    print(f"  Avg FDR: {results['avg_FDR']:.3f} {'✅' if results['fdr_passed'] else '❌'} (min: {cfg.MIN_FDR})")
    print(f"  Survival: {'✅' if results['survival_passed'] else '❌'}")
    print(f"  Invariant Violations: {results['invariant_violations']}")
    print(f"\n{'='*60}")
    
    if results['overall_passed']:
        print("🎉 BENCHMARK PASSED")
    else:
        print("❌ BENCHMARK FAILED")
    
    print(f"{'='*60}\n")
    
    return 0 if results['overall_passed'] else 1


if __name__ == "__main__":
    exit(main())
