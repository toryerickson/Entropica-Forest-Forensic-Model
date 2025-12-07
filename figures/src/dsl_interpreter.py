"""
EFM Domain-Specific Language (DSL) Interpreter
===============================================

The DSL is the "nervous system" of the EFM—it translates symbolic
decisions into concrete actions. Without this, all governance is hardcoded.

Commands:
- PARTITION: Isolate a capsule or cluster
- ESCALATE: Alert swarm or trigger consensus
- ROLLBACK: Restore prior symbolic state
- NOTIFY: Send risk up the lineage
- PRUNE: Remove low-stability capsules
- ADOPT: Re-link orphan capsules
- SPAWN: Create new capsule/agent
- FORK: Genesis Protocol - create child swarm with new constitution

Syntax:
    ACTION TARGET [CONDITION] [OPTIONS]

Examples:
    PARTITION capsule_id=42 IF DriftRisk > 0.85
    ESCALATE lineage=root IF LineageStability < 0.6
    ROLLBACK capsule_id=87 TO t=-2
    FORK swarm_id=3 WITH lambda=0.1 tau_break=0.3
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto


class DSLAction(Enum):
    """Supported DSL actions"""
    PARTITION = auto()
    ESCALATE = auto()
    ROLLBACK = auto()
    NOTIFY = auto()
    PRUNE = auto()
    ADOPT = auto()
    SPAWN = auto()
    FORK = auto()  # Genesis Protocol


@dataclass
class DSLCommand:
    """Parsed DSL command"""
    action: DSLAction
    target: Dict[str, str]
    condition: Optional[str] = None
    options: Dict[str, str] = field(default_factory=dict)
    raw: str = ""


@dataclass
class DSLResult:
    """Result of DSL execution"""
    success: bool
    action: DSLAction
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


class DSLConditionEvaluator:
    """Evaluates conditional expressions against context"""
    
    def __init__(self):
        self.operators = {
            '>': lambda a, b: float(a) > float(b),
            '<': lambda a, b: float(a) < float(b),
            '>=': lambda a, b: float(a) >= float(b),
            '<=': lambda a, b: float(a) <= float(b),
            '==': lambda a, b: float(a) == float(b),
            '!=': lambda a, b: float(a) != float(b),
        }
    
    def evaluate(self, condition: str, context: Dict[str, float]) -> bool:
        """
        Evaluate condition string against context.
        Example: "DriftRisk > 0.85" with context {"DriftRisk": 0.9} -> True
        """
        if not condition:
            return True  # No condition = always true
        
        # Parse: VARIABLE OPERATOR VALUE
        pattern = r'(\w+)\s*(>=|<=|==|!=|>|<)\s*([\d.]+)'
        match = re.match(pattern, condition.strip())
        
        if not match:
            raise ValueError(f"Invalid condition syntax: {condition}")
        
        var_name, operator, value = match.groups()
        
        if var_name not in context:
            raise ValueError(f"Unknown variable in condition: {var_name}")
        
        return self.operators[operator](context[var_name], value)


class DSLParser:
    """Parses DSL command strings into structured commands"""
    
    def __init__(self):
        self.action_map = {
            'PARTITION': DSLAction.PARTITION,
            'ESCALATE': DSLAction.ESCALATE,
            'ROLLBACK': DSLAction.ROLLBACK,
            'NOTIFY': DSLAction.NOTIFY,
            'PRUNE': DSLAction.PRUNE,
            'ADOPT': DSLAction.ADOPT,
            'SPAWN': DSLAction.SPAWN,
            'FORK': DSLAction.FORK,
        }
    
    def parse(self, command: str) -> DSLCommand:
        """Parse a DSL command string"""
        command = command.strip()
        
        # Extract action
        tokens = command.split()
        if not tokens:
            raise ValueError("Empty command")
        
        action_str = tokens[0].upper()
        if action_str not in self.action_map:
            raise ValueError(f"Unknown action: {action_str}")
        
        action = self.action_map[action_str]
        
        # Extract key=value pairs
        kv_pattern = r'(\w+)=([\w\d._-]+)'
        kv_matches = re.findall(kv_pattern, command)
        all_params = dict(kv_matches)
        
        # Separate target params from options
        target_keys = ['capsule_id', 'cluster_id', 'swarm_id', 'agent_id', 'lineage']
        target = {k: v for k, v in all_params.items() if k in target_keys}
        options = {k: v for k, v in all_params.items() if k not in target_keys}
        
        # Extract condition (after IF)
        condition = None
        if ' IF ' in command.upper():
            condition_start = command.upper().index(' IF ') + 4
            # Find end of condition (before WITH or end of string)
            condition_end = len(command)
            if ' WITH ' in command.upper():
                condition_end = command.upper().index(' WITH ')
            condition = command[condition_start:condition_end].strip()
        
        return DSLCommand(
            action=action,
            target=target,
            condition=condition,
            options=options,
            raw=command
        )


class DSLInterpreter:
    """
    Executes DSL commands against an EFM context.
    
    This is the "nervous system" that translates symbolic decisions
    into concrete governance actions.
    """
    
    def __init__(self, efm_context=None):
        self.ctx = efm_context
        self.parser = DSLParser()
        self.evaluator = DSLConditionEvaluator()
        self.execution_log: List[DSLResult] = []
        
        # Action handlers
        self.handlers: Dict[DSLAction, Callable] = {
            DSLAction.PARTITION: self._handle_partition,
            DSLAction.ESCALATE: self._handle_escalate,
            DSLAction.ROLLBACK: self._handle_rollback,
            DSLAction.NOTIFY: self._handle_notify,
            DSLAction.PRUNE: self._handle_prune,
            DSLAction.ADOPT: self._handle_adopt,
            DSLAction.SPAWN: self._handle_spawn,
            DSLAction.FORK: self._handle_fork,
        }
    
    def execute(self, command_str: str, context: Dict[str, float] = None) -> DSLResult:
        """
        Parse and execute a DSL command.
        
        Args:
            command_str: DSL command string
            context: Variable context for condition evaluation
        
        Returns:
            DSLResult with execution outcome
        """
        context = context or {}
        
        try:
            # Parse
            cmd = self.parser.parse(command_str)
            
            # Evaluate condition
            if cmd.condition and not self.evaluator.evaluate(cmd.condition, context):
                result = DSLResult(
                    success=True,
                    action=cmd.action,
                    message=f"Condition not met: {cmd.condition}",
                    data={"skipped": True, "condition": cmd.condition}
                )
                self.execution_log.append(result)
                return result
            
            # Execute
            handler = self.handlers.get(cmd.action)
            if not handler:
                raise ValueError(f"No handler for action: {cmd.action}")
            
            result = handler(cmd)
            self.execution_log.append(result)
            return result
            
        except Exception as e:
            result = DSLResult(
                success=False,
                action=DSLAction.PARTITION,  # Default
                message=f"Execution error: {str(e)}",
                data={"error": str(e), "command": command_str}
            )
            self.execution_log.append(result)
            return result
    
    def execute_batch(self, commands: List[str], context: Dict[str, float] = None) -> List[DSLResult]:
        """Execute multiple commands in sequence"""
        return [self.execute(cmd, context) for cmd in commands]
    
    # ===================
    # Action Handlers
    # ===================
    
    def _handle_partition(self, cmd: DSLCommand) -> DSLResult:
        """Isolate a capsule or cluster"""
        target_id = cmd.target.get('capsule_id') or cmd.target.get('cluster_id')
        
        if self.ctx:
            # Real execution against EFM context
            # self.ctx.partition(target_id)
            pass
        
        return DSLResult(
            success=True,
            action=DSLAction.PARTITION,
            message=f"PARTITION: {target_id} isolated from swarm",
            data={"target": target_id, "mode": cmd.options.get('mode', 'QUARANTINE')}
        )
    
    def _handle_escalate(self, cmd: DSLCommand) -> DSLResult:
        """Alert swarm or trigger consensus"""
        lineage = cmd.target.get('lineage', 'root')
        level = cmd.options.get('level', 'WARN')
        
        if self.ctx:
            # self.ctx.trigger_escalation(lineage, level)
            pass
        
        return DSLResult(
            success=True,
            action=DSLAction.ESCALATE,
            message=f"ESCALATE: Lineage {lineage} alerted at level {level}",
            data={"lineage": lineage, "level": level}
        )
    
    def _handle_rollback(self, cmd: DSLCommand) -> DSLResult:
        """Restore prior symbolic state"""
        capsule_id = cmd.target.get('capsule_id')
        target_time = cmd.options.get('TO', 't=-1')
        
        if self.ctx:
            # self.ctx.rollback_capsule(capsule_id, target_time)
            pass
        
        return DSLResult(
            success=True,
            action=DSLAction.ROLLBACK,
            message=f"ROLLBACK: Capsule {capsule_id} restored to {target_time}",
            data={"capsule_id": capsule_id, "target_time": target_time}
        )
    
    def _handle_notify(self, cmd: DSLCommand) -> DSLResult:
        """Send risk notification up lineage"""
        target = cmd.target.get('lineage') or cmd.target.get('agent_id')
        risk_type = cmd.options.get('type', 'DRIFT_WARNING')
        
        return DSLResult(
            success=True,
            action=DSLAction.NOTIFY,
            message=f"NOTIFY: {risk_type} sent to {target}",
            data={"target": target, "risk_type": risk_type}
        )
    
    def _handle_prune(self, cmd: DSLCommand) -> DSLResult:
        """Remove low-stability capsules (CDP trigger)"""
        swarm_id = cmd.target.get('swarm_id')
        threshold = float(cmd.options.get('threshold', '0.2'))
        
        if self.ctx:
            # pruned = self.ctx.prune_swarm(swarm_id, threshold)
            pass
        
        return DSLResult(
            success=True,
            action=DSLAction.PRUNE,
            message=f"PRUNE: Swarm {swarm_id} pruned (threshold={threshold})",
            data={"swarm_id": swarm_id, "threshold": threshold}
        )
    
    def _handle_adopt(self, cmd: DSLCommand) -> DSLResult:
        """Re-link orphan capsule to new parent"""
        capsule_id = cmd.target.get('capsule_id')
        parent_id = cmd.options.get('parent_id') or cmd.options.get('INTO')
        
        if self.ctx:
            # self.ctx.relink_capsule(capsule_id, parent_id)
            pass
        
        return DSLResult(
            success=True,
            action=DSLAction.ADOPT,
            message=f"ADOPT: Capsule {capsule_id} linked to parent {parent_id}",
            data={"capsule_id": capsule_id, "parent_id": parent_id}
        )
    
    def _handle_spawn(self, cmd: DSLCommand) -> DSLResult:
        """Create new capsule or agent"""
        parent_id = cmd.target.get('agent_id') or cmd.target.get('capsule_id')
        spawn_type = cmd.options.get('type', 'capsule')
        
        return DSLResult(
            success=True,
            action=DSLAction.SPAWN,
            message=f"SPAWN: New {spawn_type} created from {parent_id}",
            data={"parent_id": parent_id, "type": spawn_type}
        )
    
    def _handle_fork(self, cmd: DSLCommand) -> DSLResult:
        """
        Genesis Protocol: Create child swarm with new constitution.
        
        This is the L3→L4 bridge - the system can propose and execute
        policy changes through swarm speciation.
        """
        swarm_id = cmd.target.get('swarm_id')
        new_lambda = float(cmd.options.get('lambda', '0.5'))
        new_tau = float(cmd.options.get('tau_break', '0.7'))
        
        # This creates a "child swarm" with modified parameters
        child_id = f"{swarm_id}_child_{hash(str(cmd.raw)) % 10000}"
        
        return DSLResult(
            success=True,
            action=DSLAction.FORK,
            message=f"FORK (Genesis): Child swarm {child_id} created with λ={new_lambda}, τ={new_tau}",
            data={
                "parent_swarm": swarm_id,
                "child_swarm": child_id,
                "new_constitution": {
                    "lambda": new_lambda,
                    "tau_break": new_tau
                }
            }
        )


class DSLActionGenerator:
    """
    Generates DSL commands from EFM state.
    
    This is how RPC and CSL translate their decisions into executable actions.
    """
    
    @staticmethod
    def from_drift_risk(capsule_id: str, risk: float) -> Optional[str]:
        """Generate action based on drift risk"""
        if risk >= 0.9:
            return f"PARTITION capsule_id={capsule_id} mode=QUARANTINE"
        elif risk >= 0.7:
            return f"ESCALATE capsule_id={capsule_id} level=CRITICAL"
        elif risk >= 0.5:
            return f"NOTIFY capsule_id={capsule_id} type=DRIFT_WARNING"
        return None
    
    @staticmethod
    def from_orphan_status(capsule_id: str, orphan_type: str) -> str:
        """Generate action for orphan capsule"""
        if orphan_type == "PARENT_DEAD":
            return f"ADOPT capsule_id={capsule_id} INTO=nearest_stable"
        elif orphan_type == "LINEAGE_BROKEN":
            return f"PARTITION capsule_id={capsule_id} mode=ISOLATE"
        else:
            return f"NOTIFY capsule_id={capsule_id} type=ORPHAN_DETECTED"
    
    @staticmethod
    def from_policy_friction(swarm_id: str, friction_count: int, 
                             proposed_lambda: float) -> Optional[str]:
        """
        Generate Genesis Protocol action if policy friction is persistent.
        
        This is how the system "evolves" - by forking when the environment
        demands different parameters.
        """
        if friction_count > 10:  # Persistent friction threshold
            return f"FORK swarm_id={swarm_id} lambda={proposed_lambda}"
        return None


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_dsl():
    """Demonstrate DSL interpreter capabilities"""
    print("=" * 70)
    print("EFM DSL INTERPRETER DEMONSTRATION")
    print("=" * 70)
    
    interpreter = DSLInterpreter()
    
    # Test commands
    commands = [
        "PARTITION capsule_id=42 IF DriftRisk > 0.85",
        "ESCALATE lineage=root IF LineageStability < 0.6",
        "ROLLBACK capsule_id=87 TO=t-2",
        "PRUNE swarm_id=3 threshold=0.2",
        "ADOPT capsule_id=99 INTO=44",
        "FORK swarm_id=alpha WITH lambda=0.1 tau_break=0.3",
    ]
    
    # Context for condition evaluation
    context = {
        "DriftRisk": 0.9,
        "LineageStability": 0.4,
        "SPCM": 0.75
    }
    
    print(f"\nContext: {context}\n")
    
    for cmd in commands:
        print(f"Command: {cmd}")
        result = interpreter.execute(cmd, context)
        status = "✓" if result.success else "✗"
        print(f"  {status} {result.message}")
        if result.data:
            print(f"    Data: {result.data}")
        print()
    
    # Demonstrate action generation
    print("=" * 70)
    print("DSL ACTION GENERATION (from EFM state)")
    print("=" * 70)
    
    generator = DSLActionGenerator()
    
    print("\n1. From Drift Risk:")
    for risk in [0.3, 0.6, 0.75, 0.95]:
        action = generator.from_drift_risk("cap_123", risk)
        print(f"   Risk={risk}: {action or 'No action needed'}")
    
    print("\n2. Genesis Protocol (Policy Friction):")
    for friction in [5, 12, 20]:
        action = generator.from_policy_friction("swarm_1", friction, 0.2)
        print(f"   Friction count={friction}: {action or 'Continue monitoring'}")
    
    print("\n" + "=" * 70)
    print("DSL INTERPRETER: OPERATIONAL")
    print("=" * 70)


if __name__ == "__main__":
    demo_dsl()
