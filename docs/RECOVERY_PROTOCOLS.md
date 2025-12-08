# EFM Recovery Protocols

This document specifies failover, recovery, and reset procedures for EFM system components.

---

## Overview

Recovery protocols handle three categories of failure:

1. **Capsule-Level Failures** - Individual agent drift or corruption
2. **Cluster-Level Failures** - d-CTM consensus loss or node failures
3. **System-Level Failures** - Global coordination breakdown

---

## 1. Capsule Recovery Protocol

### 1.1 Drift Detection → Recovery Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPSULE DRIFT RECOVERY                        │
└─────────────────────────────────────────────────────────────────┘

     ┌──────────┐
     │ Normal   │
     │ Operation│
     └────┬─────┘
          │
          ▼
     ┌──────────┐     No      ┌──────────┐
     │ Drift    │────────────►│ Continue │
     │ Detected?│             │ Normal   │
     └────┬─────┘             └──────────┘
          │ Yes
          ▼
     ┌──────────┐
     │ Severity │
     │ Check    │
     └────┬─────┘
          │
    ┌─────┼─────┬─────────────┐
    │     │     │             │
    ▼     ▼     ▼             ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌──────────┐
│ MILD │ │ MOD  │ │ SEVERE │ │ CRITICAL │
└──┬───┘ └──┬───┘ └───┬────┘ └────┬─────┘
   │        │         │           │
   ▼        ▼         ▼           ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌──────────┐
│ HEAL │ │QUARAN│ │ESCALATE│ │ ISOLATE  │
│      │ │-TINE │ │+QUARAN │ │ LINEAGE  │
└──┬───┘ └──┬───┘ └───┬────┘ └────┬─────┘
   │        │         │           │
   ▼        ▼         ▼           ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌──────────┐
│Verify│ │Repair│ │Cluster │ │ Manual   │
│Healed│ │Attempt│ │Review  │ │ Review   │
└──┬───┘ └──┬───┘ └───┬────┘ └────┬─────┘
   │        │         │           │
   └────────┴─────────┴───────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Recovery   │
            │   Complete   │
            └──────────────┘
```

### 1.2 Severity Thresholds

| Severity | Drift Velocity | TTF | Stability | Action |
|----------|---------------|-----|-----------|--------|
| MILD | < 0.1 | > 60 | > 0.7 | HEAL |
| MODERATE | 0.1 - 0.3 | 30-60 | 0.5-0.7 | QUARANTINE |
| SEVERE | 0.3 - 0.5 | 10-30 | 0.3-0.5 | ESCALATE + QUARANTINE |
| CRITICAL | > 0.5 | < 10 | < 0.3 | ISOLATE_LINEAGE |

### 1.3 Recovery Actions

#### HEAL (Automatic)

```python
def heal_capsule(capsule: CapsuleState) -> bool:
    """
    Attempt automatic repair of mild drift.
    
    Steps:
    1. Identify drift source (entropy spike, stability drop)
    2. Apply correction:
       - If entropy spike: Reduce symbolic complexity
       - If stability drop: Reinforce from stable ancestors
    3. Verify stability improves over 5 ticks
    4. Return to normal if stable, escalate if not
    """
    # Find stable ancestor
    ancestor = find_stable_ancestor(capsule)
    if ancestor:
        # Blend with stable ancestor
        capsule.phi = 0.7 * capsule.phi + 0.3 * ancestor.phi
        capsule.phi = normalize(capsule.phi)
        return verify_stability(capsule, ticks=5)
    return False
```

#### QUARANTINE (Isolation)

```python
def quarantine_capsule(capsule: CapsuleState) -> None:
    """
    Isolate capsule from swarm participation.
    
    Steps:
    1. Remove from active consensus pool
    2. Block outgoing drift propagation
    3. Continue monitoring (read-only)
    4. Attempt repair in background
    5. Require manual review for release
    """
    capsule.status = CapsuleStatus.QUARANTINED
    capsule.consensus_participation = False
    capsule.propagation_blocked = True
    
    # Start background repair
    schedule_repair_attempt(capsule, interval_ticks=10)
```

#### ESCALATE (Cluster Notification)

```python
def escalate_to_cluster(capsule: CapsuleState) -> None:
    """
    Report severe drift to cluster for coordinated response.
    
    Steps:
    1. Generate drift alert message
    2. Broadcast to all d-CTM nodes
    3. Request cluster-level assessment
    4. Await consensus on response action
    """
    alert = DCTMMessage(
        message_type=DCTMMessageType.DRIFT_ALERT,
        payload={
            "capsule_id": capsule.id,
            "drift_velocity": capsule.drift_velocity,
            "ttf": capsule.ttf,
            "lineage": get_lineage_chain(capsule),
            "recommended_action": "QUARANTINE"
        }
    )
    broadcast_to_cluster(alert)
```

#### ISOLATE_LINEAGE (Critical)

```python
def isolate_lineage(capsule: CapsuleState) -> List[str]:
    """
    Quarantine entire lineage tree to prevent cascade.
    
    Steps:
    1. Identify all descendants
    2. Quarantine each descendant
    3. Mark lineage as tainted
    4. Require manual review for entire tree
    5. Log forensic audit trail
    """
    descendants = get_all_descendants(capsule)
    isolated = []
    
    for desc in descendants:
        quarantine_capsule(desc)
        isolated.append(desc.id)
    
    # Mark root
    quarantine_capsule(capsule)
    capsule.lineage_tainted = True
    
    log_forensic_event(
        event_type="LINEAGE_ISOLATION",
        root_capsule=capsule.id,
        affected_count=len(isolated) + 1
    )
    
    return isolated
```

---

## 2. Cluster Recovery Protocol

### 2.1 Node Failure → Recovery Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLUSTER NODE RECOVERY                         │
└─────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │ Normal       │
     │ Cluster Op   │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐     No      ┌──────────────┐
     │ Heartbeat    │────────────►│ Continue     │
     │ Timeout?     │             │ Normal       │
     └──────┬───────┘             └──────────────┘
            │ Yes
            ▼
     ┌──────────────┐
     │ Mark Node    │
     │ DEGRADED     │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐     No      ┌──────────────┐
     │ Retry        │────────────►│ Mark OFFLINE │
     │ 3x Success?  │             │              │
     └──────┬───────┘             └──────┬───────┘
            │ Yes                        │
            ▼                            ▼
     ┌──────────────┐             ┌──────────────┐
     │ Restore to   │             │ Initiate     │
     │ HEALTHY      │             │ FAILOVER     │
     └──────────────┘             └──────┬───────┘
                                         │
                      ┌──────────────────┴──────────────────┐
                      │                                      │
                      ▼                                      ▼
               ┌──────────────┐                      ┌──────────────┐
               │ Redistribute │                      │ Elect New    │
               │ Capsules     │                      │ Leader       │
               └──────┬───────┘                      └──────┬───────┘
                      │                                      │
                      └──────────────────┬───────────────────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │ Resume with  │
                                  │ N-1 Nodes    │
                                  └──────────────┘
```

### 2.2 Node State Transitions

```
    HEALTHY ◄──────────────────────────────┐
       │                                    │
       │ heartbeat timeout                  │ recovery success
       ▼                                    │
    DEGRADED ──────────────────────────────►│
       │                                    │
       │ 3 failed retries                   │
       ▼                                    │
    FAILING ───────────────────────────────►│
       │                                    │
       │ failover initiated                 │
       ▼                                    │
    OFFLINE                                 │
       │                                    │
       │ manual restart                     │
       ▼                                    │
    RECOVERING ────────────────────────────►┘
```

### 2.3 Consensus Loss Recovery

```python
def handle_consensus_loss(cluster: DCTMCluster) -> None:
    """
    Recovery procedure when quorum is lost.
    
    Triggers:
    - Less than 2/3 nodes responsive
    - Conflicting state digests
    - Vote deadlock
    """
    # Step 1: Halt new proposals
    cluster.proposal_acceptance = False
    
    # Step 2: Identify partition
    responsive = [n for n in cluster.nodes if n.is_responsive()]
    unresponsive = [n for n in cluster.nodes if not n.is_responsive()]
    
    # Step 3: Check if we have quorum
    if len(responsive) >= cluster.quorum_size:
        # Majority partition - continue with warning
        cluster.status = ClusterStatus.DEGRADED
        log_warning(f"Operating with {len(responsive)}/{len(cluster.nodes)} nodes")
        cluster.proposal_acceptance = True
    else:
        # No quorum - read-only mode
        cluster.status = ClusterStatus.READ_ONLY
        log_critical("QUORUM LOST - Read-only mode engaged")
        alert_operators("Manual intervention required")
    
    # Step 4: Attempt to recover unresponsive nodes
    for node in unresponsive:
        schedule_recovery_attempt(node)
```

### 2.4 Capsule Redistribution

```python
def redistribute_capsules(failed_node: DCTMNode, cluster: DCTMCluster) -> int:
    """
    Redistribute capsules from failed node to healthy nodes.
    
    Strategy:
    1. Get capsule list from failed node's last known state
    2. Distribute evenly across remaining nodes
    3. Update routing table
    4. Verify capsule health post-transfer
    """
    orphaned_capsules = failed_node.get_capsule_list()
    healthy_nodes = [n for n in cluster.nodes if n.status == NodeStatus.HEALTHY]
    
    if not healthy_nodes:
        raise NoHealthyNodesError("Cannot redistribute - no healthy nodes")
    
    # Round-robin distribution
    transfers = 0
    for i, capsule_id in enumerate(orphaned_capsules):
        target_node = healthy_nodes[i % len(healthy_nodes)]
        transfer_capsule(capsule_id, target_node)
        transfers += 1
    
    log_info(f"Redistributed {transfers} capsules from {failed_node.id}")
    return transfers
```

---

## 3. System Recovery Protocol

### 3.1 Global Coordination Failure

```
┌─────────────────────────────────────────────────────────────────┐
│                    GLOBAL RECOVERY PROCEDURE                     │
└─────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │ Global Coord │
     │ Failure      │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │ HALT ALL     │  ◄─── All clusters enter read-only
     │ MUTATIONS    │
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │ Identify     │  ◄─── Find root cause
     │ Failure Mode │
     └──────┬───────┘
            │
     ┌──────┴──────┬──────────────┬──────────────┐
     │             │              │              │
     ▼             ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│ Network │  │ Leader  │  │ State    │  │ Byzantine│
│ Partition│  │ Failure │  │ Corrupt  │  │ Attack   │
└────┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘
     │            │            │             │
     ▼            ▼            ▼             ▼
┌─────────┐  ┌─────────┐  ┌──────────┐  ┌──────────┐
│ Wait for│  │ Elect   │  │ Restore  │  │ Evict    │
│ Reunion │  │ New     │  │ from     │  │ Malicious│
│         │  │ Leader  │  │ Snapshot │  │ Nodes    │
└────┬────┘  └────┬────┘  └────┬─────┘  └────┬─────┘
     │            │            │             │
     └────────────┴────────────┴─────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Verify State │
                  │ Consistency  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Resume       │
                  │ Operations   │
                  └──────────────┘
```

### 3.2 Emergency Halt Procedure

```python
def emergency_halt(reason: str, operator_id: str) -> HaltResult:
    """
    Emergency halt of all EFM operations.
    
    Requires 2FA confirmation.
    Logs all state for forensic analysis.
    """
    # Verify operator authority
    if not verify_2fa(operator_id):
        raise UnauthorizedError("2FA required for emergency halt")
    
    # Log pre-halt state
    state_snapshot = capture_global_state()
    
    # Broadcast halt to all nodes
    halt_message = DCTMMessage(
        message_type=DCTMMessageType.EMERGENCY_HALT,
        priority=3,  # Urgent
        payload={
            "reason": reason,
            "operator": operator_id,
            "timestamp": time.time()
        }
    )
    broadcast_global(halt_message)
    
    # Wait for acknowledgment
    acks = wait_for_acks(timeout=30)
    
    return HaltResult(
        success=len(acks) >= quorum_size,
        snapshot_id=state_snapshot.id,
        nodes_halted=len(acks)
    )
```

### 3.3 State Restoration

```python
def restore_from_snapshot(snapshot_id: str, operator_id: str) -> RestoreResult:
    """
    Restore system state from a known-good snapshot.
    
    Used after:
    - State corruption
    - Byzantine attack
    - Failed upgrade
    """
    # Verify operator authority
    if not verify_restore_authority(operator_id):
        raise UnauthorizedError("Restore requires admin authority")
    
    # Load snapshot
    snapshot = load_snapshot(snapshot_id)
    if not verify_snapshot_integrity(snapshot):
        raise CorruptedSnapshotError(f"Snapshot {snapshot_id} failed integrity check")
    
    # Halt current operations
    emergency_halt("Restoration in progress", operator_id)
    
    # Restore state to each node
    for node in get_all_nodes():
        node_state = snapshot.get_node_state(node.id)
        node.restore_state(node_state)
    
    # Verify consistency
    if not verify_global_consistency():
        raise ConsistencyError("Restoration failed consistency check")
    
    # Resume operations
    resume_operations()
    
    return RestoreResult(
        success=True,
        restored_tick=snapshot.tick,
        nodes_restored=len(get_all_nodes())
    )
```

---

## 4. Recovery Metrics

### 4.1 Target Recovery Times

| Failure Type | Detection | Recovery | Total |
|--------------|-----------|----------|-------|
| Capsule MILD drift | < 1 tick | 5 ticks | < 6 ticks |
| Capsule SEVERE drift | < 1 tick | Manual | Variable |
| Single node failure | 5 seconds | 30 seconds | < 35 seconds |
| Quorum loss | Immediate | Manual | Variable |
| Byzantine attack | 1-3 ticks | 5+ minutes | Variable |
| State corruption | Varies | From snapshot | 5-30 minutes |

### 4.2 Recovery Success Criteria

| Metric | Threshold | Action if Failed |
|--------|-----------|------------------|
| Post-recovery drift velocity | < 0.05 | Re-quarantine |
| Post-recovery stability | > 0.8 | Extend monitoring |
| Consensus reformation | 100% nodes agree | Manual intervention |
| Proof verification | All proofs valid | Investigate tampering |

---

## 5. Operator Runbook

### 5.1 Daily Operations

```markdown
## Daily Checks
1. [ ] Review overnight drift alerts
2. [ ] Check node health dashboard
3. [ ] Verify proof generation succeeded
4. [ ] Review quarantine queue
5. [ ] Check consensus stability metrics
```

### 5.2 Incident Response

```markdown
## Incident Response Checklist

### Level 1: Capsule Drift
- [ ] Verify severity classification
- [ ] Check if auto-heal succeeded
- [ ] Review lineage for inherited issues
- [ ] Release from quarantine if stable 24h

### Level 2: Node Failure
- [ ] Check physical/network cause
- [ ] Verify capsule redistribution
- [ ] Monitor for cascade failures
- [ ] Schedule node recovery

### Level 3: Consensus Loss
- [ ] Identify partition cause
- [ ] Assess quorum status
- [ ] Engage manual recovery if needed
- [ ] Post-incident review required

### Level 4: Security Incident
- [ ] Engage security team
- [ ] Preserve forensic state
- [ ] Isolate affected components
- [ ] Do NOT restore until cleared
```

---

## 6. Recovery Testing

### 6.1 Chaos Engineering Schedule

| Test | Frequency | Procedure |
|------|-----------|-----------|
| Capsule drift injection | Weekly | Inject MILD drift, verify auto-heal |
| Node failure | Monthly | Kill random node, verify redistribution |
| Network partition | Quarterly | Simulate partition, verify quorum handling |
| Full restore | Annually | Restore from snapshot, verify consistency |

### 6.2 Recovery Drill Template

```python
def run_recovery_drill(drill_type: str) -> DrillResult:
    """
    Execute recovery drill and measure performance.
    """
    # Record start state
    start_state = capture_state()
    start_time = time.time()
    
    # Inject failure
    if drill_type == "capsule_drift":
        inject_drift(random_capsule(), severity=DriftSeverity.MODERATE)
    elif drill_type == "node_failure":
        kill_random_node()
    elif drill_type == "network_partition":
        create_partition(50)  # 50% split
    
    # Wait for recovery
    recovery_complete = wait_for_recovery(timeout=300)
    
    # Measure results
    end_time = time.time()
    end_state = capture_state()
    
    return DrillResult(
        drill_type=drill_type,
        success=recovery_complete,
        recovery_time=end_time - start_time,
        state_consistent=compare_states(start_state, end_state)
    )
```
