"""
EFM Abstract Interfaces
=======================

This module defines abstract base classes that allow the EFM to:
1. Run today with PLACEHOLDER implementations
2. Accept REAL implementations later without changing business logic

IMPORTANT DISCLAIMERS:

CRYPTOGRAPHY:
    The default implementations use MD5/SHA256 for DEMONSTRATION ONLY.
    These are NOT cryptographically secure for production use.
    Replace with proper signing (Ed25519, ECDSA) and key management.

CONSENSUS:
    The default consensus is "Byzantine-INSPIRED" - it captures the
    spirit of BFT but simplifies several aspects:
    - No explicit view/round protocol
    - No message phases (prepare/commit)
    - Local-only voting (no network simulation)
    For formal BFT, integrate with Tendermint, HotStuff, or similar.

STORAGE:
    Default is in-memory. For production:
    - Use PostgreSQL/TimescaleDB for time-series metrics
    - Use Neo4j/DGraph for lineage graphs
    - Use Redis for real-time state
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import hashlib
import time
import json


# =============================================================================
# CRYPTOGRAPHIC INTERFACES
# =============================================================================

class SignatureScheme(Enum):
    """Supported signature schemes."""
    PLACEHOLDER_MD5 = "placeholder_md5"   # NOT SECURE - demo only
    PLACEHOLDER_SHA256 = "placeholder_sha256"  # NOT SECURE - demo only
    ED25519 = "ed25519"                   # Real crypto (requires nacl)
    ECDSA_SECP256K1 = "ecdsa_secp256k1"  # Real crypto (requires ecdsa)
    HSM_BACKED = "hsm_backed"             # Hardware Security Module


@dataclass
class Signature:
    """Cryptographic signature container."""
    scheme: SignatureScheme
    signature_bytes: bytes
    public_key: bytes
    timestamp: float
    
    def to_hex(self) -> str:
        return self.signature_bytes.hex()


class Signer(ABC):
    """
    Abstract base class for cryptographic signing.
    
    PRODUCTION NOTE:
        Replace PlaceholderSigner with a real implementation
        backed by Ed25519 or HSM for actual deployment.
    """
    
    @abstractmethod
    def sign(self, message: bytes) -> Signature:
        """Sign a message and return signature."""
        pass
    
    @abstractmethod
    def verify(self, message: bytes, signature: Signature) -> bool:
        """Verify a signature against a message."""
        pass
    
    @abstractmethod
    def get_public_key(self) -> bytes:
        """Return the public key for this signer."""
        pass


class PlaceholderSigner(Signer):
    """
    PLACEHOLDER signer using MD5/SHA256.
    
    ⚠️ WARNING: NOT CRYPTOGRAPHICALLY SECURE ⚠️
    
    This exists only to allow the code to run without
    external crypto dependencies. Replace for production.
    """
    
    def __init__(self, node_id: str, scheme: SignatureScheme = SignatureScheme.PLACEHOLDER_SHA256):
        self.node_id = node_id
        self.scheme = scheme
        # Fake "key" derived from node_id
        self._private_key = hashlib.sha256(f"private_{node_id}".encode()).digest()
        self._public_key = hashlib.sha256(f"public_{node_id}".encode()).digest()
    
    def sign(self, message: bytes) -> Signature:
        """
        Create a PLACEHOLDER signature.
        
        NOT SECURE: Just hashes message with "private key".
        """
        if self.scheme == SignatureScheme.PLACEHOLDER_MD5:
            sig = hashlib.md5(self._private_key + message).digest()
        else:
            sig = hashlib.sha256(self._private_key + message).digest()
        
        return Signature(
            scheme=self.scheme,
            signature_bytes=sig,
            public_key=self._public_key,
            timestamp=time.time()
        )
    
    def verify(self, message: bytes, signature: Signature) -> bool:
        """
        Verify a PLACEHOLDER signature.
        
        NOT SECURE: Re-computes hash and compares.
        """
        expected = self.sign(message)
        return expected.signature_bytes == signature.signature_bytes
    
    def get_public_key(self) -> bytes:
        return self._public_key


class RealSignerInterface(Signer):
    """
    Template for real cryptographic signer.
    
    To implement:
    1. pip install pynacl (for Ed25519)
    2. Implement sign/verify using nacl.signing
    3. Manage keys securely (env vars, vault, HSM)
    """
    
    def __init__(self, private_key_path: str = None):
        # In real implementation:
        # from nacl.signing import SigningKey
        # self._signing_key = SigningKey.generate() or load from path
        raise NotImplementedError(
            "RealSignerInterface requires pynacl. "
            "Install with: pip install pynacl"
        )
    
    def sign(self, message: bytes) -> Signature:
        raise NotImplementedError()
    
    def verify(self, message: bytes, signature: Signature) -> bool:
        raise NotImplementedError()
    
    def get_public_key(self) -> bytes:
        raise NotImplementedError()


# =============================================================================
# STORAGE INTERFACES  
# =============================================================================

@dataclass
class StorageRecord:
    """Generic storage record."""
    key: str
    value: Any
    timestamp: float
    metadata: Dict = None


class Storage(ABC):
    """
    Abstract base class for persistent storage.
    
    PRODUCTION NOTE:
        Replace InMemoryStorage with PostgreSQL, Redis,
        or time-series DB for actual deployment.
    """
    
    @abstractmethod
    def put(self, key: str, value: Any, metadata: Dict = None) -> bool:
        """Store a value."""
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[StorageRecord]:
        """Retrieve a value."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value."""
        pass
    
    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix filter."""
        pass
    
    @abstractmethod
    def query(self, predicate: callable) -> List[StorageRecord]:
        """Query records matching predicate."""
        pass


class InMemoryStorage(Storage):
    """
    In-memory storage implementation.
    
    ⚠️ WARNING: DATA LOST ON RESTART ⚠️
    
    Use only for testing/demo. Replace with persistent
    storage for production.
    """
    
    def __init__(self):
        self._data: Dict[str, StorageRecord] = {}
    
    def put(self, key: str, value: Any, metadata: Dict = None) -> bool:
        self._data[key] = StorageRecord(
            key=key,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        return True
    
    def get(self, key: str) -> Optional[StorageRecord]:
        return self._data.get(key)
    
    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    def list_keys(self, prefix: str = "") -> List[str]:
        return [k for k in self._data.keys() if k.startswith(prefix)]
    
    def query(self, predicate: callable) -> List[StorageRecord]:
        return [r for r in self._data.values() if predicate(r)]
    
    def size(self) -> int:
        return len(self._data)


# =============================================================================
# CONSENSUS INTERFACES
# =============================================================================

class ConsensusState(Enum):
    """State of a consensus round."""
    PENDING = "pending"
    VOTING = "voting"
    COMMITTED = "committed"
    ABORTED = "aborted"
    QUARANTINED = "quarantined"


@dataclass
class ConsensusVote:
    """A vote in consensus."""
    voter_id: str
    proposal_id: str
    vote: bool  # True = approve, False = reject
    signature: Signature
    timestamp: float
    reason: Optional[str] = None


@dataclass
class ConsensusResult:
    """Result of consensus round."""
    proposal_id: str
    state: ConsensusState
    votes_for: int
    votes_against: int
    quorum_reached: bool
    confidence: float
    byzantine_suspects: List[str]
    timestamp: float


class ConsensusProtocol(ABC):
    """
    Abstract base class for consensus protocols.
    
    IMPORTANT: The default implementation is "Byzantine-INSPIRED"
    but NOT a formal BFT protocol. Key simplifications:
    
    1. No explicit view/round numbers
    2. No prepare/commit phases
    3. No network message simulation
    4. Simplified leader election
    
    For production BFT, integrate with:
    - Tendermint
    - HotStuff
    - PBFT reference implementation
    """
    
    @abstractmethod
    def propose(self, proposal_id: str, data: Any) -> bool:
        """Submit a proposal for consensus."""
        pass
    
    @abstractmethod
    def vote(self, proposal_id: str, approve: bool, 
             voter_id: str, reason: str = None) -> ConsensusVote:
        """Cast a vote on a proposal."""
        pass
    
    @abstractmethod
    def get_result(self, proposal_id: str) -> Optional[ConsensusResult]:
        """Get the result of a consensus round."""
        pass
    
    @abstractmethod
    def is_byzantine_suspect(self, node_id: str) -> bool:
        """Check if a node is suspected byzantine."""
        pass


class SimplifiedBFTConsensus(ConsensusProtocol):
    """
    Simplified Byzantine-INSPIRED consensus.
    
    ⚠️ NOT FORMAL BFT ⚠️
    
    Simplifications vs real PBFT:
    - Single phase voting (no prepare/commit)
    - No view change protocol
    - Local state only (no network)
    - Simple reputation instead of crypto proofs
    
    Use for simulation and demo. Replace with
    real BFT library for production.
    """
    
    def __init__(self, node_count: int, byzantine_threshold: float = 0.33):
        self.node_count = node_count
        self.byzantine_threshold = byzantine_threshold
        self.quorum_threshold = 1 - byzantine_threshold  # 2/3
        
        self._proposals: Dict[str, Any] = {}
        self._votes: Dict[str, List[ConsensusVote]] = {}
        self._results: Dict[str, ConsensusResult] = {}
        self._byzantine_suspects: set = set()
        self._reputation: Dict[str, float] = {}
        
        self._signer = PlaceholderSigner("consensus_node")
    
    def propose(self, proposal_id: str, data: Any) -> bool:
        if proposal_id in self._proposals:
            return False
        
        self._proposals[proposal_id] = {
            "data": data,
            "timestamp": time.time()
        }
        self._votes[proposal_id] = []
        return True
    
    def vote(self, proposal_id: str, approve: bool,
             voter_id: str, reason: str = None) -> ConsensusVote:
        
        if proposal_id not in self._proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        
        # Check for double voting (byzantine indicator)
        existing_votes = [v for v in self._votes[proposal_id] if v.voter_id == voter_id]
        if existing_votes:
            # Double vote detected - mark as byzantine suspect
            self._byzantine_suspects.add(voter_id)
            raise ValueError(f"Double vote detected from {voter_id}")
        
        # Create vote with signature
        vote_data = f"{proposal_id}:{voter_id}:{approve}:{time.time()}"
        signature = self._signer.sign(vote_data.encode())
        
        vote = ConsensusVote(
            voter_id=voter_id,
            proposal_id=proposal_id,
            vote=approve,
            signature=signature,
            timestamp=time.time(),
            reason=reason
        )
        
        self._votes[proposal_id].append(vote)
        self._try_finalize(proposal_id)
        
        return vote
    
    def _try_finalize(self, proposal_id: str):
        """Try to finalize consensus if quorum reached."""
        votes = self._votes[proposal_id]
        
        # Filter out byzantine suspects
        valid_votes = [v for v in votes if v.voter_id not in self._byzantine_suspects]
        
        votes_for = sum(1 for v in valid_votes if v.vote)
        votes_against = sum(1 for v in valid_votes if not v.vote)
        total_votes = len(valid_votes)
        
        # Check quorum
        quorum_needed = int(self.node_count * self.quorum_threshold)
        quorum_reached = total_votes >= quorum_needed
        
        if not quorum_reached:
            return  # Wait for more votes
        
        # Determine outcome
        approval_ratio = votes_for / total_votes if total_votes > 0 else 0
        
        if approval_ratio > 0.5:
            state = ConsensusState.COMMITTED
        elif approval_ratio < 0.5:
            state = ConsensusState.ABORTED
        else:
            # Split vote - quarantine
            state = ConsensusState.QUARANTINED
        
        self._results[proposal_id] = ConsensusResult(
            proposal_id=proposal_id,
            state=state,
            votes_for=votes_for,
            votes_against=votes_against,
            quorum_reached=quorum_reached,
            confidence=abs(approval_ratio - 0.5) * 2,  # 0 = split, 1 = unanimous
            byzantine_suspects=list(self._byzantine_suspects),
            timestamp=time.time()
        )
    
    def get_result(self, proposal_id: str) -> Optional[ConsensusResult]:
        return self._results.get(proposal_id)
    
    def is_byzantine_suspect(self, node_id: str) -> bool:
        return node_id in self._byzantine_suspects


# =============================================================================
# PROOF INTERFACES
# =============================================================================

@dataclass
class ZKProof:
    """Zero-knowledge proof container."""
    proof_id: str
    proof_type: str
    public_inputs: List[Any]
    proof_data: bytes
    timestamp: float
    verifier_hint: Optional[str] = None


class ProofSystem(ABC):
    """
    Abstract base class for proof systems.
    
    IMPORTANT: Default implementation is a STUB.
    
    For real ZK proofs, integrate with:
    - Plonky2/Plonky3 (fast recursive proofs)
    - Halo2 (trustless setup)
    - Groth16 (small proofs, trusted setup)
    - STARK (post-quantum)
    """
    
    @abstractmethod
    def generate_proof(self, statement: Any, witness: Any) -> ZKProof:
        """Generate a proof."""
        pass
    
    @abstractmethod
    def verify_proof(self, proof: ZKProof, statement: Any) -> bool:
        """Verify a proof."""
        pass


class StubProofSystem(ProofSystem):
    """
    STUB proof system for API demonstration.
    
    ⚠️ NOT A REAL ZK PROOF ⚠️
    
    This generates fake "proofs" that always verify.
    Replace with real ZK library for production.
    
    Integration path:
    1. pip install plonky2 (or similar)
    2. Define circuit for EFM state transitions
    3. Implement generate_proof with real circuit
    4. Implement verify_proof with verifier
    """
    
    def __init__(self, proof_type: str = "STUB_PROOF"):
        self.proof_type = proof_type
    
    def generate_proof(self, statement: Any, witness: Any) -> ZKProof:
        """Generate a STUB proof (not real ZK)."""
        # In real implementation:
        # circuit = EFMStateCircuit(statement)
        # proof = circuit.prove(witness)
        
        proof_data = hashlib.sha256(
            json.dumps({"statement": str(statement), "witness": str(witness)}).encode()
        ).digest()
        
        return ZKProof(
            proof_id=f"proof_{time.time_ns()}",
            proof_type=self.proof_type,
            public_inputs=[statement],
            proof_data=proof_data,
            timestamp=time.time(),
            verifier_hint="STUB - replace with real ZK verifier"
        )
    
    def verify_proof(self, proof: ZKProof, statement: Any) -> bool:
        """
        Verify a STUB proof (always returns True).
        
        ⚠️ NOT SECURE - replace for production
        """
        # In real implementation:
        # verifier = EFMVerifier()
        # return verifier.verify(proof.proof_data, proof.public_inputs)
        
        return True  # STUB always verifies


# =============================================================================
# FACTORY FOR CREATING IMPLEMENTATIONS
# =============================================================================

class EFMInterfaceFactory:
    """
    Factory for creating EFM interface implementations.
    
    Usage:
        factory = EFMInterfaceFactory(production=False)
        signer = factory.create_signer("node_1")
        storage = factory.create_storage()
        consensus = factory.create_consensus(10)
    
    Set production=True to use real implementations
    (requires additional dependencies).
    """
    
    def __init__(self, production: bool = False):
        self.production = production
        
        if production:
            # Check for required dependencies
            self._check_production_deps()
    
    def _check_production_deps(self):
        """Check that production dependencies are available."""
        missing = []
        
        try:
            import nacl
        except ImportError:
            missing.append("pynacl (for Ed25519 signing)")
        
        try:
            import psycopg2
        except ImportError:
            missing.append("psycopg2 (for PostgreSQL storage)")
        
        if missing:
            raise ImportError(
                f"Production mode requires: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(m.split()[0] for m in missing)}"
            )
    
    def create_signer(self, node_id: str) -> Signer:
        """Create a signer instance."""
        if self.production:
            return RealSignerInterface()  # Would need real implementation
        return PlaceholderSigner(node_id)
    
    def create_storage(self) -> Storage:
        """Create a storage instance."""
        if self.production:
            raise NotImplementedError("Production storage not yet implemented")
        return InMemoryStorage()
    
    def create_consensus(self, node_count: int) -> ConsensusProtocol:
        """Create a consensus protocol instance."""
        if self.production:
            raise NotImplementedError("Production consensus not yet implemented")
        return SimplifiedBFTConsensus(node_count)
    
    def create_proof_system(self) -> ProofSystem:
        """Create a proof system instance."""
        if self.production:
            raise NotImplementedError("Production proof system not yet implemented")
        return StubProofSystem()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EFM ABSTRACT INTERFACES - DEMO")
    print("=" * 70)
    
    # Create factory (demo mode)
    factory = EFMInterfaceFactory(production=False)
    
    # Demo: Signer
    print("\n🔐 SIGNER (Placeholder - NOT SECURE)")
    signer = factory.create_signer("node_001")
    message = b"Hello, EFM!"
    signature = signer.sign(message)
    verified = signer.verify(message, signature)
    print(f"   Message: {message}")
    print(f"   Signature: {signature.to_hex()[:32]}...")
    print(f"   Verified: {verified}")
    print(f"   ⚠️ WARNING: Using {signature.scheme.value} - NOT for production")
    
    # Demo: Storage
    print("\n💾 STORAGE (In-Memory)")
    storage = factory.create_storage()
    storage.put("agent_001", {"phi": [0.1, 0.2, 0.3], "stability": 0.85})
    record = storage.get("agent_001")
    print(f"   Stored: agent_001")
    print(f"   Retrieved: {record.value}")
    print(f"   Keys: {storage.list_keys()}")
    print(f"   ⚠️ WARNING: Data lost on restart")
    
    # Demo: Consensus
    print("\n🗳️ CONSENSUS (Byzantine-Inspired - NOT formal BFT)")
    consensus = factory.create_consensus(node_count=5)
    consensus.propose("prop_001", {"action": "QUARANTINE", "target": "agent_007"})
    
    for i in range(4):
        vote = consensus.vote("prop_001", approve=(i < 3), voter_id=f"voter_{i}")
        print(f"   Vote from voter_{i}: {'FOR' if vote.vote else 'AGAINST'}")
    
    result = consensus.get_result("prop_001")
    if result:
        print(f"   Result: {result.state.value}")
        print(f"   Votes: {result.votes_for} for, {result.votes_against} against")
        print(f"   Confidence: {result.confidence:.2f}")
    print(f"   ⚠️ WARNING: Simplified BFT - not formal PBFT")
    
    # Demo: Proof System
    print("\n🔒 PROOF SYSTEM (Stub - NOT real ZK)")
    prover = factory.create_proof_system()
    proof = prover.generate_proof(
        statement={"swarm_size": 100, "global_sci": 0.85},
        witness={"agent_states": "..."}
    )
    verified = prover.verify_proof(proof, {"swarm_size": 100, "global_sci": 0.85})
    print(f"   Proof ID: {proof.proof_id}")
    print(f"   Verified: {verified}")
    print(f"   ⚠️ WARNING: Stub proof - always verifies")
    
    print("\n" + "=" * 70)
    print("✅ Interfaces ready (demo mode)")
    print("   For production, set production=True and install dependencies")
    print("=" * 70)
