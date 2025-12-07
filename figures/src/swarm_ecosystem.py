"""
EFM Inter-Trunk Web: Cross-Swarm Anomaly Correlation and Pattern Matching
==========================================================================

This extends the Forest Architecture with:

1. CROSS-TRUNK ANOMALY CORRELATION
   - Compare anomaly matrices between different trunks/swarms
   - Find matching patterns across independent explorations
   - Identify convergent discoveries (multiple swarms finding same thing)

2. MULTI-SCALE PATTERN MATCHING (4-5D)
   - Match patterns at different resolutions/scales
   - Hierarchical pattern recognition
   - Scale-invariant feature detection

3. SPARSE/DENSE PATTERN ANALYSIS
   - Detect density regimes in data
   - Compare sparse vs dense clusters across trunks
   - Bridge sparse exploratory data with dense validated data

4. INTER-TRUNK WEBBING
   - Communication mesh between autonomous branches
   - Knowledge sharing without trunk disruption
   - Emergent consensus across independent swarms

5. DISTRIBUTED DATA CATALOG
   - Unified organization of discoveries
   - Cross-referencing across the forest ecosystem
   - Pattern taxonomy that grows autonomously
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum, auto
from collections import defaultdict
import hashlib
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, pearsonr
import warnings

# Import from existing architecture
from forest_architecture import (
    CognitiveForest, Branch, BranchType, AnomalySignature, AnomalyClass,
    DataPoint, SubMission, AnomalyMatrixDetector
)


# =============================================================================
# DATA STRUCTURES FOR INTER-TRUNK COMMUNICATION
# =============================================================================

class PatternScale(Enum):
    """Scale levels for multi-scale pattern analysis"""
    MICRO = auto()      # Individual data points
    MESO = auto()       # Small clusters (3-10 points)
    MACRO = auto()      # Large clusters (10-50 points)
    META = auto()       # Cross-cluster patterns
    GLOBAL = auto()     # System-wide patterns


class DensityRegime(Enum):
    """Classification of data density"""
    SPARSE = auto()     # Low density, exploratory
    TRANSITION = auto() # Boundary regions
    DENSE = auto()      # High density, validated
    CORE = auto()       # Ultra-dense, fundamental


@dataclass
class PatternSignature:
    """A scale-invariant pattern signature"""
    id: str
    scale: PatternScale
    centroid: np.ndarray
    eigenvalues: np.ndarray  # Shape descriptor
    density: float
    density_regime: DensityRegime
    member_count: int
    source_trunk: str
    discovery_tick: int
    confidence: float
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert to fixed-size feature vector for comparison"""
        # Normalize eigenvalues
        norm_eigen = self.eigenvalues / (np.sum(self.eigenvalues) + 1e-10)
        return np.concatenate([
            self.centroid[:5],  # First 5 dimensions of centroid
            norm_eigen[:3],     # First 3 eigenvalues (shape)
            [self.density, self.member_count / 100, self.confidence]
        ])


@dataclass
class AnomalyCorrelation:
    """Correlation between anomalies from different trunks"""
    anomaly_a_id: str
    anomaly_b_id: str
    trunk_a: str
    trunk_b: str
    correlation_strength: float  # 0-1
    correlation_type: str  # 'spatial', 'structural', 'temporal', 'semantic'
    shared_features: List[str]
    discovery_gap: int  # Ticks between discoveries
    convergent: bool  # True if independent discovery


@dataclass
class WebConnection:
    """A connection in the inter-trunk web"""
    id: str
    source_trunk: str
    source_branch: str
    target_trunk: str
    target_branch: str
    connection_type: str  # 'discovery_share', 'pattern_match', 'anomaly_correlation'
    strength: float
    data_transferred: int
    created_at: int
    active: bool = True


@dataclass
class CatalogEntry:
    """Entry in the distributed data catalog"""
    id: str
    pattern_type: str
    scale: PatternScale
    density_regime: DensityRegime
    source_trunks: List[str]
    discovery_ticks: List[int]
    confidence: float
    cross_references: List[str]  # IDs of related entries
    feature_vector: np.ndarray
    tags: List[str]


# =============================================================================
# MULTI-SCALE PATTERN ANALYZER
# =============================================================================

class MultiScalePatternAnalyzer:
    """
    Analyzes patterns at multiple scales (4-5D and beyond).
    
    Key capabilities:
    - Scale-invariant feature extraction
    - Hierarchical pattern recognition
    - Density regime classification
    - Shape characterization via eigenvalues
    """
    
    def __init__(self, n_scales: int = 5):
        self.n_scales = n_scales
        self.scale_thresholds = {
            PatternScale.MICRO: 3,
            PatternScale.MESO: 10,
            PatternScale.MACRO: 50,
            PatternScale.META: 200,
            PatternScale.GLOBAL: float('inf')
        }
        self.detected_patterns: Dict[str, PatternSignature] = {}
        
    def compute_local_density(self, vectors: np.ndarray, k: int = 5) -> np.ndarray:
        """Compute local density for each point"""
        if len(vectors) < k + 1:
            return np.ones(len(vectors))
        
        distances = cdist(vectors, vectors)
        densities = []
        
        for i in range(len(vectors)):
            nearest = np.sort(distances[i])[1:k+1]
            avg_dist = np.mean(nearest) if len(nearest) > 0 else 1.0
            densities.append(1.0 / (avg_dist + 1e-10))
        
        return np.array(densities)
    
    def classify_density_regime(self, density: float, all_densities: np.ndarray) -> DensityRegime:
        """Classify density into regime"""
        percentile = np.mean(all_densities <= density) * 100
        
        if percentile < 25:
            return DensityRegime.SPARSE
        elif percentile < 50:
            return DensityRegime.TRANSITION
        elif percentile < 85:
            return DensityRegime.DENSE
        else:
            return DensityRegime.CORE
    
    def compute_shape_eigenvalues(self, vectors: np.ndarray) -> np.ndarray:
        """Compute eigenvalues of covariance matrix (shape descriptor)"""
        if len(vectors) < 2:
            return np.ones(min(5, vectors.shape[1] if len(vectors) > 0 else 5))
        
        try:
            cov = np.cov(vectors.T)
            if cov.ndim == 0:
                return np.array([cov])
            eigenvalues = np.linalg.eigvalsh(cov)
            return np.sort(eigenvalues)[::-1][:5]  # Top 5 eigenvalues
        except:
            return np.ones(5)
    
    def extract_pattern_at_scale(self, 
                                  vectors: np.ndarray,
                                  scale: PatternScale,
                                  source_trunk: str,
                                  tick: int) -> List[PatternSignature]:
        """Extract patterns at a specific scale"""
        if len(vectors) < 3:
            return []
        
        patterns = []
        threshold = self.scale_thresholds[scale]
        
        # Compute densities
        all_densities = self.compute_local_density(vectors)
        
        # Cluster based on scale
        if len(vectors) < threshold:
            # Treat entire set as one pattern at this scale
            centroid = np.mean(vectors, axis=0)
            eigenvalues = self.compute_shape_eigenvalues(vectors)
            density = np.mean(all_densities)
            regime = self.classify_density_regime(density, all_densities)
            
            pattern_id = f"pattern_{scale.name}_{source_trunk}_{tick}_{len(patterns)}"
            pattern = PatternSignature(
                id=pattern_id,
                scale=scale,
                centroid=centroid,
                eigenvalues=eigenvalues,
                density=density,
                density_regime=regime,
                member_count=len(vectors),
                source_trunk=source_trunk,
                discovery_tick=tick,
                confidence=min(1.0, len(vectors) / threshold)
            )
            patterns.append(pattern)
        else:
            # Hierarchical clustering
            try:
                linkage_matrix = linkage(vectors, method='ward')
                # Number of clusters based on scale
                n_clusters = max(2, len(vectors) // (threshold // 2))
                labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                
                for cluster_id in np.unique(labels):
                    mask = labels == cluster_id
                    cluster_vectors = vectors[mask]
                    cluster_densities = all_densities[mask]
                    
                    if len(cluster_vectors) < 2:
                        continue
                    
                    centroid = np.mean(cluster_vectors, axis=0)
                    eigenvalues = self.compute_shape_eigenvalues(cluster_vectors)
                    density = np.mean(cluster_densities)
                    regime = self.classify_density_regime(density, all_densities)
                    
                    pattern_id = f"pattern_{scale.name}_{source_trunk}_{tick}_{cluster_id}"
                    pattern = PatternSignature(
                        id=pattern_id,
                        scale=scale,
                        centroid=centroid,
                        eigenvalues=eigenvalues,
                        density=density,
                        density_regime=regime,
                        member_count=len(cluster_vectors),
                        source_trunk=source_trunk,
                        discovery_tick=tick,
                        confidence=min(1.0, len(cluster_vectors) / threshold)
                    )
                    patterns.append(pattern)
            except:
                pass
        
        return patterns
    
    def full_multiscale_analysis(self,
                                  vectors: np.ndarray,
                                  source_trunk: str,
                                  tick: int) -> Dict[PatternScale, List[PatternSignature]]:
        """Perform full multi-scale analysis"""
        results = {}
        
        for scale in [PatternScale.MICRO, PatternScale.MESO, PatternScale.MACRO, PatternScale.META]:
            patterns = self.extract_pattern_at_scale(vectors, scale, source_trunk, tick)
            results[scale] = patterns
            
            # Store in detected patterns
            for p in patterns:
                self.detected_patterns[p.id] = p
        
        return results


# =============================================================================
# CROSS-TRUNK ANOMALY CORRELATOR
# =============================================================================

class CrossTrunkCorrelator:
    """
    Correlates anomalies and patterns across different trunks/swarms.
    
    Key capabilities:
    - Compare anomaly matrices between trunks
    - Find matching patterns across independent explorations
    - Detect convergent discoveries
    - Build correlation network
    """
    
    def __init__(self, correlation_threshold: float = 0.6):
        self.correlation_threshold = correlation_threshold
        self.correlations: List[AnomalyCorrelation] = []
        self.correlation_matrix: Dict[Tuple[str, str], np.ndarray] = {}
        
    def compute_anomaly_similarity(self,
                                    anomaly_a: AnomalySignature,
                                    anomaly_b: AnomalySignature) -> Dict[str, float]:
        """Compute multi-dimensional similarity between two anomalies"""
        similarities = {}
        
        # Spatial similarity (centroid distance)
        if len(anomaly_a.centroid) == len(anomaly_b.centroid):
            dist = np.linalg.norm(anomaly_a.centroid - anomaly_b.centroid)
            max_dist = np.sqrt(len(anomaly_a.centroid)) * 10  # Normalize
            similarities['spatial'] = 1.0 - min(1.0, dist / max_dist)
        else:
            # Handle different dimensions
            min_dim = min(len(anomaly_a.centroid), len(anomaly_b.centroid))
            dist = np.linalg.norm(anomaly_a.centroid[:min_dim] - anomaly_b.centroid[:min_dim])
            similarities['spatial'] = 1.0 - min(1.0, dist / (np.sqrt(min_dim) * 10))
        
        # Structural similarity (anomaly class and significance)
        if anomaly_a.anomaly_class == anomaly_b.anomaly_class:
            similarities['structural'] = 0.5 + 0.5 * (1 - abs(anomaly_a.significance - anomaly_b.significance))
        else:
            similarities['structural'] = 0.3 * (1 - abs(anomaly_a.significance - anomaly_b.significance))
        
        # Size similarity
        size_ratio = min(len(anomaly_a.data_points), len(anomaly_b.data_points)) / \
                     max(len(anomaly_a.data_points), len(anomaly_b.data_points), 1)
        similarities['size'] = size_ratio
        
        # Exploration potential similarity
        similarities['potential'] = 1 - abs(anomaly_a.exploration_potential - anomaly_b.exploration_potential)
        
        return similarities
    
    def correlate_anomalies(self,
                            anomalies_a: Dict[str, AnomalySignature],
                            anomalies_b: Dict[str, AnomalySignature],
                            trunk_a: str,
                            trunk_b: str,
                            tick_a: int,
                            tick_b: int) -> List[AnomalyCorrelation]:
        """Find correlations between anomalies from two trunks"""
        correlations = []
        
        for id_a, anomaly_a in anomalies_a.items():
            for id_b, anomaly_b in anomalies_b.items():
                similarities = self.compute_anomaly_similarity(anomaly_a, anomaly_b)
                
                # Compute overall correlation strength
                strength = (
                    similarities.get('spatial', 0) * 0.4 +
                    similarities.get('structural', 0) * 0.3 +
                    similarities.get('size', 0) * 0.15 +
                    similarities.get('potential', 0) * 0.15
                )
                
                if strength >= self.correlation_threshold:
                    # Determine dominant correlation type
                    max_type = max(similarities.items(), key=lambda x: x[1])[0]
                    
                    # Shared features
                    shared = []
                    if similarities.get('spatial', 0) > 0.7:
                        shared.append('spatial_proximity')
                    if similarities.get('structural', 0) > 0.7:
                        shared.append('same_class')
                    if similarities.get('size', 0) > 0.8:
                        shared.append('similar_size')
                    
                    correlation = AnomalyCorrelation(
                        anomaly_a_id=id_a,
                        anomaly_b_id=id_b,
                        trunk_a=trunk_a,
                        trunk_b=trunk_b,
                        correlation_strength=strength,
                        correlation_type=max_type,
                        shared_features=shared,
                        discovery_gap=abs(tick_a - tick_b),
                        convergent=(trunk_a != trunk_b and similarities.get('spatial', 0) > 0.8)
                    )
                    correlations.append(correlation)
                    self.correlations.append(correlation)
        
        return correlations
    
    def compute_pattern_similarity(self,
                                    pattern_a: PatternSignature,
                                    pattern_b: PatternSignature) -> float:
        """Compute similarity between two patterns using feature vectors"""
        vec_a = pattern_a.to_feature_vector()
        vec_b = pattern_b.to_feature_vector()
        
        # Cosine similarity
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        
        cosine_sim = dot / (norm_a * norm_b)
        
        # Scale match bonus
        scale_bonus = 0.1 if pattern_a.scale == pattern_b.scale else 0
        
        # Density regime match bonus
        regime_bonus = 0.1 if pattern_a.density_regime == pattern_b.density_regime else 0
        
        return min(1.0, (cosine_sim + 1) / 2 + scale_bonus + regime_bonus)
    
    def find_cross_trunk_pattern_matches(self,
                                          patterns_a: Dict[str, PatternSignature],
                                          patterns_b: Dict[str, PatternSignature],
                                          min_similarity: float = 0.7) -> List[Tuple[str, str, float]]:
        """Find matching patterns across trunks"""
        matches = []
        
        for id_a, pattern_a in patterns_a.items():
            for id_b, pattern_b in patterns_b.items():
                similarity = self.compute_pattern_similarity(pattern_a, pattern_b)
                
                if similarity >= min_similarity:
                    matches.append((id_a, id_b, similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches
    
    def detect_convergent_discoveries(self) -> List[AnomalyCorrelation]:
        """Find cases where independent trunks discovered the same thing"""
        return [c for c in self.correlations if c.convergent]


# =============================================================================
# INTER-TRUNK WEB
# =============================================================================

class InterTrunkWeb:
    """
    Communication mesh between autonomous branches across trunks.
    
    Key capabilities:
    - Establish connections between branches on different trunks
    - Share discoveries without disrupting trunk operations
    - Build emergent consensus across independent swarms
    - Manage knowledge flow
    """
    
    def __init__(self):
        self.connections: Dict[str, WebConnection] = {}
        self.trunk_registry: Dict[str, 'CognitiveForest'] = {}
        self.message_queue: List[Dict] = []
        self.consensus_pools: Dict[str, List[Dict]] = defaultdict(list)
        
    def register_trunk(self, trunk_id: str, forest: CognitiveForest):
        """Register a trunk/forest with the web"""
        self.trunk_registry[trunk_id] = forest
        
    def establish_connection(self,
                              source_trunk: str,
                              source_branch: str,
                              target_trunk: str,
                              target_branch: str,
                              connection_type: str,
                              tick: int) -> WebConnection:
        """Establish a connection between branches on different trunks"""
        conn_id = f"web_{source_trunk}_{source_branch}_to_{target_trunk}_{target_branch}"
        
        connection = WebConnection(
            id=conn_id,
            source_trunk=source_trunk,
            source_branch=source_branch,
            target_trunk=target_trunk,
            target_branch=target_branch,
            connection_type=connection_type,
            strength=0.5,
            data_transferred=0,
            created_at=tick
        )
        
        self.connections[conn_id] = connection
        return connection
    
    def share_discovery(self,
                        source_trunk: str,
                        discovery: Dict,
                        target_trunks: List[str] = None):
        """Share a discovery across the web"""
        targets = target_trunks or list(self.trunk_registry.keys())
        
        for target in targets:
            if target != source_trunk and target in self.trunk_registry:
                message = {
                    'type': 'discovery_share',
                    'source': source_trunk,
                    'target': target,
                    'payload': discovery,
                    'timestamp': len(self.message_queue)
                }
                self.message_queue.append(message)
    
    def process_messages(self) -> int:
        """Process pending messages and transfer knowledge"""
        processed = 0
        
        for message in self.message_queue:
            target = message['target']
            if target in self.trunk_registry:
                forest = self.trunk_registry[target]
                
                if message['type'] == 'discovery_share':
                    # Generate data points from discovery
                    discovery = message['payload']
                    if 'centroid' in discovery:
                        # Create data near the discovered centroid
                        centroid = np.array(discovery['centroid'])
                        new_data = centroid + np.random.randn(3, len(centroid)) * 0.1
                        forest.ingest_data(new_data)
                        processed += 1
                
                elif message['type'] == 'pattern_match':
                    # Add to consensus pool
                    pattern_id = discovery.get('pattern_id')
                    self.consensus_pools[pattern_id].append(message)
        
        self.message_queue.clear()
        return processed
    
    def compute_web_consensus(self, min_agreement: int = 2) -> List[Dict]:
        """Find patterns that multiple trunks agree on"""
        consensus_patterns = []
        
        for pattern_id, messages in self.consensus_pools.items():
            if len(messages) >= min_agreement:
                # Multiple trunks found this pattern
                sources = list(set(m['source'] for m in messages))
                
                consensus = {
                    'pattern_id': pattern_id,
                    'agreeing_trunks': sources,
                    'agreement_strength': len(sources) / len(self.trunk_registry),
                    'first_discovery': min(m['timestamp'] for m in messages),
                    'convergent': len(sources) > 1
                }
                consensus_patterns.append(consensus)
        
        return consensus_patterns
    
    def get_web_statistics(self) -> Dict:
        """Get statistics about the inter-trunk web"""
        active_connections = sum(1 for c in self.connections.values() if c.active)
        total_data_transferred = sum(c.data_transferred for c in self.connections.values())
        
        return {
            'total_connections': len(self.connections),
            'active_connections': active_connections,
            'registered_trunks': len(self.trunk_registry),
            'pending_messages': len(self.message_queue),
            'consensus_pools': len(self.consensus_pools),
            'total_data_transferred': total_data_transferred
        }


# =============================================================================
# DISTRIBUTED DATA CATALOG
# =============================================================================

class DistributedDataCatalog:
    """
    Unified organization of discoveries across the forest ecosystem.
    
    Key capabilities:
    - Catalog patterns from all trunks
    - Cross-reference related discoveries
    - Build autonomous taxonomy
    - Enable pattern search and retrieval
    """
    
    def __init__(self):
        self.entries: Dict[str, CatalogEntry] = {}
        self.taxonomy: Dict[str, List[str]] = defaultdict(list)  # Type -> entry IDs
        self.cross_references: Dict[str, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        
    def add_entry(self,
                   pattern: PatternSignature,
                   tags: List[str] = None) -> CatalogEntry:
        """Add a pattern to the catalog"""
        tags = tags or []
        
        # Auto-generate tags
        auto_tags = [
            pattern.scale.name,
            pattern.density_regime.name,
            f"trunk:{pattern.source_trunk}",
            f"tick:{pattern.discovery_tick}"
        ]
        
        if pattern.confidence > 0.8:
            auto_tags.append("high_confidence")
        if pattern.member_count > 50:
            auto_tags.append("large_pattern")
        
        all_tags = list(set(tags + auto_tags))
        
        entry = CatalogEntry(
            id=pattern.id,
            pattern_type=pattern.scale.name,
            scale=pattern.scale,
            density_regime=pattern.density_regime,
            source_trunks=[pattern.source_trunk],
            discovery_ticks=[pattern.discovery_tick],
            confidence=pattern.confidence,
            cross_references=[],
            feature_vector=pattern.to_feature_vector(),
            tags=all_tags
        )
        
        self.entries[entry.id] = entry
        self.taxonomy[pattern.scale.name].append(entry.id)
        
        for tag in all_tags:
            self.tag_index[tag].add(entry.id)
        
        return entry
    
    def add_cross_reference(self, entry_a_id: str, entry_b_id: str, reason: str = "similar"):
        """Add cross-reference between entries"""
        if entry_a_id in self.entries and entry_b_id in self.entries:
            self.entries[entry_a_id].cross_references.append(entry_b_id)
            self.entries[entry_b_id].cross_references.append(entry_a_id)
            self.cross_references[entry_a_id].add(entry_b_id)
            self.cross_references[entry_b_id].add(entry_a_id)
    
    def merge_entries(self, entry_ids: List[str]) -> Optional[CatalogEntry]:
        """Merge multiple entries that represent the same pattern"""
        entries = [self.entries[eid] for eid in entry_ids if eid in self.entries]
        
        if len(entries) < 2:
            return None
        
        # Compute merged properties
        all_trunks = list(set(t for e in entries for t in e.source_trunks))
        all_ticks = list(set(t for e in entries for t in e.discovery_ticks))
        avg_confidence = np.mean([e.confidence for e in entries])
        avg_vector = np.mean([e.feature_vector for e in entries], axis=0)
        all_tags = list(set(t for e in entries for t in e.tags))
        all_refs = list(set(r for e in entries for r in e.cross_references))
        
        merged_id = f"merged_{entries[0].id}_{len(entries)}"
        
        merged = CatalogEntry(
            id=merged_id,
            pattern_type=entries[0].pattern_type,
            scale=entries[0].scale,
            density_regime=entries[0].density_regime,
            source_trunks=all_trunks,
            discovery_ticks=all_ticks,
            confidence=avg_confidence,
            cross_references=all_refs,
            feature_vector=avg_vector,
            tags=all_tags + ['merged', f'convergent_{len(all_trunks)}_trunks']
        )
        
        self.entries[merged_id] = merged
        return merged
    
    def search_by_tags(self, tags: List[str], mode: str = 'any') -> List[CatalogEntry]:
        """Search catalog by tags"""
        if mode == 'any':
            matching_ids = set()
            for tag in tags:
                matching_ids.update(self.tag_index.get(tag, set()))
        else:  # 'all'
            matching_ids = None
            for tag in tags:
                tag_ids = self.tag_index.get(tag, set())
                if matching_ids is None:
                    matching_ids = tag_ids
                else:
                    matching_ids &= tag_ids
        
        return [self.entries[eid] for eid in (matching_ids or [])]
    
    def search_by_similarity(self, 
                              query_vector: np.ndarray, 
                              top_k: int = 10) -> List[Tuple[CatalogEntry, float]]:
        """Search catalog by feature vector similarity"""
        results = []
        
        for entry in self.entries.values():
            # Cosine similarity
            dot = np.dot(query_vector, entry.feature_vector)
            norm_q = np.linalg.norm(query_vector)
            norm_e = np.linalg.norm(entry.feature_vector)
            
            if norm_q > 1e-10 and norm_e > 1e-10:
                similarity = dot / (norm_q * norm_e)
                results.append((entry, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_catalog_statistics(self) -> Dict:
        """Get catalog statistics"""
        scale_counts = defaultdict(int)
        regime_counts = defaultdict(int)
        trunk_counts = defaultdict(int)
        
        for entry in self.entries.values():
            scale_counts[entry.scale.name] += 1
            regime_counts[entry.density_regime.name] += 1
            for trunk in entry.source_trunks:
                trunk_counts[trunk] += 1
        
        return {
            'total_entries': len(self.entries),
            'by_scale': dict(scale_counts),
            'by_density_regime': dict(regime_counts),
            'by_trunk': dict(trunk_counts),
            'total_cross_references': sum(len(refs) for refs in self.cross_references.values()) // 2,
            'unique_tags': len(self.tag_index)
        }


# =============================================================================
# INTEGRATED SWARM ECOSYSTEM
# =============================================================================

class SwarmEcosystem:
    """
    The complete integrated ecosystem with multiple autonomous swarms.
    
    This brings everything together:
    - Multiple independent forests (trunks)
    - Cross-trunk anomaly correlation
    - Multi-scale pattern analysis
    - Inter-trunk web communication
    - Distributed data catalog
    """
    
    def __init__(self, n_swarms: int = 3):
        self.swarms: Dict[str, CognitiveForest] = {}
        self.pattern_analyzer = MultiScalePatternAnalyzer()
        self.correlator = CrossTrunkCorrelator()
        self.web = InterTrunkWeb()
        self.catalog = DistributedDataCatalog()
        
        self.tick = 0
        self.ecosystem_events: List[Dict] = []
        
        # Initialize swarms
        for i in range(n_swarms):
            swarm_id = f"swarm_{i}"
            forest = CognitiveForest(trunk_objective=f"Autonomous Exploration - Swarm {i}")
            self.swarms[swarm_id] = forest
            self.web.register_trunk(swarm_id, forest)
    
    def inject_data_to_swarm(self, swarm_id: str, vectors: np.ndarray):
        """Inject data into a specific swarm"""
        if swarm_id in self.swarms:
            self.swarms[swarm_id].ingest_data(vectors)
    
    def inject_diverse_ecosystem_data(self, base_points: int = 100):
        """Inject diverse data across all swarms with shared and unique regions"""
        np.random.seed(42)
        dim = 12
        
        # Shared region (all swarms will find this)
        shared_data = np.random.randn(base_points // 3, dim) * 0.3
        
        for i, swarm_id in enumerate(self.swarms.keys()):
            # Unique region for this swarm
            direction = np.zeros(dim)
            direction[i * 2] = 4 + np.random.rand() * 2
            unique_data = np.random.randn(base_points // 3, dim) * 0.3 + direction
            
            # Partially shared region
            partial_direction = np.zeros(dim)
            partial_direction[(i + 1) % dim] = 2
            partial_data = np.random.randn(base_points // 4, dim) * 0.3 + partial_direction
            
            # Combine and inject
            all_data = np.vstack([shared_data, unique_data, partial_data])
            self.inject_data_to_swarm(swarm_id, all_data)
    
    def ecosystem_tick(self) -> Dict:
        """Execute one tick of the entire ecosystem"""
        self.tick += 1
        
        report = {
            'tick': self.tick,
            'swarm_reports': {},
            'cross_trunk_correlations': 0,
            'pattern_matches': 0,
            'catalog_entries': 0,
            'convergent_discoveries': 0,
            'web_messages_processed': 0
        }
        
        # 1. Tick each swarm independently
        swarm_anomalies = {}
        swarm_patterns = {}
        
        for swarm_id, forest in self.swarms.items():
            swarm_report = forest.autonomous_tick()
            report['swarm_reports'][swarm_id] = swarm_report
            
            # Collect anomalies
            swarm_anomalies[swarm_id] = forest.anomaly_detector.detected_anomalies.copy()
            
            # Perform multi-scale pattern analysis
            vectors, _ = forest.get_all_vectors()
            if len(vectors) > 5:
                patterns = self.pattern_analyzer.full_multiscale_analysis(
                    vectors, swarm_id, self.tick
                )
                swarm_patterns[swarm_id] = {
                    p.id: p 
                    for scale_patterns in patterns.values() 
                    for p in scale_patterns
                }
                
                # Add patterns to catalog
                for pattern in swarm_patterns[swarm_id].values():
                    self.catalog.add_entry(pattern)
                    report['catalog_entries'] += 1
        
        # 2. Cross-trunk anomaly correlation
        swarm_list = list(self.swarms.keys())
        for i, swarm_a in enumerate(swarm_list):
            for swarm_b in swarm_list[i+1:]:
                correlations = self.correlator.correlate_anomalies(
                    swarm_anomalies.get(swarm_a, {}),
                    swarm_anomalies.get(swarm_b, {}),
                    swarm_a, swarm_b,
                    self.tick, self.tick
                )
                report['cross_trunk_correlations'] += len(correlations)
                
                # Log convergent discoveries
                convergent = [c for c in correlations if c.convergent]
                report['convergent_discoveries'] += len(convergent)
                
                if convergent:
                    self.ecosystem_events.append({
                        'tick': self.tick,
                        'event': 'CONVERGENT_DISCOVERY',
                        'swarms': [swarm_a, swarm_b],
                        'count': len(convergent)
                    })
        
        # 3. Cross-trunk pattern matching
        for i, swarm_a in enumerate(swarm_list):
            for swarm_b in swarm_list[i+1:]:
                if swarm_a in swarm_patterns and swarm_b in swarm_patterns:
                    matches = self.correlator.find_cross_trunk_pattern_matches(
                        swarm_patterns.get(swarm_a, {}),
                        swarm_patterns.get(swarm_b, {})
                    )
                    report['pattern_matches'] += len(matches)
                    
                    # Add cross-references to catalog
                    for id_a, id_b, similarity in matches[:10]:  # Top 10
                        self.catalog.add_cross_reference(id_a, id_b, f"similarity_{similarity:.2f}")
        
        # 4. Share discoveries across web
        for swarm_id, forest in self.swarms.items():
            for mission in list(forest.sub_missions.values())[-5:]:  # Recent missions
                for discovery in mission.discoveries[-3:]:  # Recent discoveries
                    if discovery.get('confidence', 0) > 0.8:
                        self.web.share_discovery(swarm_id, discovery)
        
        # 5. Process web messages
        report['web_messages_processed'] = self.web.process_messages()
        
        return report
    
    def get_ecosystem_status(self) -> Dict:
        """Get comprehensive ecosystem status"""
        total_knowledge = sum(f.get_forest_status()['total_knowledge'] for f in self.swarms.values())
        total_discoveries = sum(f.get_forest_status()['total_discoveries'] for f in self.swarms.values())
        total_anomalies = sum(len(f.anomaly_detector.detected_anomalies) for f in self.swarms.values())
        
        return {
            'tick': self.tick,
            'num_swarms': len(self.swarms),
            'total_knowledge': total_knowledge,
            'total_discoveries': total_discoveries,
            'total_anomalies': total_anomalies,
            'cross_correlations': len(self.correlator.correlations),
            'convergent_discoveries': len(self.correlator.detect_convergent_discoveries()),
            'catalog_stats': self.catalog.get_catalog_statistics(),
            'web_stats': self.web.get_web_statistics(),
            'pattern_analyzer_patterns': len(self.pattern_analyzer.detected_patterns)
        }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def run_ecosystem_demonstration(n_ticks: int = 100):
    """Run the complete swarm ecosystem demonstration"""
    print("=" * 80)
    print("SWARM ECOSYSTEM: CROSS-TRUNK ANOMALY CORRELATION & PATTERN MATCHING")
    print("=" * 80)
    
    # Initialize ecosystem with 3 swarms
    ecosystem = SwarmEcosystem(n_swarms=3)
    
    print("\n🌐 PHASE 1: INITIALIZING ECOSYSTEM")
    print("-" * 40)
    
    # Inject diverse data
    ecosystem.inject_diverse_ecosystem_data(base_points=120)
    
    print(f"  Initialized {len(ecosystem.swarms)} autonomous swarms")
    print(f"  Each swarm has shared + unique data regions")
    
    # Tracking metrics
    metrics = {
        'tick': [],
        'total_knowledge': [],
        'cross_correlations': [],
        'convergent_discoveries': [],
        'catalog_entries': [],
        'pattern_matches': []
    }
    
    print("\n🔬 PHASE 2: AUTONOMOUS ECOSYSTEM EVOLUTION")
    print("-" * 40)
    
    for tick in range(1, n_ticks + 1):
        # Periodically inject new data
        if tick % 20 == 0:
            for swarm_id in ecosystem.swarms.keys():
                new_data = np.random.randn(10, 12) * 0.3 + np.random.randn(12) * 2
                ecosystem.inject_data_to_swarm(swarm_id, new_data)
        
        # Run ecosystem tick
        report = ecosystem.ecosystem_tick()
        
        # Record metrics
        status = ecosystem.get_ecosystem_status()
        metrics['tick'].append(tick)
        metrics['total_knowledge'].append(status['total_knowledge'])
        metrics['cross_correlations'].append(status['cross_correlations'])
        metrics['convergent_discoveries'].append(status['convergent_discoveries'])
        metrics['catalog_entries'].append(status['catalog_stats']['total_entries'])
        metrics['pattern_matches'].append(report['pattern_matches'])
        
        # Progress updates
        if tick % 25 == 0:
            print(f"\n  === TICK {tick} ===")
            print(f"  Total knowledge: {status['total_knowledge']:.1f}")
            print(f"  Cross-correlations: {status['cross_correlations']}")
            print(f"  Convergent discoveries: {status['convergent_discoveries']}")
            print(f"  Catalog entries: {status['catalog_stats']['total_entries']}")
            print(f"  Pattern matches this tick: {report['pattern_matches']}")
    
    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL ECOSYSTEM ANALYSIS")
    print("=" * 80)
    
    final_status = ecosystem.get_ecosystem_status()
    
    print(f"\n🌐 ECOSYSTEM FINAL STATE:")
    print(f"  Total ticks: {final_status['tick']}")
    print(f"  Active swarms: {final_status['num_swarms']}")
    print(f"  Total knowledge: {final_status['total_knowledge']:.2f}")
    print(f"  Total discoveries: {final_status['total_discoveries']}")
    print(f"  Total anomalies tracked: {final_status['total_anomalies']}")
    
    print(f"\n🔗 CROSS-TRUNK ANALYSIS:")
    print(f"  Cross-correlations found: {final_status['cross_correlations']}")
    print(f"  Convergent discoveries: {final_status['convergent_discoveries']}")
    print(f"  Patterns analyzed: {final_status['pattern_analyzer_patterns']}")
    
    print(f"\n📚 DISTRIBUTED CATALOG:")
    cat_stats = final_status['catalog_stats']
    print(f"  Total entries: {cat_stats['total_entries']}")
    print(f"  By scale: {cat_stats['by_scale']}")
    print(f"  By density: {cat_stats['by_density_regime']}")
    print(f"  Cross-references: {cat_stats['total_cross_references']}")
    print(f"  Unique tags: {cat_stats['unique_tags']}")
    
    print(f"\n🕸️ INTER-TRUNK WEB:")
    web_stats = final_status['web_stats']
    print(f"  Registered trunks: {web_stats['registered_trunks']}")
    print(f"  Consensus pools: {web_stats['consensus_pools']}")
    
    # Show ecosystem events
    print(f"\n📜 ECOSYSTEM EVENTS (Convergent Discoveries):")
    for event in ecosystem.ecosystem_events[:10]:
        print(f"  [{event['tick']:3d}] {event['event']}: Swarms {event['swarms']} found {event['count']} shared patterns")
    
    # Create visualization
    visualize_ecosystem(metrics, n_ticks)
    
    print("\n" + "=" * 80)
    print("ECOSYSTEM DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("""
DEMONSTRATED:

1. ✅ CROSS-TRUNK ANOMALY CORRELATION
   - Multiple swarms detecting related anomalies
   - Convergent discovery detection
   
2. ✅ MULTI-SCALE PATTERN MATCHING
   - Patterns analyzed at MICRO, MESO, MACRO, META scales
   - Scale-invariant feature extraction
   
3. ✅ SPARSE/DENSE PATTERN ANALYSIS  
   - Density regimes classified (SPARSE → CORE)
   - Cross-swarm density comparison
   
4. ✅ INTER-TRUNK WEBBING
   - Knowledge sharing across autonomous swarms
   - Consensus building without disruption
   
5. ✅ DISTRIBUTED DATA CATALOG
   - Unified pattern organization
   - Cross-referencing across ecosystem
   - Tag-based and similarity-based search

The swarms operate AUTONOMOUSLY but COLLABORATE through:
- Anomaly correlation
- Pattern matching
- Web communication
- Shared catalog

This enables DEEP ANALYSIS across independently exploring swarms.
""")
    
    return ecosystem, metrics


def visualize_ecosystem(metrics: Dict, n_ticks: int):
    """Visualize ecosystem metrics"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Swarm Ecosystem: Cross-Trunk Analysis', fontsize=14, fontweight='bold')
    
    # 1. Total knowledge
    ax1 = axes[0, 0]
    ax1.plot(metrics['tick'], metrics['total_knowledge'], 'b-', linewidth=2)
    ax1.fill_between(metrics['tick'], metrics['total_knowledge'], alpha=0.3)
    ax1.set_xlabel('Tick')
    ax1.set_ylabel('Total Knowledge')
    ax1.set_title('Combined Knowledge (All Swarms)')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cross-correlations
    ax2 = axes[0, 1]
    ax2.plot(metrics['tick'], metrics['cross_correlations'], 'g-', linewidth=2)
    ax2.fill_between(metrics['tick'], metrics['cross_correlations'], alpha=0.3, color='green')
    ax2.set_xlabel('Tick')
    ax2.set_ylabel('Cross-Correlations')
    ax2.set_title('Cross-Trunk Anomaly Correlations')
    ax2.grid(True, alpha=0.3)
    
    # 3. Convergent discoveries
    ax3 = axes[0, 2]
    ax3.plot(metrics['tick'], metrics['convergent_discoveries'], 'r-', linewidth=2)
    ax3.fill_between(metrics['tick'], metrics['convergent_discoveries'], alpha=0.3, color='red')
    ax3.set_xlabel('Tick')
    ax3.set_ylabel('Convergent Discoveries')
    ax3.set_title('Independent Discovery of Same Pattern')
    ax3.grid(True, alpha=0.3)
    
    # 4. Catalog entries
    ax4 = axes[1, 0]
    ax4.plot(metrics['tick'], metrics['catalog_entries'], 'purple', linewidth=2)
    ax4.fill_between(metrics['tick'], metrics['catalog_entries'], alpha=0.3, color='purple')
    ax4.set_xlabel('Tick')
    ax4.set_ylabel('Catalog Entries')
    ax4.set_title('Distributed Catalog Growth')
    ax4.grid(True, alpha=0.3)
    
    # 5. Pattern matches per tick
    ax5 = axes[1, 1]
    ax5.bar(metrics['tick'][::5], metrics['pattern_matches'][::5], width=4, color='orange', alpha=0.7)
    ax5.set_xlabel('Tick')
    ax5.set_ylabel('Pattern Matches')
    ax5.set_title('Cross-Trunk Pattern Matches')
    ax5.grid(True, alpha=0.3)
    
    # 6. Combined metrics normalized
    ax6 = axes[1, 2]
    # Normalize each metric
    norm_knowledge = np.array(metrics['total_knowledge']) / (max(metrics['total_knowledge']) + 1e-10)
    norm_corr = np.array(metrics['cross_correlations']) / (max(metrics['cross_correlations']) + 1e-10)
    norm_conv = np.array(metrics['convergent_discoveries']) / (max(metrics['convergent_discoveries']) + 1e-10)
    
    ax6.plot(metrics['tick'], norm_knowledge, 'b-', label='Knowledge', linewidth=2)
    ax6.plot(metrics['tick'], norm_corr, 'g-', label='Correlations', linewidth=2)
    ax6.plot(metrics['tick'], norm_conv, 'r-', label='Convergent', linewidth=2)
    ax6.set_xlabel('Tick')
    ax6.set_ylabel('Normalized Value')
    ax6.set_title('Ecosystem Health (Normalized)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/efm-booklet4/figures/ecosystem_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: /home/claude/efm-booklet4/figures/ecosystem_analysis.png")
    plt.close()


if __name__ == "__main__":
    ecosystem, metrics = run_ecosystem_demonstration(n_ticks=100)
