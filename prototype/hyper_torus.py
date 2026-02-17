"""
SCBE-AETHERMOORE Hyper-Torus Module
====================================

Implements T^n (n-dimensional torus) geometry as an "escape hatch" for resolving
3D Hamiltonian dead-ends in the PHDM polyhedral lattice.

Key Concepts:
    1. Hyper-Torus T^4 = S^1 x S^1 x S^1 x S^1 (4D torus)
    2. Toroidal distance metric with periodic boundaries
    3. Torus Lift: Elevate stuck 3D graphs to 4D toroidal space
    4. Mirror Symmetry Duality for key swapping (Calabi-Yau inspired)
    5. Sacred Tongue integration (6 tongues as circles on torus)

Architecture Role:
    - Primary: Poincare ball (hyperbolic containment)
    - Secondary: PHDM polyhedra (cognitive regions)
    - Escape Hatch: Hyper-Torus (resolve dead-ends via extra dimension)

Author: SCBE-AETHERMOORE Team
Version: 1.0.0
Date: January 31, 2026
"""

import numpy as np
import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Set, Any
from enum import Enum, auto
from abc import ABC, abstractmethod
import time

# Golden ratio for tongue weights
PHI: float = (1 + np.sqrt(5)) / 2

# Pythagorean comma for toroidal edge costs
PYTHAGOREAN_COMMA: float = 531441 / 524288  # ~1.0136

# Default torus periods
DEFAULT_PERIOD: int = 3


# =============================================================================
# SECTION 1: DATA STRUCTURES
# =============================================================================

class TorusDimension(Enum):
    """Named dimensions on the 4D torus."""
    SPATIAL_X = 0    # x-position (spatial)
    SPATIAL_Y = 1    # y-position (spatial)
    SPATIAL_Z = 2    # z-position (spatial)
    TEMPORAL = 3     # t-position (causal/temporal winding)


@dataclass
class TorusPoint:
    """
    A point on the n-dimensional torus T^n.

    Coordinates are angles in [0, 2*pi) for each circle S^1.
    """
    angles: np.ndarray  # Angles in radians [0, 2*pi)
    dimension: int = 4

    def __post_init__(self):
        # Normalize angles to [0, 2*pi)
        self.angles = np.mod(self.angles, 2 * np.pi)
        if len(self.angles) != self.dimension:
            raise ValueError(f"Expected {self.dimension} angles, got {len(self.angles)}")

    def to_cartesian(self) -> np.ndarray:
        """
        Embed torus in higher-dimensional Euclidean space.

        For T^n embedded in R^{2n}, each angle theta_i maps to (cos(theta_i), sin(theta_i)).
        """
        result = []
        for theta in self.angles:
            result.extend([np.cos(theta), np.sin(theta)])
        return np.array(result)

    def to_normalized(self) -> np.ndarray:
        """Normalize angles to [-1, 1] range."""
        return (self.angles / np.pi) - 1.0


@dataclass
class TorusNode:
    """A node in a torus-embedded graph."""
    id: str
    position_3d: np.ndarray  # Original 3D position
    layer: int = 0           # Winding number in 4th dimension
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def position_4d(self) -> np.ndarray:
        """Full 4D position including layer as angle."""
        layer_angle = (self.layer / DEFAULT_PERIOD) * 2 * np.pi
        return np.append(self.position_3d, layer_angle)


@dataclass
class TorusEdge:
    """An edge in a torus-embedded graph."""
    source: str
    target: str
    weight: float = 1.0
    is_wrap: bool = False  # True if this is a toroidal wrap-around edge


# =============================================================================
# SECTION 2: TORUS GEOMETRY
# =============================================================================

class HyperTorus:
    """
    N-dimensional torus T^n = S^1 x S^1 x ... x S^1.

    The flat torus metric makes this a Riemannian manifold with
    zero curvature (unlike the Poincare ball's negative curvature).
    """

    def __init__(self, dimension: int = 4, radii: Optional[np.ndarray] = None):
        """
        Initialize hyper-torus.

        Args:
            dimension: Number of S^1 circles (default 4 for T^4)
            radii: Radius of each circle (default all 1.0)
        """
        self.dim = dimension
        self.radii = radii if radii is not None else np.ones(dimension)

    def geodesic_distance(self, p: TorusPoint, q: TorusPoint) -> float:
        """
        Compute geodesic distance on the flat torus.

        For the flat torus with metric ds^2 = sum(r_i^2 * d(theta_i)^2),
        the geodesic distance accounts for periodic boundaries.

        The distance in each dimension is min(|delta|, 2*pi - |delta|).
        """
        if p.dimension != self.dim or q.dimension != self.dim:
            raise ValueError("Point dimensions must match torus dimension")

        total_sq = 0.0
        for i in range(self.dim):
            delta = abs(p.angles[i] - q.angles[i])
            # Account for periodicity: take shorter path around circle
            delta = min(delta, 2 * np.pi - delta)
            total_sq += (self.radii[i] * delta) ** 2

        return np.sqrt(total_sq)

    def parallel_transport(
        self,
        vector: np.ndarray,
        path_start: TorusPoint,
        path_end: TorusPoint
    ) -> np.ndarray:
        """
        Parallel transport a vector along geodesic.

        On a flat torus, parallel transport is trivial (no rotation),
        but phase accumulates based on path winding.
        """
        # Compute winding number for each dimension
        winding = np.round((path_end.angles - path_start.angles) / (2 * np.pi))

        # Phase accumulation (holonomy)
        phase = np.sum(winding * self.radii)

        # For flat torus, vector unchanged but we track phase
        return vector  # Trivial transport on flat torus

    def wrap_coordinate(self, angle: float) -> float:
        """Wrap angle to [0, 2*pi)."""
        return np.mod(angle, 2 * np.pi)

    def is_on_torus(self, point: TorusPoint) -> bool:
        """Check if point is validly on the torus."""
        return all(0 <= a < 2 * np.pi for a in point.angles)


# =============================================================================
# SECTION 3: SACRED TONGUE TORUS EMBEDDING
# =============================================================================

class SacredTongueTorus:
    """
    Embed the Six Sacred Tongues on a 6-torus T^6.

    Each tongue corresponds to one S^1 factor, with phase angles
    determining the tongue's "activation state".
    """

    TONGUE_PHASES = {
        'KO': 0,                  # 0 degrees
        'AV': np.pi / 3,          # 60 degrees
        'RU': 2 * np.pi / 3,      # 120 degrees
        'CA': np.pi,              # 180 degrees
        'UM': 4 * np.pi / 3,      # 240 degrees
        'DR': 5 * np.pi / 3,      # 300 degrees
    }

    TONGUE_WEIGHTS = {
        'KO': PHI ** 0,  # 1.000
        'AV': PHI ** 1,  # 1.618
        'RU': PHI ** 2,  # 2.618
        'CA': PHI ** 3,  # 4.236
        'UM': PHI ** 4,  # 6.854
        'DR': PHI ** 5,  # 11.090
    }

    def __init__(self):
        self.torus = HyperTorus(dimension=6)
        self.tongues = list(self.TONGUE_PHASES.keys())

    def embed_tongues(self, activations: Dict[str, float]) -> TorusPoint:
        """
        Create torus point from tongue activations.

        Args:
            activations: Dict mapping tongue name to activation [0, 1]

        Returns:
            Point on T^6 where each angle = base_phase + activation * 2*pi
        """
        angles = []
        for tongue in self.tongues:
            base_phase = self.TONGUE_PHASES[tongue]
            activation = activations.get(tongue, 0.0)
            angle = base_phase + activation * 2 * np.pi
            angles.append(angle)

        return TorusPoint(np.array(angles), dimension=6)

    def tongue_distance(
        self,
        state1: Dict[str, float],
        state2: Dict[str, float]
    ) -> float:
        """Compute distance between two tongue activation states."""
        p1 = self.embed_tongues(state1)
        p2 = self.embed_tongues(state2)
        return self.torus.geodesic_distance(p1, p2)

    def project_to_t4(self, t6_point: TorusPoint) -> TorusPoint:
        """
        Project T^6 to T^4 by combining tongue pairs.

        Mapping:
            KO + AV -> Spatial X (control + transport)
            RU + CA -> Spatial Y (policy + compute)
            UM + DR -> Temporal (security + schema)
            Mean of all -> Spatial Z
        """
        angles = t6_point.angles

        # Combine pairs
        spatial_x = (angles[0] + angles[1]) / 2  # KO + AV
        spatial_y = (angles[2] + angles[3]) / 2  # RU + CA
        temporal = (angles[4] + angles[5]) / 2   # UM + DR
        spatial_z = np.mean(angles)              # All

        return TorusPoint(np.array([spatial_x, spatial_y, spatial_z, temporal]))


# =============================================================================
# SECTION 4: TORUS LIFT FOR DEAD-END RESOLUTION
# =============================================================================

class TorusGraph:
    """
    Graph structure embedded on the hyper-torus.

    Used for lifting 3D polyhedral graphs to 4D when Hamiltonian
    paths don't exist in 3D.
    """

    def __init__(self):
        self.nodes: Dict[str, TorusNode] = {}
        self.edges: List[TorusEdge] = []
        self.adjacency: Dict[str, Set[str]] = {}

    def add_node(self, node: TorusNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.adjacency:
            self.adjacency[node.id] = set()

    def add_edge(self, edge: TorusEdge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        if edge.source not in self.adjacency:
            self.adjacency[edge.source] = set()
        if edge.target not in self.adjacency:
            self.adjacency[edge.target] = set()
        self.adjacency[edge.source].add(edge.target)
        self.adjacency[edge.target].add(edge.source)

    def has_hamiltonian_path(self) -> bool:
        """
        Heuristic check for Hamiltonian path existence.

        Uses Ore's theorem approximation: if deg(u) + deg(v) >= n
        for all non-adjacent u, v, then Hamiltonian cycle exists.
        """
        n = len(self.nodes)
        if n < 2:
            return True

        # Check minimum degree condition (necessary but not sufficient)
        min_degree = min(len(self.adjacency[nid]) for nid in self.nodes)
        if min_degree < 1:
            return False  # Isolated node

        # Ore's theorem check (sufficient condition)
        node_ids = list(self.nodes.keys())
        for i, u in enumerate(node_ids):
            for v in node_ids[i+1:]:
                if v not in self.adjacency[u]:
                    deg_u = len(self.adjacency[u])
                    deg_v = len(self.adjacency[v])
                    if deg_u + deg_v < n:
                        return False  # Fails Ore condition

        return True

    def find_path_greedy(self, start: str, end: str) -> Optional[List[str]]:
        """Greedy path finding (for demo purposes)."""
        if start not in self.nodes or end not in self.nodes:
            return None

        visited = {start}
        path = [start]
        current = start

        while current != end:
            neighbors = self.adjacency.get(current, set()) - visited
            if not neighbors:
                return None  # Dead end

            # Choose closest unvisited neighbor
            next_node = min(neighbors, key=lambda n:
                np.linalg.norm(
                    self.nodes[n].position_4d - self.nodes[end].position_4d
                )
            )
            visited.add(next_node)
            path.append(next_node)
            current = next_node

        return path


class TorusLift:
    """
    Lift 3D graphs to 4D toroidal space to resolve Hamiltonian dead-ends.

    When a 3D polyhedral subgraph (e.g., unbalanced bipartite) has no
    Hamiltonian path, we lift it to T^4 by:
    1. Duplicating nodes across periodic 4th dimension (layers)
    2. Adding wrap-around edges between layers
    3. Finding path in lifted graph
    4. Projecting solution back to 3D
    """

    def __init__(self, max_period: int = DEFAULT_PERIOD):
        """
        Initialize torus lift.

        Args:
            max_period: Number of layers in 4th dimension
        """
        self.max_period = max_period
        self.torus = HyperTorus(dimension=4)

    def lift(self, graph_3d: TorusGraph) -> TorusGraph:
        """
        Lift a 3D graph to 4D toroidal space.

        Args:
            graph_3d: Original 3D graph (layer=0 for all nodes)

        Returns:
            Lifted 4D graph with periodic boundaries in 4th dimension
        """
        # Check if lift is needed
        if graph_3d.has_hamiltonian_path():
            return graph_3d  # No lift needed

        lifted = TorusGraph()

        # Step 1: Duplicate nodes across k layers
        for node_id, node in graph_3d.nodes.items():
            for k in range(self.max_period):
                new_id = f"{node_id}_L{k}"
                new_node = TorusNode(
                    id=new_id,
                    position_3d=node.position_3d.copy(),
                    layer=k,
                    attributes={
                        **node.attributes,
                        'original_id': node_id,
                        'layer': k
                    }
                )
                lifted.add_node(new_node)

        # Step 2: Add original edges within each layer
        for edge in graph_3d.edges:
            for k in range(self.max_period):
                new_edge = TorusEdge(
                    source=f"{edge.source}_L{k}",
                    target=f"{edge.target}_L{k}",
                    weight=edge.weight,
                    is_wrap=False
                )
                lifted.add_edge(new_edge)

        # Step 3: Add toroidal wrap-around edges between layers
        for node_id in graph_3d.nodes:
            for k in range(self.max_period):
                curr_id = f"{node_id}_L{k}"
                next_k = (k + 1) % self.max_period
                prev_k = (k - 1) % self.max_period

                # Forward wrap (comma-weighted)
                lifted.add_edge(TorusEdge(
                    source=curr_id,
                    target=f"{node_id}_L{next_k}",
                    weight=PYTHAGOREAN_COMMA * (k + 1),
                    is_wrap=True
                ))

                # Backward wrap
                lifted.add_edge(TorusEdge(
                    source=curr_id,
                    target=f"{node_id}_L{prev_k}",
                    weight=PYTHAGOREAN_COMMA * (k + 1),
                    is_wrap=True
                ))

        return lifted

    def resolve_path(
        self,
        graph_3d: TorusGraph,
        start: str,
        end: str
    ) -> Tuple[bool, Optional[List[str]], str]:
        """
        Attempt to find path, lifting to 4D if needed.

        Returns:
            (success, path, method) where method is '3D' or '4D_LIFT'
        """
        # Try 3D first
        path_3d = graph_3d.find_path_greedy(start, end)
        if path_3d:
            return (True, path_3d, '3D')

        # Lift to 4D
        lifted = self.lift(graph_3d)

        # Try to find path in lifted graph (any layer)
        for start_k in range(self.max_period):
            for end_k in range(self.max_period):
                start_4d = f"{start}_L{start_k}"
                end_4d = f"{end}_L{end_k}"

                path_4d = lifted.find_path_greedy(start_4d, end_4d)
                if path_4d:
                    # Project back to 3D (strip layer suffixes)
                    path_3d_proj = [p.split('_L')[0] for p in path_4d]
                    return (True, path_3d_proj, '4D_LIFT')

        return (False, None, 'FAILED')


# =============================================================================
# SECTION 5: MIRROR SYMMETRY KEY SWAPPING
# =============================================================================

class MirrorSymmetryKeySwapper:
    """
    Use Calabi-Yau mirror symmetry duality for cryptographic key swapping.

    Inspired by string theory's mirror symmetry where dual Calabi-Yau
    manifolds X and Y produce equivalent physics despite different
    geometries (Hodge numbers swapped: h^{1,1}(X) = h^{2,1}(Y)).

    Application:
        - Keys are "moduli" on primary manifold X
        - Mirror map transforms to dual manifold Y
        - Swapped keys work equivalently but resist interception
    """

    def __init__(
        self,
        h11_primary: int = 1,    # Kahler moduli (e.g., quintic: 1)
        h21_primary: int = 101   # Complex structure moduli (quintic: 101)
    ):
        """
        Initialize mirror key swapper.

        Args:
            h11_primary: h^{1,1} of primary Calabi-Yau
            h21_primary: h^{2,1} of primary Calabi-Yau
        """
        self.h11_primary = h11_primary
        self.h21_primary = h21_primary

        # Mirror has swapped Hodge numbers
        self.h11_mirror = h21_primary
        self.h21_mirror = h11_primary

        # Internal state
        self._mirror_salt = secrets.token_bytes(32)

    def generate_primary_key(
        self,
        context_vector: np.ndarray,
        master_secret: bytes
    ) -> bytes:
        """
        Generate key on primary Calabi-Yau (Kahler moduli embedding).

        Args:
            context_vector: 6D context from user state
            master_secret: Master key material

        Returns:
            32-byte derived key
        """
        # Project context to Kahler moduli space
        kappa = self._embed_kahler(context_vector)

        # Derive key from moduli coordinates
        key_material = master_secret + kappa.tobytes()
        return hashlib.sha256(key_material).digest()

    def _embed_kahler(self, context: np.ndarray) -> np.ndarray:
        """
        Embed context into Kahler moduli space.

        For a Calabi-Yau threefold, Kahler moduli parametrize
        volumes of 2-cycles (divisors).
        """
        # Simplified: use first h11 components weighted by phi
        weights = np.array([PHI ** i for i in range(self.h11_primary)])
        moduli = np.zeros(self.h11_primary)

        for i in range(min(len(context), self.h11_primary)):
            moduli[i] = context[i] * weights[i % len(weights)]

        return moduli

    def mirror_transform(self, primary_key: bytes) -> bytes:
        """
        Apply mirror map to transform key to dual manifold.

        The mirror map exchanges:
            - Kahler moduli <-> Complex structure moduli
            - A-model (worldsheet instantons) <-> B-model (classical)

        This is approximated by a cryptographic transformation that
        preserves key properties but changes representation.
        """
        # Combine with mirror salt for unique transformation
        mirror_input = primary_key + self._mirror_salt

        # Apply "inversion" (approximating SYZ T-duality)
        # Real mirror maps are much more complex, but this captures the idea
        inverted = bytes([255 - b for b in mirror_input[:32]])

        # Hash to get clean 32-byte output
        mirror_key = hashlib.sha256(inverted).digest()

        return mirror_key

    def swap_keys(
        self,
        context_vector: np.ndarray,
        master_secret: bytes
    ) -> Tuple[bytes, bytes]:
        """
        Generate key pair: primary and mirror-swapped.

        Returns:
            (primary_key, mirror_key) - both valid, related by duality
        """
        primary = self.generate_primary_key(context_vector, master_secret)
        mirror = self.mirror_transform(primary)

        return (primary, mirror)

    def verify_duality(
        self,
        received_key: bytes,
        context_vector: np.ndarray,
        master_secret: bytes,
        is_mirror: bool = False
    ) -> bool:
        """
        Verify a received key matches expected (primary or mirror).

        Args:
            received_key: Key received for verification
            context_vector: Context from sender
            master_secret: Shared master secret
            is_mirror: True if expecting mirror key

        Returns:
            True if valid, False otherwise
        """
        primary, mirror = self.swap_keys(context_vector, master_secret)
        expected = mirror if is_mirror else primary

        return secrets.compare_digest(received_key, expected)

    def fail_to_noise(self) -> bytes:
        """Return cryptographically random noise on verification failure."""
        return secrets.token_bytes(32)


# =============================================================================
# SECTION 6: INTEGRATION WITH POINCARE BALL
# =============================================================================

class HybridGeometry:
    """
    Unified geometry combining Poincare ball and Hyper-Torus.

    - Poincare Ball: Primary containment (adversarial blocking)
    - Hyper-Torus: Escape hatch (resolve dead-ends)

    Switching criterion: If path_cost -> infinity in hyperbolic space,
    lift to toroidal space for resolution.
    """

    def __init__(self):
        self.torus = HyperTorus(dimension=4)
        self.lifter = TorusLift(max_period=3)
        self.mirror = MirrorSymmetryKeySwapper()
        self.tongue_torus = SacredTongueTorus()

    def hyperbolic_to_torus(
        self,
        poincare_point: np.ndarray,
        curvature: float = 1.0
    ) -> TorusPoint:
        """
        Map Poincare ball point to torus.

        Uses stereographic-like projection: points near center map to
        torus center, points near boundary map to torus equator.
        """
        # Compute hyperbolic "radius"
        norm = np.linalg.norm(poincare_point)

        # Map norm to torus radial coordinate
        # norm in [0, 1) -> angle in [0, pi]
        radial_angle = np.arctan(norm / (1 - norm + 1e-10))

        # Distribute across torus dimensions
        if len(poincare_point) >= 4:
            angles = poincare_point[:4] / (norm + 1e-10) * radial_angle
        else:
            angles = np.zeros(4)
            angles[:len(poincare_point)] = poincare_point / (norm + 1e-10) * radial_angle

        return TorusPoint(angles)

    def torus_to_hyperbolic(
        self,
        torus_point: TorusPoint,
        curvature: float = 1.0
    ) -> np.ndarray:
        """
        Map torus point back to Poincare ball.

        Inverse of hyperbolic_to_torus.
        """
        # Compute torus "radius" (distance from origin on flat torus)
        radial = np.sqrt(np.sum(torus_point.angles ** 2))

        # Map back to hyperbolic radius
        norm = np.tanh(radial / 2)

        # Direction from angles
        if radial > 1e-10:
            direction = torus_point.angles / radial
        else:
            direction = np.zeros(4)

        return direction * norm

    def compute_combined_distance(
        self,
        u: np.ndarray,
        v: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute distances in both geometries.

        Returns dict with hyperbolic and toroidal distances.
        """
        # Hyperbolic distance (Poincare)
        u_norm_sq = np.clip(np.dot(u, u), 0, 1 - 1e-10)
        v_norm_sq = np.clip(np.dot(v, v), 0, 1 - 1e-10)
        diff_sq = np.dot(u - v, u - v)
        denom = (1 - u_norm_sq) * (1 - v_norm_sq)

        if denom > 1e-10:
            delta = 2 * diff_sq / denom
            d_hyperbolic = float(np.arccosh(1 + delta))
        else:
            d_hyperbolic = float('inf')

        # Toroidal distance
        u_torus = self.hyperbolic_to_torus(u)
        v_torus = self.hyperbolic_to_torus(v)
        d_torus = self.torus.geodesic_distance(u_torus, v_torus)

        return {
            'hyperbolic': d_hyperbolic,
            'toroidal': d_torus,
            'ratio': d_torus / (d_hyperbolic + 1e-10)
        }

    def should_lift(
        self,
        hyperbolic_cost: float,
        threshold: float = 100.0
    ) -> bool:
        """
        Determine if we should lift to torus to escape dead-end.

        Args:
            hyperbolic_cost: Cost/distance in hyperbolic space
            threshold: Cost threshold triggering lift

        Returns:
            True if lift recommended
        """
        return hyperbolic_cost > threshold or hyperbolic_cost == float('inf')


# =============================================================================
# SECTION 7: DEMO & TESTING
# =============================================================================

def demo():
    """Demonstrate hyper-torus functionality."""
    print("=" * 70)
    print("SCBE-AETHERMOORE HYPER-TORUS MODULE DEMO")
    print("T^4 = S^1 x S^1 x S^1 x S^1 (4-Dimensional Torus)")
    print("=" * 70)

    # 1. Basic Torus Operations
    print("\n1. BASIC TORUS GEOMETRY")
    print("-" * 70)

    torus = HyperTorus(dimension=4)
    p1 = TorusPoint(np.array([0.0, 0.0, 0.0, 0.0]))
    p2 = TorusPoint(np.array([np.pi, 0.0, 0.0, 0.0]))
    p3 = TorusPoint(np.array([0.0, 0.0, 0.0, np.pi]))

    print(f"   Point 1: {p1.angles}")
    print(f"   Point 2: {p2.angles}")
    print(f"   Point 3: {p3.angles}")
    print(f"   Distance(P1, P2): {torus.geodesic_distance(p1, p2):.4f} (spatial)")
    print(f"   Distance(P1, P3): {torus.geodesic_distance(p1, p3):.4f} (temporal)")

    # 2. Sacred Tongue Torus
    print("\n2. SACRED TONGUE EMBEDDING ON T^6")
    print("-" * 70)

    tongue_torus = SacredTongueTorus()

    state1 = {'KO': 0.5, 'AV': 0.3, 'RU': 0.7, 'CA': 0.2, 'UM': 0.8, 'DR': 0.1}
    state2 = {'KO': 0.5, 'AV': 0.3, 'RU': 0.7, 'CA': 0.2, 'UM': 0.2, 'DR': 0.9}

    print(f"   State 1: {state1}")
    print(f"   State 2: {state2}")
    print(f"   Tongue Distance: {tongue_torus.tongue_distance(state1, state2):.4f}")

    t6_point = tongue_torus.embed_tongues(state1)
    t4_point = tongue_torus.project_to_t4(t6_point)
    print(f"   T^6 Point: {t6_point.angles[:3]}... (first 3)")
    print(f"   T^4 Projection: {t4_point.angles}")

    # 3. Torus Lift Demo
    print("\n3. TORUS LIFT FOR DEAD-END RESOLUTION")
    print("-" * 70)

    # Create a simple 3D graph that has no easy path
    graph = TorusGraph()

    # Triangle with one disconnected node
    graph.add_node(TorusNode("A", np.array([0, 0, 0])))
    graph.add_node(TorusNode("B", np.array([1, 0, 0])))
    graph.add_node(TorusNode("C", np.array([0.5, 0.866, 0])))
    graph.add_node(TorusNode("D", np.array([0.5, 0.3, 0.5])))  # Poorly connected

    graph.add_edge(TorusEdge("A", "B"))
    graph.add_edge(TorusEdge("B", "C"))
    graph.add_edge(TorusEdge("C", "A"))
    # D is only connected to C
    graph.add_edge(TorusEdge("C", "D"))

    print(f"   Graph nodes: {list(graph.nodes.keys())}")
    print(f"   Graph edges: {[(e.source, e.target) for e in graph.edges]}")
    print(f"   Has Hamiltonian path (3D): {graph.has_hamiltonian_path()}")

    lifter = TorusLift(max_period=3)
    success, path, method = lifter.resolve_path(graph, "A", "D")

    print(f"   Path found: {success}")
    print(f"   Method used: {method}")
    if path:
        print(f"   Path: {' -> '.join(path)}")

    # 4. Mirror Symmetry Key Swapping
    print("\n4. MIRROR SYMMETRY KEY SWAPPING")
    print("-" * 70)

    mirror = MirrorSymmetryKeySwapper(h11_primary=1, h21_primary=101)

    context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
    master_secret = b"super_secret_master_key_32bytes!"

    primary_key, mirror_key = mirror.swap_keys(context, master_secret)

    print(f"   Context vector: {context}")
    print(f"   Primary Hodge: h^{{1,1}}={mirror.h11_primary}, h^{{2,1}}={mirror.h21_primary}")
    print(f"   Mirror Hodge:  h^{{1,1}}={mirror.h11_mirror}, h^{{2,1}}={mirror.h21_mirror}")
    print(f"   Primary key:  {primary_key.hex()[:32]}...")
    print(f"   Mirror key:   {mirror_key.hex()[:32]}...")
    print(f"   Keys differ:  {primary_key != mirror_key}")

    # Verify duality
    valid_primary = mirror.verify_duality(primary_key, context, master_secret, is_mirror=False)
    valid_mirror = mirror.verify_duality(mirror_key, context, master_secret, is_mirror=True)
    wrong_key = mirror.verify_duality(b"x" * 32, context, master_secret, is_mirror=False)

    print(f"   Verify primary: {valid_primary}")
    print(f"   Verify mirror:  {valid_mirror}")
    print(f"   Verify wrong:   {wrong_key}")

    # 5. Hybrid Geometry
    print("\n5. HYBRID POINCARE-TORUS GEOMETRY")
    print("-" * 70)

    hybrid = HybridGeometry()

    # Point near center (safe)
    safe_point = np.array([0.1, 0.1, 0.1, 0.1, 0.0, 0.0])
    # Point near boundary (dangerous)
    danger_point = np.array([0.95, 0.0, 0.0, 0.0, 0.0, 0.0])

    safe_distances = hybrid.compute_combined_distance(
        safe_point[:4], np.zeros(4)
    )
    danger_distances = hybrid.compute_combined_distance(
        danger_point[:4], np.zeros(4)
    )

    print(f"   Safe point ({np.linalg.norm(safe_point[:4]):.2f} from center):")
    print(f"      Hyperbolic distance: {safe_distances['hyperbolic']:.4f}")
    print(f"      Toroidal distance:   {safe_distances['toroidal']:.4f}")
    print(f"      Should lift: {hybrid.should_lift(safe_distances['hyperbolic'])}")

    print(f"\n   Danger point ({np.linalg.norm(danger_point[:4]):.2f} from center):")
    print(f"      Hyperbolic distance: {danger_distances['hyperbolic']:.4f}")
    print(f"      Toroidal distance:   {danger_distances['toroidal']:.4f}")
    print(f"      Should lift: {hybrid.should_lift(danger_distances['hyperbolic'])}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    Hyper-Torus T^4 provides:
    1. Escape hatch for 3D dead-ends (torus lift)
    2. Periodic boundaries (wrap-around paths)
    3. Sacred Tongue embedding on T^6 -> T^4
    4. Mirror symmetry key swapping (Calabi-Yau inspired)
    5. Hybrid Poincare-Torus geometry switching

    "When hyperbolic space blocks, the torus provides a way around."
    """)


if __name__ == "__main__":
    demo()
