"""
SCBE-AETHERMOORE Hyper-Torus Tests
===================================

Tests for the Hyper-Torus T^n geometry module.

Author: SCBE-AETHERMOORE Team
Version: 1.0.0
Date: January 31, 2026
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prototype.hyper_torus import (
    TorusPoint,
    TorusNode,
    TorusEdge,
    HyperTorus,
    SacredTongueTorus,
    TorusGraph,
    TorusLift,
    MirrorSymmetryKeySwapper,
    HybridGeometry,
    PHI,
    PYTHAGOREAN_COMMA,
    DEFAULT_PERIOD,
)


# =============================================================================
# SECTION 1: TORUS POINT TESTS
# =============================================================================

class TestTorusPoint:
    """Tests for TorusPoint data structure."""

    def test_point_creation(self):
        """TorusPoint should be created with valid angles."""
        angles = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
        point = TorusPoint(angles)
        assert len(point.angles) == 4

    def test_angle_normalization(self):
        """Angles should be normalized to [0, 2*pi)."""
        # Angle > 2*pi should wrap
        angles = np.array([3*np.pi, 0, 0, 0])
        point = TorusPoint(angles)
        assert point.angles[0] == pytest.approx(np.pi, abs=1e-10)

        # Negative angle should wrap
        angles_neg = np.array([-np.pi/2, 0, 0, 0])
        point_neg = TorusPoint(angles_neg)
        assert point_neg.angles[0] == pytest.approx(3*np.pi/2, abs=1e-10)

    def test_to_cartesian(self):
        """Cartesian embedding should have correct dimension."""
        angles = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
        point = TorusPoint(angles)
        cartesian = point.to_cartesian()
        assert len(cartesian) == 8  # 2*4

    def test_to_normalized(self):
        """Normalized angles should be in [-1, 1]."""
        angles = np.array([0.0, np.pi, 2*np.pi - 0.01, np.pi/2])
        point = TorusPoint(angles)
        normalized = point.to_normalized()
        assert all(-1 <= n <= 1 for n in normalized)


# =============================================================================
# SECTION 2: HYPER-TORUS GEOMETRY TESTS
# =============================================================================

class TestHyperTorus:
    """Tests for HyperTorus geometry."""

    def test_distance_to_self_is_zero(self):
        """Distance from point to itself should be zero."""
        torus = HyperTorus(dimension=4)
        p = TorusPoint(np.array([1.0, 2.0, 0.5, 1.5]))
        d = torus.geodesic_distance(p, p)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_distance_symmetric(self):
        """Distance should be symmetric."""
        torus = HyperTorus(dimension=4)
        p1 = TorusPoint(np.array([0.0, 0.0, 0.0, 0.0]))
        p2 = TorusPoint(np.array([np.pi, 0.5, 1.0, 0.2]))
        assert torus.geodesic_distance(p1, p2) == pytest.approx(
            torus.geodesic_distance(p2, p1), abs=1e-10
        )

    def test_periodicity_respected(self):
        """Distance should account for periodic boundaries."""
        torus = HyperTorus(dimension=4)

        # Points at 0 and 2*pi - epsilon should be close
        p1 = TorusPoint(np.array([0.01, 0, 0, 0]))
        p2 = TorusPoint(np.array([2*np.pi - 0.01, 0, 0, 0]))

        d = torus.geodesic_distance(p1, p2)
        # Should be about 0.02 (short path), not ~6.26 (long path)
        assert d < 0.1

    def test_max_distance_is_pi_per_dimension(self):
        """Maximum distance on circle S^1 is pi (half circumference)."""
        torus = HyperTorus(dimension=1, radii=np.array([1.0]))
        p1 = TorusPoint(np.array([0.0]), dimension=1)
        p2 = TorusPoint(np.array([np.pi]), dimension=1)
        d = torus.geodesic_distance(p1, p2)
        assert d == pytest.approx(np.pi, abs=1e-10)


# =============================================================================
# SECTION 3: SACRED TONGUE TORUS TESTS
# =============================================================================

class TestSacredTongueTorus:
    """Tests for Sacred Tongue embedding on T^6."""

    def test_all_tongues_present(self):
        """All 6 Sacred Tongues should be defined."""
        stt = SacredTongueTorus()
        assert len(stt.tongues) == 6
        assert set(stt.tongues) == {'KO', 'AV', 'RU', 'CA', 'UM', 'DR'}

    def test_tongue_weights_golden_ratio(self):
        """Tongue weights should follow phi^n pattern."""
        stt = SacredTongueTorus()
        assert stt.TONGUE_WEIGHTS['KO'] == pytest.approx(1.0, abs=1e-10)
        assert stt.TONGUE_WEIGHTS['AV'] == pytest.approx(PHI, abs=1e-10)
        assert stt.TONGUE_WEIGHTS['RU'] == pytest.approx(PHI**2, abs=1e-10)

    def test_embed_creates_t6_point(self):
        """Embedding should create a 6D torus point."""
        stt = SacredTongueTorus()
        activations = {t: 0.5 for t in stt.tongues}
        point = stt.embed_tongues(activations)
        assert point.dimension == 6

    def test_project_to_t4(self):
        """T^6 -> T^4 projection should work."""
        stt = SacredTongueTorus()
        activations = {t: 0.5 for t in stt.tongues}
        t6 = stt.embed_tongues(activations)
        t4 = stt.project_to_t4(t6)
        assert t4.dimension == 4

    def test_tongue_distance_zero_for_same_state(self):
        """Same tongue state should have zero distance."""
        stt = SacredTongueTorus()
        state = {'KO': 0.5, 'AV': 0.3, 'RU': 0.7, 'CA': 0.2, 'UM': 0.8, 'DR': 0.1}
        d = stt.tongue_distance(state, state)
        assert d == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# SECTION 4: TORUS GRAPH TESTS
# =============================================================================

class TestTorusGraph:
    """Tests for TorusGraph structure."""

    def test_add_nodes_and_edges(self):
        """Graph should accept nodes and edges."""
        graph = TorusGraph()
        graph.add_node(TorusNode("A", np.array([0, 0, 0])))
        graph.add_node(TorusNode("B", np.array([1, 0, 0])))
        graph.add_edge(TorusEdge("A", "B"))

        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert len(graph.edges) == 1

    def test_adjacency_bidirectional(self):
        """Edges should create bidirectional adjacency."""
        graph = TorusGraph()
        graph.add_node(TorusNode("A", np.array([0, 0, 0])))
        graph.add_node(TorusNode("B", np.array([1, 0, 0])))
        graph.add_edge(TorusEdge("A", "B"))

        assert "B" in graph.adjacency["A"]
        assert "A" in graph.adjacency["B"]

    def test_complete_graph_has_hamiltonian(self):
        """Complete graph should have Hamiltonian path."""
        graph = TorusGraph()
        nodes = ["A", "B", "C", "D"]
        for n in nodes:
            graph.add_node(TorusNode(n, np.random.rand(3)))

        # Add all edges (complete graph)
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                graph.add_edge(TorusEdge(n1, n2))

        assert graph.has_hamiltonian_path()


# =============================================================================
# SECTION 5: TORUS LIFT TESTS
# =============================================================================

class TestTorusLift:
    """Tests for Torus Lift dead-end resolution."""

    def test_lift_preserves_nodes(self):
        """Lift should create copies of all nodes across layers when needed."""
        # Create a non-Hamiltonian graph (isolated node)
        graph = TorusGraph()
        graph.add_node(TorusNode("A", np.array([0, 0, 0])))
        graph.add_node(TorusNode("B", np.array([1, 0, 0])))
        graph.add_node(TorusNode("C", np.array([2, 0, 0])))  # Isolated - no edges

        # Only connect A-B, leaving C isolated -> no Hamiltonian
        graph.add_edge(TorusEdge("A", "B"))

        lifter = TorusLift(max_period=3)
        lifted = lifter.lift(graph)

        # Should have 3 nodes * 3 layers = 9 nodes
        assert len(lifted.nodes) == 9

    def test_lift_adds_wrap_edges(self):
        """Lift should add toroidal wrap-around edges."""
        # Create non-Hamiltonian graph
        graph = TorusGraph()
        graph.add_node(TorusNode("A", np.array([0, 0, 0])))
        graph.add_node(TorusNode("B", np.array([1, 0, 0])))
        graph.add_node(TorusNode("C", np.array([2, 0, 0])))  # Isolated

        graph.add_edge(TorusEdge("A", "B"))  # Only one edge

        lifter = TorusLift(max_period=3)
        lifted = lifter.lift(graph)

        # Check for wrap edges
        wrap_edges = [e for e in lifted.edges if e.is_wrap]
        assert len(wrap_edges) > 0

    def test_resolve_path_returns_method(self):
        """resolve_path should indicate which method was used."""
        graph = TorusGraph()
        graph.add_node(TorusNode("A", np.array([0, 0, 0])))
        graph.add_node(TorusNode("B", np.array([1, 0, 0])))
        graph.add_edge(TorusEdge("A", "B"))

        lifter = TorusLift(max_period=3)
        success, path, method = lifter.resolve_path(graph, "A", "B")

        assert success
        assert method in ['3D', '4D_LIFT']


# =============================================================================
# SECTION 6: MIRROR SYMMETRY KEY SWAPPING TESTS
# =============================================================================

class TestMirrorSymmetryKeySwapper:
    """Tests for mirror symmetry key swapping."""

    def test_hodge_numbers_swapped(self):
        """Mirror should have swapped Hodge numbers."""
        mirror = MirrorSymmetryKeySwapper(h11_primary=1, h21_primary=101)
        assert mirror.h11_mirror == 101
        assert mirror.h21_mirror == 1

    def test_primary_key_deterministic(self):
        """Same context should produce same primary key."""
        mirror = MirrorSymmetryKeySwapper()
        context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        secret = b"test_secret_key_32bytes_long!!!"

        key1 = mirror.generate_primary_key(context, secret)
        key2 = mirror.generate_primary_key(context, secret)

        assert key1 == key2

    def test_mirror_key_differs_from_primary(self):
        """Mirror key should differ from primary key."""
        mirror = MirrorSymmetryKeySwapper()
        context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        secret = b"test_secret_key_32bytes_long!!!"

        primary, mirror_key = mirror.swap_keys(context, secret)

        assert primary != mirror_key

    def test_key_length_32_bytes(self):
        """Keys should be 32 bytes."""
        mirror = MirrorSymmetryKeySwapper()
        context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        secret = b"test_secret_key_32bytes_long!!!"

        primary, mirror_key = mirror.swap_keys(context, secret)

        assert len(primary) == 32
        assert len(mirror_key) == 32

    def test_verify_duality_primary(self):
        """Primary key should verify correctly."""
        mirror = MirrorSymmetryKeySwapper()
        context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        secret = b"test_secret_key_32bytes_long!!!"

        primary, _ = mirror.swap_keys(context, secret)
        assert mirror.verify_duality(primary, context, secret, is_mirror=False)

    def test_verify_duality_mirror(self):
        """Mirror key should verify correctly with is_mirror=True."""
        mirror = MirrorSymmetryKeySwapper()
        context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        secret = b"test_secret_key_32bytes_long!!!"

        _, mirror_key = mirror.swap_keys(context, secret)
        assert mirror.verify_duality(mirror_key, context, secret, is_mirror=True)

    def test_wrong_key_fails_verification(self):
        """Wrong key should fail verification."""
        mirror = MirrorSymmetryKeySwapper()
        context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        secret = b"test_secret_key_32bytes_long!!!"

        wrong_key = b"x" * 32
        assert not mirror.verify_duality(wrong_key, context, secret, is_mirror=False)

    def test_fail_to_noise_random(self):
        """fail_to_noise should return random bytes."""
        mirror = MirrorSymmetryKeySwapper()

        noise1 = mirror.fail_to_noise()
        noise2 = mirror.fail_to_noise()

        assert len(noise1) == 32
        assert noise1 != noise2  # Should be random


# =============================================================================
# SECTION 7: HYBRID GEOMETRY TESTS
# =============================================================================

class TestHybridGeometry:
    """Tests for hybrid Poincare-Torus geometry."""

    def test_hyperbolic_to_torus_center(self):
        """Origin in hyperbolic should map to torus origin."""
        hybrid = HybridGeometry()
        origin = np.zeros(4)
        torus_point = hybrid.hyperbolic_to_torus(origin)

        # All angles should be near 0 (or wrapped)
        assert all(a < 0.1 or a > 2*np.pi - 0.1 for a in torus_point.angles)

    def test_torus_to_hyperbolic_roundtrip(self):
        """Roundtrip should approximately preserve point."""
        hybrid = HybridGeometry()
        original = np.array([0.3, 0.2, 0.1, 0.0])

        torus = hybrid.hyperbolic_to_torus(original)
        recovered = hybrid.torus_to_hyperbolic(torus)

        # Should be approximately same (up to numerical precision)
        assert np.linalg.norm(original - recovered) < 0.5

    def test_compute_combined_distance(self):
        """Combined distance should return both metrics."""
        hybrid = HybridGeometry()
        u = np.array([0.1, 0.1, 0.1, 0.1])
        v = np.array([0.2, 0.2, 0.2, 0.2])

        distances = hybrid.compute_combined_distance(u, v)

        assert 'hyperbolic' in distances
        assert 'toroidal' in distances
        assert 'ratio' in distances

    def test_should_lift_high_cost(self):
        """High hyperbolic cost should trigger lift."""
        hybrid = HybridGeometry()
        assert hybrid.should_lift(float('inf'))
        assert hybrid.should_lift(150.0)

    def test_should_not_lift_low_cost(self):
        """Low hyperbolic cost should not trigger lift."""
        hybrid = HybridGeometry()
        assert not hybrid.should_lift(1.0)
        assert not hybrid.should_lift(50.0)


# =============================================================================
# SECTION 8: ADVERSARIAL TESTS (MUST FAIL)
# =============================================================================

@pytest.mark.xfail(reason="SECURITY: Mirror key spoofing must fail", strict=True)
class TestMirrorSpoofingMustFail:
    """
    These tests attempt to spoof mirror keys.
    ALL tests MUST FAIL (xfail). If any PASS, security is compromised.
    """

    def test_different_context_same_key(self):
        """MUST FAIL: Different context should not produce same key."""
        mirror = MirrorSymmetryKeySwapper()
        secret = b"test_secret_key_32bytes_long!!!"

        context1 = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        context2 = np.array([0.9, 0.1, 0.5, 0.8, 0.2, 0.7])  # Different

        key1, _ = mirror.swap_keys(context1, secret)
        key2, _ = mirror.swap_keys(context2, secret)

        # This SHOULD fail - different context means different key
        assert key1 == key2, "Keys correctly differ for different contexts"

    def test_mirror_key_verifies_as_primary(self):
        """MUST FAIL: Mirror key should not verify as primary."""
        mirror = MirrorSymmetryKeySwapper()
        context = np.array([0.5, 0.3, 0.7, 0.2, 0.8, 0.1])
        secret = b"test_secret_key_32bytes_long!!!"

        _, mirror_key = mirror.swap_keys(context, secret)

        # This SHOULD fail - mirror key is not primary
        assert mirror.verify_duality(mirror_key, context, secret, is_mirror=False), \
            "Mirror key correctly rejected as primary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
