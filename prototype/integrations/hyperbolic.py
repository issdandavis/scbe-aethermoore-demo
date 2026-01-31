"""
Hyperbolic Geometry Integration using geoopt.

Drop-in replacement for manual Poincaré ball implementation in toy_phdm.py.
Uses geoopt for numerically stable hyperbolic operations.

Reference: https://github.com/geoopt/geoopt
"""

import numpy as np

try:
    import torch
    import geoopt
    GEOOPT_AVAILABLE = True
except ImportError:
    GEOOPT_AVAILABLE = False
    print("Warning: geoopt not installed. Using fallback implementation.")


# Golden ratio for tongue weights
PHI = (1 + np.sqrt(5)) / 2


class GeooptPoincareBall:
    """
    Poincaré Ball manifold wrapper using geoopt.

    Provides the same interface as ToyPHDM's hyperbolic methods
    but uses geoopt's numerically stable implementation.
    """

    def __init__(self, dim: int = 2, curvature: float = 1.0):
        """
        Initialize the Poincaré ball.

        Args:
            dim: Dimension of the ball (2 for visualization, 6 for production)
            curvature: Negative curvature parameter (default c=1 gives κ=-1)
        """
        self.dim = dim
        self.curvature = curvature

        if GEOOPT_AVAILABLE:
            self.manifold = geoopt.PoincareBall(c=curvature)
        else:
            self.manifold = None

    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute hyperbolic distance between two points.

        Uses geoopt's implementation which handles numerical edge cases.
        """
        if GEOOPT_AVAILABLE:
            u_t = torch.tensor(u, dtype=torch.float64)
            v_t = torch.tensor(v, dtype=torch.float64)
            return self.manifold.dist(u_t, v_t).item()
        else:
            return self._fallback_distance(u, v)

    def project(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """
        Project point onto the Poincaré ball (clamp to radius < 1).

        Args:
            x: Point to project
            eps: Distance from boundary to clamp to
        """
        if GEOOPT_AVAILABLE:
            x_t = torch.tensor(x, dtype=torch.float64)
            projected = self.manifold.projx(x_t)
            return projected.numpy()
        else:
            norm = np.linalg.norm(x)
            if norm >= 1.0:
                return x * (1.0 - eps) / norm
            return x

    def expmap(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map: move from x in direction v (tangent vector).

        This is the Möbius addition that respects hyperbolic geometry.
        """
        if GEOOPT_AVAILABLE:
            x_t = torch.tensor(x, dtype=torch.float64)
            v_t = torch.tensor(v, dtype=torch.float64)
            result = self.manifold.expmap(x_t, v_t)
            return result.numpy()
        else:
            return self._fallback_mobius_add(x, v)

    def logmap(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: compute tangent vector from x to y.

        Inverse of expmap.
        """
        if GEOOPT_AVAILABLE:
            x_t = torch.tensor(x, dtype=torch.float64)
            y_t = torch.tensor(y, dtype=torch.float64)
            result = self.manifold.logmap(x_t, y_t)
            return result.numpy()
        else:
            return y - x  # Euclidean fallback

    def geodesic(self, x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
        """
        Compute point along geodesic from x to y at parameter t.

        t=0 gives x, t=1 gives y.
        """
        v = self.logmap(x, y)
        return self.expmap(x, t * v)

    # Fallback implementations when geoopt is not available

    def _fallback_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Manual hyperbolic distance (less numerically stable)."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        norm_u_sq = np.dot(u, u)
        norm_v_sq = np.dot(v, v)

        # Clamp to avoid boundary issues
        norm_u_sq = min(norm_u_sq, 0.9999)
        norm_v_sq = min(norm_v_sq, 0.9999)

        diff = u - v
        diff_sq = np.dot(diff, diff)

        denominator = (1 - norm_u_sq) * (1 - norm_v_sq)
        if denominator <= 0:
            return float('inf')

        delta = 2 * diff_sq / denominator
        return np.arccosh(1 + delta)

    def _fallback_mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Manual Möbius addition."""
        x_sq = np.dot(x, x)
        y_sq = np.dot(y, y)
        xy = np.dot(x, y)

        num = (1 + 2 * xy + y_sq) * x + (1 - x_sq) * y
        denom = 1 + 2 * xy + x_sq * y_sq

        result = num / denom
        return self.project(result)


def create_tongue_manifold(dim: int = 2) -> tuple:
    """
    Create a Poincaré ball with the 6 Sacred Tongues positioned.

    Returns:
        (manifold, positions): The manifold and a dict of tongue positions
    """
    ball = GeooptPoincareBall(dim=dim)

    # Security radii (from toy_phdm.py)
    security_radius = {
        'KO': 0.0,   # Center - safest
        'AV': 0.2,   # Close - transport
        'RU': 0.25,  # Close - policy
        'CA': 0.4,   # Medium - compute
        'UM': 0.6,   # Far - security
        'DR': 0.75,  # Furthest - schema
    }

    # Phase angles (60° apart)
    phases = {
        'KO': 0,
        'AV': 60,
        'RU': 120,
        'CA': 180,
        'UM': 240,
        'DR': 300,
    }

    positions = {}
    for name in ['KO', 'AV', 'RU', 'CA', 'UM', 'DR']:
        radius = security_radius[name]
        if name == 'KO':
            positions[name] = np.zeros(dim)
        else:
            angle = np.radians(phases[name])
            pos = np.zeros(dim)
            pos[0] = radius * np.cos(angle)
            pos[1] = radius * np.sin(angle)
            positions[name] = pos

    return ball, positions


def harmonic_wall_cost(distance: float, base: float = np.e) -> float:
    """
    Compute Harmonic Wall cost H(d) = exp(d²).

    This creates exponential resistance to boundary approaches.
    """
    return base ** (distance ** 2)


def demo():
    """Demonstrate geoopt integration."""
    print("=" * 60)
    print("Geoopt Integration Demo")
    print("=" * 60)

    ball, positions = create_tongue_manifold()

    print(f"\nGeoopt available: {GEOOPT_AVAILABLE}")
    print("\nTongue positions:")
    for name, pos in positions.items():
        weight = PHI ** list(positions.keys()).index(name)
        print(f"  {name}: {pos} (weight: {weight:.3f})")

    print("\nHyperbolic distances from KO (center):")
    ko_pos = positions['KO']
    for name, pos in positions.items():
        if name != 'KO':
            dist = ball.distance(ko_pos, pos)
            cost = harmonic_wall_cost(dist)
            print(f"  KO → {name}: d={dist:.3f}, H(d)={cost:.2f}")

    print("\nPath costs (KO → DR):")
    # Direct path (blocked by adjacency in toy_phdm.py)
    direct_dist = ball.distance(positions['KO'], positions['DR'])
    print(f"  Direct: d={direct_dist:.3f}, H(d)={harmonic_wall_cost(direct_dist):.2f}")

    # Via CA (allowed path)
    via_av = ball.distance(positions['KO'], positions['AV'])
    via_ca = ball.distance(positions['AV'], positions['CA'])
    via_dr = ball.distance(positions['CA'], positions['DR'])
    total = via_av + via_ca + via_dr
    print(f"  Via AV→CA: d={total:.3f}, H(d)={harmonic_wall_cost(total):.2f}")


if __name__ == "__main__":
    demo()
