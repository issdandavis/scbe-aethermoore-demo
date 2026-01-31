"""
GeoSeal Gate - Trust Ring Validation using Poincare Ball

Validates context vectors against trust rings:
    - CORE (r < 0.3): Full access, low latency (5ms)
    - OUTER (0.3 <= r < 0.7): Read-only, high latency (200ms)
    - WALL (0.7 <= r < 0.9): Blocked, behavioral anomaly
    - EVENT_HORIZON (r >= 0.9): Critical security violation

The hyperbolic distance grows exponentially near the boundary,
making it impossible for attackers to "sneak" past the wall.
"""

import numpy as np
import time
import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

# Import from sibling modules
try:
    from .context_capture import ContextVector, create_baseline_vector
except ImportError:
    from context_capture import ContextVector, create_baseline_vector


class TrustRing(Enum):
    """Trust ring classifications."""
    CORE = "CORE"
    OUTER = "OUTER"
    WALL = "WALL"
    EVENT_HORIZON = "EVENT_HORIZON"


@dataclass
class AccessResult:
    """Result of trust ring validation."""
    allowed: bool
    ring: TrustRing
    distance: float
    latency: float = 0.0
    access_level: str = "none"
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'allowed': self.allowed,
            'ring': self.ring.value,
            'distance': self.distance,
            'latency': self.latency,
            'access_level': self.access_level,
            'reason': self.reason,
            'timestamp': self.timestamp,
        }


class PoincareBall:
    """
    Poincare Ball model for hyperbolic trust validation.

    The key property: distance grows exponentially near the boundary,
    creating a natural "wall" that's geometrically impossible to cross.
    """

    def __init__(self, dim: int = 6, curvature: float = 1.0):
        self.dim = dim
        self.c = curvature
        self.eps = 1e-10

    def embed(self, vector: List[float]) -> np.ndarray:
        """
        Embed a unit hypercube vector [0,1]^n into the Poincare ball.

        Maps [0,1] -> [-0.9, 0.9] to stay strictly inside the ball.
        """
        v = np.array(vector, dtype=np.float64)

        # Center and scale: [0,1] -> [-0.9, 0.9]
        v = (v - 0.5) * 1.8

        # Project to ball if needed
        norm = np.linalg.norm(v)
        if norm >= 1.0:
            v = v * 0.99 / norm

        return v

    def hyperbolic_distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute hyperbolic distance in the Poincare ball.

        d(u,v) = arccosh(1 + 2|u-v|^2 / ((1-|u|^2)(1-|v|^2)))

        This distance grows exponentially near the boundary.
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        norm_u_sq = np.clip(np.dot(u, u), 0, 1 - self.eps)
        norm_v_sq = np.clip(np.dot(v, v), 0, 1 - self.eps)

        diff_sq = np.dot(u - v, u - v)
        denominator = (1 - norm_u_sq) * (1 - norm_v_sq)

        if denominator <= self.eps:
            return float('inf')

        delta = 2 * diff_sq / denominator

        return float(np.arccosh(1 + delta))

    def radius(self, point: np.ndarray) -> float:
        """Get Euclidean radius of point (distance from origin)."""
        return float(np.linalg.norm(point))


class AetherAuthGate:
    """
    Main authentication gate using GeoSeal validation.

    Checks if a context vector falls within trusted rings
    and enforces appropriate access controls.
    """

    # Default configuration
    DEFAULT_CONFIG = {
        'core_ring': {
            'radius': 0.3,
            'latency': 0.005,  # 5ms
            'access_level': 'full',
        },
        'outer_ring': {
            'radius': 0.7,
            'latency': 0.2,  # 200ms
            'access_level': 'read_only',
        },
        'wall_threshold': 0.9,
        'max_time_drift': 300,  # 5 minutes
        'allowed_ips': [],
        'allowed_containers': [],
    }

    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None):
        """
        Initialize the gate with configuration.

        Args:
            config: Direct configuration dict
            config_path: Path to YAML config file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
                self.config = file_config.get('aether_auth', self.DEFAULT_CONFIG)
        elif config:
            self.config = {**self.DEFAULT_CONFIG, **config}
        else:
            self.config = self.DEFAULT_CONFIG

        # Initialize Poincare Ball
        self.ball = PoincareBall(dim=6)

        # Baseline (origin for distance calculations)
        self.baseline = create_baseline_vector()
        self.origin = self.ball.embed(self.baseline.dimensions)

    def check_access(self, context: ContextVector) -> AccessResult:
        """
        Validate context vector against trust rings.

        Args:
            context: The captured context vector

        Returns:
            AccessResult with allowed status, ring, and metadata
        """
        # Embed context into Poincare ball
        point = self.ball.embed(context.dimensions)

        # Calculate hyperbolic distance from baseline
        distance = self.ball.hyperbolic_distance(point, self.origin)

        # Also check Euclidean radius (for wall detection)
        radius = self.ball.radius(point)

        # Determine ring based on distance
        core_radius = self.config['core_ring']['radius']
        outer_radius = self.config['outer_ring']['radius']
        wall_threshold = self.config['wall_threshold']

        if distance < core_radius:
            return AccessResult(
                allowed=True,
                ring=TrustRing.CORE,
                distance=distance,
                latency=self.config['core_ring']['latency'],
                access_level=self.config['core_ring']['access_level'],
                reason="Context within Core Ring"
            )

        elif distance < outer_radius:
            return AccessResult(
                allowed=True,
                ring=TrustRing.OUTER,
                distance=distance,
                latency=self.config['outer_ring']['latency'],
                access_level=self.config['outer_ring']['access_level'],
                reason="Context within Outer Ring - degraded access"
            )

        elif distance < wall_threshold or radius < 0.95:
            return AccessResult(
                allowed=False,
                ring=TrustRing.WALL,
                distance=distance,
                reason=f"Behavioral anomaly detected (distance={distance:.3f})"
            )

        else:
            return AccessResult(
                allowed=False,
                ring=TrustRing.EVENT_HORIZON,
                distance=distance,
                reason="Critical security violation - beyond event horizon"
            )

    def enforce_latency(self, access: AccessResult) -> AccessResult:
        """
        Enforce latency penalty based on trust ring.

        Outer ring requests are artificially slowed to
        make brute-force attacks economically infeasible.
        """
        if access.allowed and access.latency > 0:
            time.sleep(access.latency)

        return access

    def validate_with_enforcement(self, context: ContextVector) -> AccessResult:
        """
        Full validation with latency enforcement.

        Combines check_access and enforce_latency.
        """
        access = self.check_access(context)
        return self.enforce_latency(access)

    def get_ring_thresholds(self) -> Dict[str, float]:
        """Get current ring threshold configuration."""
        return {
            'core': self.config['core_ring']['radius'],
            'outer': self.config['outer_ring']['radius'],
            'wall': self.config['wall_threshold'],
        }

    def update_baseline(self, new_baseline: ContextVector):
        """
        Update the baseline (origin) for distance calculations.

        Useful for adapting to legitimate behavioral changes.
        """
        self.baseline = new_baseline
        self.origin = self.ball.embed(new_baseline.dimensions)


def create_default_config() -> str:
    """Generate default configuration YAML."""
    config = {
        'aether_auth': {
            'core_ring': {
                'radius': 0.3,
                'latency': 0.005,
                'access_level': 'full',
            },
            'outer_ring': {
                'radius': 0.7,
                'latency': 0.2,
                'access_level': 'read_only',
            },
            'wall_threshold': 0.9,
            'max_time_drift': 300,
            'allowed_ips': [
                '192.168.1.0/24',
                '10.0.0.0/8',
            ],
            'allowed_containers': [
                'knowledge-bot-prod',
                'knowledge-bot-staging',
            ],
        }
    }
    return yaml.dump(config, default_flow_style=False)


if __name__ == "__main__":
    # Demo
    print("GeoSeal Gate Demo")
    print("=" * 50)

    gate = AetherAuthGate()

    # Test contexts
    test_cases = [
        ("Normal operation", [0.5, 0.5, 0.2, 0.3, 0.5, 0.8]),
        ("High CPU anomaly", [0.5, 0.5, 0.95, 0.8, 0.5, 0.1]),
        ("Unknown location", [0.5, 0.99, 0.3, 0.3, 0.5, 0.5]),
        ("Extreme deviation", [0.99, 0.99, 0.99, 0.99, 0.01, 0.0]),
    ]

    for name, dims in test_cases:
        ctx = ContextVector(
            dimensions=dims,
            timestamp=time.time(),
            hostname="test",
            caller="demo"
        )

        access = gate.check_access(ctx)

        print(f"\n{name}")
        print(f"  Context: {dims}")
        print(f"  Ring: {access.ring.value}")
        print(f"  Distance: {access.distance:.3f}")
        print(f"  Allowed: {access.allowed}")
        if access.reason:
            print(f"  Reason: {access.reason}")
