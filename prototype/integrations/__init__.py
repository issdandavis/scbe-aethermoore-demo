"""
SCBE-AETHERMOORE Integration Modules

Drop-in replacements using production-grade open source libraries.
"""

from .hyperbolic import GeooptPoincareBall, create_tongue_manifold
from .dynamics import FluxODE, evolve_swarm_flux

__all__ = [
    'GeooptPoincareBall',
    'create_tongue_manifold',
    'FluxODE',
    'evolve_swarm_flux',
]
