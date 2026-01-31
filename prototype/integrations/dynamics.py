"""
Neural ODE Integration for Swarm Flux Dynamics.

Uses torchdiffeq to model continuous flux evolution as a
dynamical system. This replaces discrete flux stepping with
smooth, learnable dynamics.

Reference: https://github.com/rtqichen/torchdiffeq
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not installed. Using Euler fallback.")


# Pythagorean comma - the "decimal drift" constant
PYTHAGOREAN_COMMA = 531441 / 524288  # 1.0136432648...


class FluxODE(nn.Module):
    """
    Neural ODE for swarm flux dynamics.

    Models the continuous evolution of flux states (ν) for each tongue.
    The dynamics drive agents toward coherence while respecting
    the Pythagorean comma drift constant.

    Differential equation:
        dν/dt = -α(ν - ν_target) + τ * mean_field + σ * noise

    Where:
        α: Decay rate toward target state
        τ: Mean-field coupling strength
        σ: Stochastic drift (Pythagorean comma influence)
    """

    def __init__(
        self,
        n_tongues: int = 6,
        alpha: float = 0.1,
        tau: float = 0.05,
        sigma: float = 0.01,
        target_state: str = 'POLLY'
    ):
        super().__init__()
        self.n_tongues = n_tongues
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.sigma = sigma

        # Target flux values for each state
        self.target_values = {
            'POLLY': 1.0,
            'QUASI': 0.5,
            'DEMI': 0.1,
            'COLLAPSED': 0.0,
        }
        self.target = self.target_values.get(target_state, 1.0)

        # Learnable tongue-specific coupling weights
        self.coupling = nn.Parameter(torch.ones(n_tongues) / n_tongues)

    def forward(self, t, nu):
        """
        Compute dν/dt at time t given flux state ν.

        Args:
            t: Current time (unused in autonomous systems)
            nu: Tensor of shape (n_tongues,) with current flux values
        """
        # Target attraction
        target_attraction = -self.alpha * (nu - self.target)

        # Mean-field coupling (weighted average)
        weights = torch.softmax(self.coupling, dim=0)
        mean_nu = torch.sum(weights * nu)
        mean_field = self.tau * (mean_nu - nu)

        # Pythagorean comma drift (deterministic noise-like term)
        drift = self.sigma * PYTHAGOREAN_COMMA * torch.sin(2 * np.pi * nu)

        return target_attraction + mean_field + drift

    def get_state(self, nu: float) -> str:
        """Determine dimensional state from flux value."""
        if nu >= 0.9:
            return 'POLLY'
        elif nu >= 0.4:
            return 'QUASI'
        elif nu >= 0.1:
            return 'DEMI'
        else:
            return 'COLLAPSED'


class HarmonicWallODE(nn.Module):
    """
    Neural ODE for agent position dynamics under Harmonic Wall potential.

    Agents are pushed away from the boundary by the exponential potential:
        V(r) = exp(r²) where r = ||position||

    The force is the negative gradient of this potential.
    """

    def __init__(self, dim: int = 2, wall_strength: float = 1.0):
        super().__init__()
        self.dim = dim
        self.wall_strength = wall_strength

    def forward(self, t, x):
        """
        Compute dx/dt under Harmonic Wall potential.

        Args:
            t: Current time
            x: Position tensor of shape (..., dim)
        """
        r_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        r = torch.sqrt(r_sq + 1e-8)

        # Force = -∇V = -2r * exp(r²) * (x/r)
        force = -2 * self.wall_strength * r * torch.exp(r_sq) * (x / (r + 1e-8))

        # Clamp force near boundary
        force = torch.clamp(force, -10.0, 10.0)

        return force


def evolve_swarm_flux(
    initial_flux: np.ndarray,
    t_span: tuple = (0, 10),
    n_steps: int = 100,
    target_state: str = 'POLLY',
    use_adjoint: bool = False
) -> tuple:
    """
    Evolve swarm flux states using Neural ODE.

    Args:
        initial_flux: Initial flux values for each tongue
        t_span: (t_start, t_end) time interval
        n_steps: Number of time steps to output
        target_state: Target dimensional state ('POLLY', 'QUASI', 'DEMI')
        use_adjoint: Use adjoint method for O(1) memory backprop

    Returns:
        (times, trajectories): Time points and flux trajectories
    """
    if not TORCHDIFFEQ_AVAILABLE:
        return _fallback_evolve(initial_flux, t_span, n_steps)

    n_tongues = len(initial_flux)
    dynamics = FluxODE(n_tongues=n_tongues, target_state=target_state)

    nu_0 = torch.tensor(initial_flux, dtype=torch.float32)
    t = torch.linspace(t_span[0], t_span[1], n_steps)

    solver = odeint_adjoint if use_adjoint else odeint
    trajectory = solver(dynamics, nu_0, t, method='dopri5')

    return t.numpy(), trajectory.detach().numpy()


def _fallback_evolve(
    initial_flux: np.ndarray,
    t_span: tuple,
    n_steps: int
) -> tuple:
    """Simple Euler integration fallback."""
    dt = (t_span[1] - t_span[0]) / n_steps
    times = np.linspace(t_span[0], t_span[1], n_steps)
    trajectory = np.zeros((n_steps, len(initial_flux)))
    trajectory[0] = initial_flux

    alpha, tau = 0.1, 0.05
    target = 1.0

    for i in range(1, n_steps):
        nu = trajectory[i - 1]
        # Simple dynamics
        dnu = -alpha * (nu - target) + tau * (np.mean(nu) - nu)
        trajectory[i] = np.clip(nu + dt * dnu, 0, 1)

    return times, trajectory


def demo():
    """Demonstrate Neural ODE flux evolution."""
    print("=" * 60)
    print("Neural ODE Flux Dynamics Demo")
    print("=" * 60)

    print(f"\nTorchdiffeq available: {TORCHDIFFEQ_AVAILABLE}")

    # Initial flux: mixed states
    initial = np.array([1.0, 0.8, 0.6, 0.5, 0.3, 0.2])
    tongue_names = ['KO', 'AV', 'RU', 'CA', 'UM', 'DR']

    print("\nInitial flux states:")
    for name, nu in zip(tongue_names, initial):
        state = 'POLLY' if nu >= 0.9 else 'QUASI' if nu >= 0.4 else 'DEMI' if nu >= 0.1 else 'COLLAPSED'
        print(f"  {name}: ν={nu:.2f} ({state})")

    # Evolve toward POLLY
    times, trajectory = evolve_swarm_flux(initial, t_span=(0, 20), n_steps=50)

    print("\nFinal flux states (after evolution):")
    final = trajectory[-1]
    for name, nu in zip(tongue_names, final):
        state = 'POLLY' if nu >= 0.9 else 'QUASI' if nu >= 0.4 else 'DEMI' if nu >= 0.1 else 'COLLAPSED'
        print(f"  {name}: ν={nu:.2f} ({state})")

    # Coherence metric
    coherence = 1 - np.std(final) / np.mean(final)
    print(f"\nSwarm coherence: {coherence:.3f}")


if __name__ == "__main__":
    demo()
