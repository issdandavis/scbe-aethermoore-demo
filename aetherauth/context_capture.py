"""
Context Vector Capture for AetherAuth

Generates a 6D vector representing the bot's current state:
    - Dimension 1: Temporal (time of day)
    - Dimension 2: Spatial (IP/hostname hash)
    - Dimension 3: Computational load (CPU %)
    - Dimension 4: Memory state (RAM %)
    - Dimension 5: Intent code (caller function hash)
    - Dimension 6: Historical behavior (recent success count)

This vector is used to validate that the requesting system
is operating within expected parameters.
"""

import time
import socket
import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional, Callable
import inspect

# Try to import psutil, fall back to mock values
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class ContextVector:
    """A 6D context vector with metadata."""
    dimensions: List[float]
    timestamp: float
    hostname: str
    caller: str

    def __post_init__(self):
        if len(self.dimensions) != 6:
            raise ValueError("Context vector must have exactly 6 dimensions")
        # Normalize to [0, 1]
        self.dimensions = [max(0.0, min(1.0, d)) for d in self.dimensions]

    def to_list(self) -> List[float]:
        return self.dimensions

    def __repr__(self):
        dims = ', '.join(f'{d:.3f}' for d in self.dimensions)
        return f"ContextVector([{dims}])"


# In-memory cache for success tracking
_success_cache: List[float] = []
_CACHE_WINDOW = 3600  # 1 hour


def record_success():
    """Record a successful API call for historical tracking."""
    global _success_cache
    now = time.time()
    _success_cache.append(now)
    # Prune old entries
    _success_cache = [t for t in _success_cache if now - t < _CACHE_WINDOW]


def get_recent_success_count() -> int:
    """Get count of successful calls in the last hour."""
    global _success_cache
    now = time.time()
    _success_cache = [t for t in _success_cache if now - t < _CACHE_WINDOW]
    return len(_success_cache)


def _get_cpu_percent() -> float:
    """Get CPU usage percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.cpu_percent(interval=0.1)
    else:
        # Fallback: use load average on Unix, or return 50% as default
        try:
            load = os.getloadavg()[0]
            return min(load * 10, 100.0)  # Rough approximation
        except (OSError, AttributeError):
            return 50.0


def _get_memory_percent() -> float:
    """Get memory usage percentage."""
    if PSUTIL_AVAILABLE:
        return psutil.virtual_memory().percent
    else:
        return 50.0  # Default fallback


def _get_ip_hash() -> int:
    """Get hash of current IP address."""
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return int(hashlib.sha256(ip_address.encode()).hexdigest()[:8], 16)
    except Exception:
        # Fallback to hostname hash
        return int(hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8], 16)


def _get_intent_hash(caller_name: Optional[str] = None) -> int:
    """Get hash of the calling function name."""
    if caller_name is None:
        # Walk up the stack to find the real caller
        for frame_info in inspect.stack()[2:]:
            if not frame_info.function.startswith('_'):
                caller_name = frame_info.function
                break
        else:
            caller_name = "unknown"

    return int(hashlib.sha256(caller_name.encode()).hexdigest()[:8], 16)


def capture_context_vector(
    caller_name: Optional[str] = None,
    custom_dimensions: Optional[dict] = None
) -> ContextVector:
    """
    Generate 6D context vector representing the bot's current state.

    Args:
        caller_name: Override for the calling function name
        custom_dimensions: Dict of dimension overrides (keys: 0-5 or names)

    Returns:
        ContextVector with normalized [0, 1] dimensions

    Dimension mapping:
        0: temporal - Time of day (0 = midnight, 0.5 = noon, 1 = midnight)
        1: spatial - IP/hostname hash normalized
        2: cpu - CPU usage percentage
        3: memory - Memory usage percentage
        4: intent - Calling function hash
        5: history - Recent success count (capped at 100)
    """
    timestamp = time.time()

    # Dimension 1: Temporal (time of day)
    time_of_day = (timestamp % 86400) / 86400

    # Dimension 2: Spatial (IP hash)
    ip_hash = _get_ip_hash()
    spatial = ip_hash / (2 ** 32)

    # Dimension 3: Computational Load
    cpu = _get_cpu_percent() / 100.0

    # Dimension 4: Memory State
    memory = _get_memory_percent() / 100.0

    # Dimension 5: Intent Code
    intent_hash = _get_intent_hash(caller_name)
    intent = intent_hash / (2 ** 32)

    # Dimension 6: Historical Behavior
    recent_count = get_recent_success_count()
    history = min(recent_count / 100.0, 1.0)

    dimensions = [time_of_day, spatial, cpu, memory, intent, history]

    # Apply custom overrides
    if custom_dimensions:
        dim_names = ['temporal', 'spatial', 'cpu', 'memory', 'intent', 'history']
        for key, value in custom_dimensions.items():
            if isinstance(key, int) and 0 <= key < 6:
                dimensions[key] = value
            elif isinstance(key, str) and key in dim_names:
                dimensions[dim_names.index(key)] = value

    # Get hostname and caller for metadata
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    if caller_name is None:
        for frame_info in inspect.stack()[1:]:
            if not frame_info.function.startswith('_'):
                caller_name = frame_info.function
                break
        else:
            caller_name = "unknown"

    return ContextVector(
        dimensions=dimensions,
        timestamp=timestamp,
        hostname=hostname,
        caller=caller_name
    )


def create_baseline_vector() -> ContextVector:
    """
    Create a baseline context vector representing "normal" operation.

    This is used as the origin point for trust ring calculations.
    """
    return ContextVector(
        dimensions=[0.5, 0.5, 0.3, 0.4, 0.5, 0.8],
        timestamp=time.time(),
        hostname="baseline",
        caller="baseline"
    )


if __name__ == "__main__":
    # Demo
    print("Context Vector Capture Demo")
    print("=" * 40)

    ctx = capture_context_vector()
    print(f"Captured: {ctx}")
    print(f"Timestamp: {ctx.timestamp}")
    print(f"Hostname: {ctx.hostname}")
    print(f"Caller: {ctx.caller}")

    print("\nDimension breakdown:")
    names = ['Temporal', 'Spatial', 'CPU', 'Memory', 'Intent', 'History']
    for name, val in zip(names, ctx.dimensions):
        print(f"  {name}: {val:.4f}")
