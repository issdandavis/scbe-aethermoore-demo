"""
SCBE-AETHERMOORE Geographic Vector Module

Transforms geographic location from a static "allow/deny" concept into a
cryptographic coordinate that feeds directly into the 6D security vector.

Key Concepts:
    1. GPS → 6D Intent Vector mapping (Terrestrial)
    2. Super-exponential risk via Harmonic Scaling H(d) = R^(d²)
    3. Context-Bound Key Derivation (GPS + Time + Mission → K_final)
    4. ECI Coordinates for Space-Based Swarm routing

Geography is a cryptographic input - you don't just "check" location;
you use location coordinates to mathematically construct the key and
the validity of the user's existence in 6D space.

Author: SCBE-AETHERMOORE Team
Version: 1.0.0
Date: January 31, 2026
"""

import numpy as np
import hashlib
import hmac
import struct
import time
import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Any
from enum import Enum, auto
from abc import ABC, abstractmethod

# Try to import argon2 for key derivation
try:
    import argon2
    from argon2.low_level import hash_secret_raw, Type
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False


# =============================================================================
# SECTION 1: CONSTANTS
# =============================================================================

# Golden Ratio for Harmonic Wall
PHI: float = (1 + np.sqrt(5)) / 2  # 1.618...

# Default Harmonic Wall base
HARMONIC_R: float = PHI ** 2  # ≈ 2.618

# Earth constants for ECI conversion
EARTH_RADIUS_KM: float = 6371.0
EARTH_ROTATION_RAD_PER_SEC: float = 7.2921159e-5

# Trust realm radius (km) - how far user can travel before risk increases
DEFAULT_TRUST_RADIUS_KM: float = 50.0

# Impossible travel threshold (km/hour) - detects teleportation attacks
MAX_HUMAN_SPEED_KPH: float = 1200.0  # Supersonic flight


# =============================================================================
# SECTION 2: DATA STRUCTURES
# =============================================================================

class CoordinateSystem(Enum):
    """Coordinate system for location input."""
    GPS = auto()        # Lat/Lon (terrestrial)
    ECI = auto()        # Earth-Centered Inertial (space)
    ECEF = auto()       # Earth-Centered Earth-Fixed


@dataclass
class GPSLocation:
    """GPS coordinates with metadata."""
    latitude: float     # -90 to 90
    longitude: float    # -180 to 180
    altitude_m: float = 0.0
    accuracy_m: float = 10.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Validate ranges
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be in [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be in [-180, 180], got {self.longitude}")

    def to_normalized(self) -> Tuple[float, float]:
        """Normalize to [-1, 1] range for 6D vector."""
        x1 = self.latitude / 90.0    # [-90, 90] → [-1, 1]
        x2 = self.longitude / 180.0  # [-180, 180] → [-1, 1]
        return (x1, x2)

    def to_eci(self) -> np.ndarray:
        """Convert GPS to Earth-Centered Inertial coordinates."""
        # Convert to radians
        lat_rad = np.radians(self.latitude)
        lon_rad = np.radians(self.longitude)

        # Earth rotation since J2000 epoch
        # Simplified: use current sidereal time
        sidereal_angle = EARTH_ROTATION_RAD_PER_SEC * self.timestamp

        # Adjusted longitude for Earth rotation
        lon_inertial = lon_rad + sidereal_angle

        # Altitude in km
        r = EARTH_RADIUS_KM + (self.altitude_m / 1000.0)

        # Cartesian ECI
        x = r * np.cos(lat_rad) * np.cos(lon_inertial)
        y = r * np.cos(lat_rad) * np.sin(lon_inertial)
        z = r * np.sin(lat_rad)

        return np.array([x, y, z])


@dataclass
class ECILocation:
    """Earth-Centered Inertial coordinates for space applications."""
    x: float  # km
    y: float  # km
    z: float  # km
    timestamp: float = field(default_factory=time.time)

    def to_normalized(self, max_distance_km: float = 50000.0) -> Tuple[float, float, float]:
        """Normalize ECI to [-1, 1] range for 6D vector."""
        # Normalize by max expected distance (e.g., GEO orbit ~36000 km)
        x1 = np.clip(self.x / max_distance_km, -1, 1)
        x2 = np.clip(self.y / max_distance_km, -1, 1)
        x3 = np.clip(self.z / max_distance_km, -1, 1)
        return (x1, x2, x3)

    def distance_to(self, other: 'ECILocation') -> float:
        """Euclidean distance in km."""
        return np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )


@dataclass
class DeviceContext:
    """Device fingerprint and biometric context."""
    device_id: str
    fingerprint_hash: bytes = b''
    biometric_score: float = 0.5  # 0=no biometric, 1=verified
    threat_level: float = 0.0     # 0=safe, 1=maximum threat

    def to_vector(self) -> Tuple[float, float, float]:
        """Convert to x4, x5, x6 dimensions."""
        # Device ID → hash → normalized float
        device_hash = hashlib.sha256(self.device_id.encode()).digest()
        x4 = (int.from_bytes(device_hash[:4], 'big') / (2**32 - 1)) * 2 - 1

        # Biometric score → [-1, 1] (higher biometric = safer = more negative)
        x5 = 1.0 - 2.0 * self.biometric_score

        # Threat level → [-1, 1] (higher threat = more positive)
        x6 = 2.0 * self.threat_level - 1.0

        return (x4, x5, x6)


@dataclass
class Intent6DVector:
    """
    The 6D Intent Vector for terrestrial applications.

    Dimensions:
        x1: GPS Latitude (normalized to [-1, 1])
        x2: GPS Longitude (normalized to [-1, 1])
        x3: Time of day (normalized to [-1, 1])
        x4: Device fingerprint
        x5: Biometric score
        x6: Threat level
    """
    x1: float  # Latitude
    x2: float  # Longitude
    x3: float  # Time
    x4: float  # Device
    x5: float  # Biometric
    x6: float  # Threat

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for Poincaré ball operations."""
        return np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6])

    def to_poincare(self, scale: float = 0.8) -> np.ndarray:
        """
        Map to interior of Poincaré ball.

        Uses tanh-based projection to ensure ||v|| < 1.
        """
        v = self.to_array()
        # Project each dimension through tanh to bound in (-1, 1)
        v_bounded = np.tanh(v * 0.5)  # Softer bounding
        # Scale to ensure we stay within ball
        norm = np.linalg.norm(v_bounded)
        if norm > 0:
            v_bounded = v_bounded * scale * np.tanh(norm) / norm
        return v_bounded

    @classmethod
    def from_context(
        cls,
        gps: GPSLocation,
        device: DeviceContext,
        time_of_day: Optional[float] = None
    ) -> 'Intent6DVector':
        """
        Construct 6D vector from GPS and device context.

        Args:
            gps: GPS location
            device: Device context
            time_of_day: Hour of day [0, 24] or None for current
        """
        lat_norm, lon_norm = gps.to_normalized()

        # Time of day normalization
        if time_of_day is None:
            # Extract hour from timestamp
            time_of_day = (gps.timestamp % 86400) / 3600  # seconds → hours
        # Map [0, 24] → [-1, 1] with midnight at 0
        x3 = np.cos(2 * np.pi * time_of_day / 24)

        x4, x5, x6 = device.to_vector()

        return cls(
            x1=lat_norm,
            x2=lon_norm,
            x3=x3,
            x4=x4,
            x5=x5,
            x6=x6
        )


# =============================================================================
# SECTION 3: HYPERBOLIC DISTANCE & HARMONIC WALL
# =============================================================================

def hyperbolic_distance(u: np.ndarray, v: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute hyperbolic distance in Poincaré ball.

    Formula:
        d(u,v) = arccosh(1 + 2|u-v|² / ((1-|u|²)(1-|v|²)))

    Properties:
        - d → ∞ as either point approaches boundary
        - d = 0 iff u = v
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    norm_u_sq = np.clip(np.dot(u, u), 0, 1 - eps)
    norm_v_sq = np.clip(np.dot(v, v), 0, 1 - eps)

    diff_sq = np.dot(u - v, u - v)
    denominator = (1 - norm_u_sq) * (1 - norm_v_sq)

    if denominator <= eps:
        return float('inf')

    delta = 2 * diff_sq / denominator
    return float(np.arccosh(1 + delta))


def harmonic_wall(d: float, R: float = HARMONIC_R) -> float:
    """
    Harmonic Scaling Law: H(d, R) = R^(d²)

    Super-exponential cost amplification:
        - Near center (d ≈ 0): H ≈ 1 (negligible)
        - Moderate deviation (d ≈ 0.5): H ≈ 1.5 - 3.5
        - Large deviation (d ≈ 1): H ≈ 2.7 - 54
        - Boundary (d → ∞): H → ∞ (event horizon)

    Args:
        d: Hyperbolic distance from trusted center
        R: Base of exponential (default φ² ≈ 2.618)

    Returns:
        Risk multiplier
    """
    if d == float('inf'):
        return float('inf')
    return R ** (d ** 2)


def compute_location_risk(
    current: Intent6DVector,
    trusted_center: Intent6DVector,
    R: float = HARMONIC_R
) -> Tuple[float, float, str]:
    """
    Compute risk score from location deviation.

    Returns:
        (hyperbolic_distance, harmonic_cost, decision)
    """
    u = current.to_poincare()
    v = trusted_center.to_poincare()

    d = hyperbolic_distance(u, v)
    H = harmonic_wall(d, R)

    # Decision thresholds
    if H < 2.0:
        decision = "ALLOW"
    elif H < 10.0:
        decision = "REVIEW"
    elif H < 100.0:
        decision = "QUARANTINE"
    else:
        decision = "DENY"

    return (d, H, decision)


# =============================================================================
# SECTION 4: CONTEXT-BOUND KEY DERIVATION
# =============================================================================

def derive_context_key_argon2(
    master_key: bytes,
    gps: GPSLocation,
    mission_id: str,
    time_window: int = 300  # 5-minute windows
) -> bytes:
    """
    Derive context-bound key using Argon2id.

    Formula:
        K_classical = Argon2id(master_key, salt=GPS||Time||Mission)
        K_final = K_classical ⊕ K_PQC (if available)

    Properties:
        - Same key only derivable with matching GPS/Time/Mission
        - Wrong context → wrong key → decoy plaintext
        - Time window prevents replay attacks

    Args:
        master_key: Base secret key
        gps: GPS location
        mission_id: Mission/session identifier
        time_window: Time quantization in seconds

    Returns:
        32-byte derived key
    """
    # Quantize time to window
    time_quant = int(gps.timestamp / time_window) * time_window

    # Build context salt
    lat_bytes = struct.pack('>d', gps.latitude)
    lon_bytes = struct.pack('>d', gps.longitude)
    alt_bytes = struct.pack('>d', gps.altitude_m)
    time_bytes = struct.pack('>Q', time_quant)
    mission_bytes = mission_id.encode('utf-8')

    salt = lat_bytes + lon_bytes + alt_bytes + time_bytes + mission_bytes

    # Compute salt hash
    salt_hash = hashlib.sha256(salt).digest()

    if ARGON2_AVAILABLE:
        # Use Argon2id (memory-hard, side-channel resistant)
        derived = hash_secret_raw(
            secret=master_key,
            salt=salt_hash[:16],  # Argon2 needs 8-16 byte salt
            time_cost=3,
            memory_cost=65536,  # 64 MB
            parallelism=4,
            hash_len=32,
            type=Type.ID
        )
    else:
        # Fallback to HKDF-like construction with PBKDF2
        derived = hashlib.pbkdf2_hmac(
            'sha256',
            master_key,
            salt,
            iterations=100000,
            dklen=32
        )

    return derived


def derive_context_key_simple(
    master_key: bytes,
    gps: GPSLocation,
    mission_id: str,
    time_window: int = 300
) -> bytes:
    """
    Simple HMAC-based context key derivation (fallback).

    Less secure than Argon2id but always available.
    """
    time_quant = int(gps.timestamp / time_window) * time_window

    context = f"{gps.latitude:.6f}|{gps.longitude:.6f}|{time_quant}|{mission_id}"

    return hmac.new(
        master_key,
        context.encode('utf-8'),
        hashlib.sha256
    ).digest()


def verify_context_binding(
    ciphertext: bytes,
    expected_mac: bytes,
    key: bytes
) -> bool:
    """
    Verify that ciphertext was encrypted with context-bound key.

    If context is wrong, verification fails silently (no error message).
    """
    computed_mac = hmac.new(key, ciphertext, hashlib.sha256).digest()
    return hmac.compare_digest(computed_mac, expected_mac)


# =============================================================================
# SECTION 5: SPACE-BASED PROXIMITY PROTOCOL
# =============================================================================

@dataclass
class ProximityProtocol:
    """
    Space-based protocol selection based on physical distance.

    Agents that are physically close use lighter protocols to save
    bandwidth (70-80% savings), while distant connections require
    full "6-tongue" verification.
    """
    CLOSE_THRESHOLD_KM: float = 100.0    # Same constellation
    MEDIUM_THRESHOLD_KM: float = 5000.0  # Same orbit region
    FAR_THRESHOLD_KM: float = 40000.0    # Cross-orbit

    def select_protocol(self, distance_km: float) -> Dict[str, Any]:
        """
        Select security protocol based on physical distance.

        Returns:
            Protocol configuration dict
        """
        if distance_km < self.CLOSE_THRESHOLD_KM:
            return {
                "level": "LIGHT",
                "tongues": ["KO", "AV"],  # 2 tongues
                "key_size": 128,
                "auth_rounds": 1,
                "bandwidth_savings": 0.80,
                "description": "Close proximity - minimal verification"
            }
        elif distance_km < self.MEDIUM_THRESHOLD_KM:
            return {
                "level": "STANDARD",
                "tongues": ["KO", "AV", "RU", "CA"],  # 4 tongues
                "key_size": 192,
                "auth_rounds": 2,
                "bandwidth_savings": 0.50,
                "description": "Medium distance - standard verification"
            }
        elif distance_km < self.FAR_THRESHOLD_KM:
            return {
                "level": "FULL",
                "tongues": ["KO", "AV", "RU", "CA", "UM", "DR"],  # 6 tongues
                "key_size": 256,
                "auth_rounds": 3,
                "bandwidth_savings": 0.20,
                "description": "Far distance - full 6-tongue verification"
            }
        else:
            return {
                "level": "MAXIMUM",
                "tongues": ["KO", "AV", "RU", "CA", "UM", "DR"],
                "key_size": 256,
                "auth_rounds": 5,
                "bandwidth_savings": 0.0,
                "pqc_required": True,
                "description": "Extreme distance - PQC + full verification"
            }


# =============================================================================
# SECTION 6: TRAVEL ANOMALY DETECTION
# =============================================================================

def great_circle_distance_km(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Haversine formula for great-circle distance.

    Returns distance in km.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


def detect_impossible_travel(
    previous: GPSLocation,
    current: GPSLocation,
    max_speed_kph: float = MAX_HUMAN_SPEED_KPH
) -> Tuple[bool, float, str]:
    """
    Detect physically impossible travel (teleportation attacks).

    Args:
        previous: Previous GPS location
        current: Current GPS location
        max_speed_kph: Maximum plausible travel speed

    Returns:
        (is_impossible, required_speed_kph, reason)
    """
    distance_km = great_circle_distance_km(
        previous.latitude, previous.longitude,
        current.latitude, current.longitude
    )

    time_delta_hours = (current.timestamp - previous.timestamp) / 3600

    if time_delta_hours <= 0:
        return (True, float('inf'), "Time travel detected (negative delta)")

    required_speed = distance_km / time_delta_hours

    if required_speed > max_speed_kph:
        return (
            True,
            required_speed,
            f"Impossible travel: {distance_km:.0f}km in {time_delta_hours:.2f}h = {required_speed:.0f}km/h"
        )

    return (False, required_speed, "Travel speed within limits")


# =============================================================================
# SECTION 7: UNIFIED GEO-VECTOR PROCESSOR
# =============================================================================

class GeoVectorProcessor:
    """
    Main processor for geographic → 6D vector transformation.

    Integrates:
        - GPS normalization
        - Device context
        - Hyperbolic embedding
        - Risk computation
        - Context-bound key derivation
        - Travel anomaly detection
    """

    def __init__(
        self,
        trusted_location: Optional[GPSLocation] = None,
        trusted_device: Optional[DeviceContext] = None,
        harmonic_base: float = HARMONIC_R,
        trust_radius_km: float = DEFAULT_TRUST_RADIUS_KM
    ):
        self.trusted_location = trusted_location
        self.trusted_device = trusted_device
        self.harmonic_base = harmonic_base
        self.trust_radius_km = trust_radius_km
        self.proximity = ProximityProtocol()

        # History for anomaly detection
        self.location_history: List[GPSLocation] = []

        # Precompute trusted center vector if available
        if trusted_location and trusted_device:
            self.trusted_center = Intent6DVector.from_context(
                trusted_location, trusted_device
            )
        else:
            self.trusted_center = None

    def set_trusted_realm(
        self,
        location: GPSLocation,
        device: DeviceContext
    ) -> None:
        """Set the trusted center (home base) for this user."""
        self.trusted_location = location
        self.trusted_device = device
        self.trusted_center = Intent6DVector.from_context(location, device)

    def process(
        self,
        current_gps: GPSLocation,
        current_device: DeviceContext,
        master_key: Optional[bytes] = None,
        mission_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full processing pipeline for geographic vector.

        Returns:
            Complete result dict with vectors, distances, risk, keys
        """
        result = {
            "timestamp": time.time(),
            "input": {
                "gps": {
                    "latitude": current_gps.latitude,
                    "longitude": current_gps.longitude,
                    "altitude_m": current_gps.altitude_m
                },
                "device_id": current_device.device_id
            }
        }

        # Build 6D vector
        intent_vector = Intent6DVector.from_context(current_gps, current_device)
        poincare_vector = intent_vector.to_poincare()

        result["vector"] = {
            "6d": intent_vector.to_array().tolist(),
            "poincare": poincare_vector.tolist(),
            "poincare_norm": float(np.linalg.norm(poincare_vector))
        }

        # Compute risk if trusted center exists
        if self.trusted_center:
            d, H, decision = compute_location_risk(
                intent_vector,
                self.trusted_center,
                self.harmonic_base
            )

            result["risk"] = {
                "hyperbolic_distance": d,
                "harmonic_cost": H,
                "decision": decision,
                "harmonic_base": self.harmonic_base
            }

            # Geographic distance
            if self.trusted_location:
                geo_distance = great_circle_distance_km(
                    self.trusted_location.latitude,
                    self.trusted_location.longitude,
                    current_gps.latitude,
                    current_gps.longitude
                )
                result["risk"]["geographic_distance_km"] = geo_distance

        # Anomaly detection (impossible travel)
        if self.location_history:
            previous = self.location_history[-1]
            is_impossible, speed, reason = detect_impossible_travel(
                previous, current_gps
            )
            result["anomaly"] = {
                "impossible_travel": is_impossible,
                "required_speed_kph": speed,
                "reason": reason
            }

        # Update history
        self.location_history.append(current_gps)
        if len(self.location_history) > 100:
            self.location_history = self.location_history[-100:]

        # Context-bound key derivation
        if master_key and mission_id:
            derived_key = derive_context_key_argon2(
                master_key, current_gps, mission_id
            )
            result["key_derivation"] = {
                "algorithm": "Argon2id" if ARGON2_AVAILABLE else "PBKDF2-SHA256",
                "key_id": hashlib.sha256(derived_key).hexdigest()[:16],
                "context_bound": True
            }
            # Don't include actual key in result!

        return result


# =============================================================================
# SECTION 8: DEMO & TESTING
# =============================================================================

def demo():
    """Demonstrate geographic vector processing."""
    print("=" * 70)
    print("SCBE-AETHERMOORE GEOGRAPHIC VECTOR DEMO")
    print("Geography as Cryptographic Input")
    print("=" * 70)

    # Create trusted home location (New York City)
    home_gps = GPSLocation(
        latitude=40.7128,
        longitude=-74.0060,
        altitude_m=10.0
    )
    home_device = DeviceContext(
        device_id="trusted-laptop-001",
        biometric_score=0.95,
        threat_level=0.0
    )

    # Initialize processor
    processor = GeoVectorProcessor()
    processor.set_trusted_realm(home_gps, home_device)

    print("\n1. TRUSTED REALM (Home Base)")
    print(f"   Location: NYC ({home_gps.latitude:.4f}, {home_gps.longitude:.4f})")
    print(f"   Device: {home_device.device_id}")
    print(f"   Biometric: {home_device.biometric_score:.0%}")

    # Test scenarios
    test_cases = [
        {
            "name": "Same City (10km away)",
            "gps": GPSLocation(40.7580, -73.9855),  # Times Square
            "device": home_device,
            "expected": "ALLOW"
        },
        {
            "name": "Different State (500km - Boston)",
            "gps": GPSLocation(42.3601, -71.0589),
            "device": home_device,
            "expected": "REVIEW"
        },
        {
            "name": "Different Continent (London)",
            "gps": GPSLocation(51.5074, -0.1278),
            "device": home_device,
            "expected": "QUARANTINE/DENY"
        },
        {
            "name": "Opposite Side of Earth (Sydney)",
            "gps": GPSLocation(-33.8688, 151.2093),
            "device": home_device,
            "expected": "DENY"
        },
        {
            "name": "Same Location, Different Device",
            "gps": home_gps,
            "device": DeviceContext(
                device_id="unknown-device-999",
                biometric_score=0.0,
                threat_level=0.5
            ),
            "expected": "REVIEW/QUARANTINE"
        },
        {
            "name": "Same Location, High Threat",
            "gps": home_gps,
            "device": DeviceContext(
                device_id="trusted-laptop-001",
                biometric_score=0.95,
                threat_level=0.9
            ),
            "expected": "QUARANTINE"
        }
    ]

    print("\n2. LOCATION DEVIATION TESTS")
    print("-" * 70)

    for tc in test_cases:
        result = processor.process(tc["gps"], tc["device"])

        risk = result.get("risk", {})
        d_H = risk.get("hyperbolic_distance", "N/A")
        H = risk.get("harmonic_cost", "N/A")
        decision = risk.get("decision", "N/A")
        geo_km = risk.get("geographic_distance_km", 0)

        print(f"\n   {tc['name']}")
        print(f"   Geographic Distance: {geo_km:,.0f} km")
        if isinstance(d_H, float):
            print(f"   Hyperbolic Distance: {d_H:.4f}")
        if isinstance(H, float):
            print(f"   Harmonic Cost (H):   {H:.2f}x")
        print(f"   Decision:            {decision}")
        print(f"   Expected:            {tc['expected']}")

    print("\n3. CONTEXT-BOUND KEY DERIVATION")
    print("-" * 70)

    master_key = b"super_secret_master_key_32bytes!"
    mission_id = "mission-alpha-001"

    # Same location, same time → same key
    key1 = derive_context_key_argon2(master_key, home_gps, mission_id)
    key2 = derive_context_key_argon2(master_key, home_gps, mission_id)

    print(f"   Master Key: {master_key[:20]}...")
    print(f"   Mission ID: {mission_id}")
    print(f"   Same context => Same key: {key1 == key2}")

    # Different location => different key
    london_gps = GPSLocation(51.5074, -0.1278)
    key3 = derive_context_key_argon2(master_key, london_gps, mission_id)

    print(f"   Different GPS => Different key: {key1 != key3}")
    print(f"   Key ID (home):   {hashlib.sha256(key1).hexdigest()[:16]}")
    print(f"   Key ID (london): {hashlib.sha256(key3).hexdigest()[:16]}")

    print("\n4. SPACE-BASED PROXIMITY PROTOCOL")
    print("-" * 70)

    prox = ProximityProtocol()
    distances = [50, 500, 10000, 50000]

    for dist in distances:
        proto = prox.select_protocol(dist)
        print(f"\n   Distance: {dist:,} km")
        print(f"   Protocol: {proto['level']}")
        print(f"   Tongues:  {proto['tongues']}")
        print(f"   Savings:  {proto['bandwidth_savings']:.0%}")

    print("\n5. IMPOSSIBLE TRAVEL DETECTION")
    print("-" * 70)

    # Normal travel
    t0 = time.time()
    loc1 = GPSLocation(40.7128, -74.0060, timestamp=t0)
    loc2 = GPSLocation(40.7580, -73.9855, timestamp=t0 + 3600)  # 1 hour later

    is_imp, speed, reason = detect_impossible_travel(loc1, loc2)
    print(f"\n   NYC -> Times Square (1 hour)")
    print(f"   Impossible: {is_imp}")
    print(f"   Speed: {speed:.1f} km/h")

    # Impossible travel (teleportation)
    loc3 = GPSLocation(51.5074, -0.1278, timestamp=t0 + 3600)  # London, 1 hour later
    is_imp, speed, reason = detect_impossible_travel(loc1, loc3)
    print(f"\n   NYC -> London (1 hour)")
    print(f"   Impossible: {is_imp}")
    print(f"   Speed: {speed:.1f} km/h")
    print(f"   Reason: {reason}")

    print("\n" + "=" * 70)
    print("SUMMARY: Geography is not a policy rule - it's a cryptographic coordinate")
    print("=" * 70)


if __name__ == "__main__":
    demo()
