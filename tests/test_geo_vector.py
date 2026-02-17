"""
SCBE-AETHERMOORE Geographic Vector Tests
=========================================

Tests for the geographic location integration into the 6D security vector.

Key concepts tested:
    1. GPS -> 6D Intent Vector mapping
    2. Super-exponential risk via Harmonic Wall
    3. Context-bound key derivation
    4. ECI coordinates for space-based applications
    5. Impossible travel detection

Author: SCBE-AETHERMOORE Team
Version: 1.0.0
Date: January 31, 2026
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prototype.geo_vector import (
    GPSLocation,
    ECILocation,
    DeviceContext,
    Intent6DVector,
    GeoVectorProcessor,
    ProximityProtocol,
    hyperbolic_distance,
    harmonic_wall,
    compute_location_risk,
    derive_context_key_argon2,
    derive_context_key_simple,
    great_circle_distance_km,
    detect_impossible_travel,
    PHI,
    HARMONIC_R,
)


# =============================================================================
# SECTION 1: GPS LOCATION TESTS
# =============================================================================

class TestGPSLocation:
    """Tests for GPS coordinate handling."""

    def test_valid_gps_creation(self):
        """GPS location with valid coordinates should be created."""
        gps = GPSLocation(latitude=40.7128, longitude=-74.0060)
        assert gps.latitude == 40.7128
        assert gps.longitude == -74.0060

    def test_invalid_latitude_rejected(self):
        """Latitude outside [-90, 90] should raise ValueError."""
        with pytest.raises(ValueError):
            GPSLocation(latitude=91.0, longitude=0.0)

        with pytest.raises(ValueError):
            GPSLocation(latitude=-91.0, longitude=0.0)

    def test_invalid_longitude_rejected(self):
        """Longitude outside [-180, 180] should raise ValueError."""
        with pytest.raises(ValueError):
            GPSLocation(latitude=0.0, longitude=181.0)

        with pytest.raises(ValueError):
            GPSLocation(latitude=0.0, longitude=-181.0)

    def test_normalization(self):
        """GPS coordinates should normalize to [-1, 1]."""
        # Equator, prime meridian
        gps1 = GPSLocation(latitude=0.0, longitude=0.0)
        x1, x2 = gps1.to_normalized()
        assert x1 == 0.0
        assert x2 == 0.0

        # North pole
        gps2 = GPSLocation(latitude=90.0, longitude=0.0)
        x1, x2 = gps2.to_normalized()
        assert x1 == 1.0

        # South pole
        gps3 = GPSLocation(latitude=-90.0, longitude=0.0)
        x1, x2 = gps3.to_normalized()
        assert x1 == -1.0

    def test_eci_conversion(self):
        """GPS should convert to ECI coordinates."""
        gps = GPSLocation(latitude=0.0, longitude=0.0, altitude_m=0.0)
        eci = gps.to_eci()

        assert len(eci) == 3
        # At equator/prime meridian, should be roughly (R_earth, 0, 0) + rotation
        assert np.linalg.norm(eci) > 6000  # > Earth radius in km


# =============================================================================
# SECTION 2: 6D INTENT VECTOR TESTS
# =============================================================================

class TestIntent6DVector:
    """Tests for 6D intent vector construction."""

    def test_from_context(self):
        """6D vector should be constructed from GPS and device context."""
        gps = GPSLocation(latitude=40.7128, longitude=-74.0060)
        device = DeviceContext(
            device_id="test-device",
            biometric_score=0.9,
            threat_level=0.1
        )

        vector = Intent6DVector.from_context(gps, device)

        assert vector.x1 == pytest.approx(40.7128 / 90.0, abs=0.001)
        assert vector.x2 == pytest.approx(-74.0060 / 180.0, abs=0.001)

    def test_poincare_embedding_within_ball(self):
        """Poincare embedding should have norm < 1."""
        gps = GPSLocation(latitude=45.0, longitude=90.0)
        device = DeviceContext(device_id="test", biometric_score=0.5, threat_level=0.5)

        vector = Intent6DVector.from_context(gps, device)
        poincare = vector.to_poincare()

        norm = np.linalg.norm(poincare)
        assert norm < 1.0, f"Poincare embedding must be within ball, got norm={norm}"

    def test_extreme_coordinates_within_ball(self):
        """Even extreme coordinates should map within Poincare ball."""
        # Max latitude/longitude
        gps = GPSLocation(latitude=90.0, longitude=180.0)
        device = DeviceContext(device_id="x", biometric_score=0.0, threat_level=1.0)

        vector = Intent6DVector.from_context(gps, device)
        poincare = vector.to_poincare()

        norm = np.linalg.norm(poincare)
        assert norm < 1.0


# =============================================================================
# SECTION 3: HYPERBOLIC DISTANCE TESTS
# =============================================================================

class TestHyperbolicDistance:
    """Tests for hyperbolic distance computation."""

    def test_distance_to_self_is_zero(self):
        """Distance from point to itself should be zero."""
        u = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1])
        d = hyperbolic_distance(u, u)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_distance_symmetric(self):
        """d(u, v) should equal d(v, u)."""
        u = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        v = np.array([0.3, 0.1, 0.0, 0.0, 0.0, 0.0])

        assert hyperbolic_distance(u, v) == pytest.approx(hyperbolic_distance(v, u), abs=1e-10)

    def test_distance_to_origin(self):
        """Distance from origin should be finite for points inside ball."""
        origin = np.zeros(6)
        point = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        d = hyperbolic_distance(origin, point)
        assert 0 < d < float('inf')

    def test_distance_increases_near_boundary(self):
        """Distance should increase as points approach boundary."""
        origin = np.zeros(6)

        d1 = hyperbolic_distance(origin, np.array([0.5, 0, 0, 0, 0, 0]))
        d2 = hyperbolic_distance(origin, np.array([0.9, 0, 0, 0, 0, 0]))
        d3 = hyperbolic_distance(origin, np.array([0.99, 0, 0, 0, 0, 0]))

        assert d1 < d2 < d3


# =============================================================================
# SECTION 4: HARMONIC WALL TESTS
# =============================================================================

class TestHarmonicWall:
    """Tests for Harmonic Scaling Law H(d) = R^(d^2)."""

    def test_zero_distance_unit_cost(self):
        """H(0) should be 1 (no amplification)."""
        H = harmonic_wall(0.0)
        assert H == pytest.approx(1.0, abs=1e-10)

    def test_super_exponential_growth(self):
        """Cost should grow super-exponentially with distance."""
        H1 = harmonic_wall(0.5)
        H2 = harmonic_wall(1.0)
        H3 = harmonic_wall(2.0)

        # H(2) should be MUCH larger than H(1) which should be larger than H(0.5)
        assert H1 < H2 < H3
        # Super-exponential: H(2)/H(1) > H(1)/H(0.5)
        ratio_2_1 = H3 / H2
        ratio_1_05 = H2 / H1
        assert ratio_2_1 > ratio_1_05

    def test_phi_squared_base(self):
        """Default base should be phi^2."""
        expected_R = PHI ** 2
        assert HARMONIC_R == pytest.approx(expected_R, abs=1e-10)

    def test_infinity_returns_infinity(self):
        """H(inf) should be inf."""
        H = harmonic_wall(float('inf'))
        assert H == float('inf')


# =============================================================================
# SECTION 5: LOCATION RISK COMPUTATION TESTS
# =============================================================================

class TestLocationRisk:
    """Tests for location-based risk computation."""

    def test_same_location_low_risk(self):
        """Same location should have low risk."""
        gps = GPSLocation(latitude=40.7128, longitude=-74.0060)
        device = DeviceContext(device_id="home", biometric_score=0.9, threat_level=0.0)

        current = Intent6DVector.from_context(gps, device)
        trusted = Intent6DVector.from_context(gps, device)

        d, H, decision = compute_location_risk(current, trusted)

        assert d == pytest.approx(0.0, abs=0.01)
        assert H == pytest.approx(1.0, abs=0.1)
        assert decision == "ALLOW"

    def test_different_device_increases_risk(self):
        """Different device should increase risk even at same location."""
        gps = GPSLocation(latitude=40.7128, longitude=-74.0060)

        trusted_device = DeviceContext(device_id="home-laptop", biometric_score=0.95, threat_level=0.0)
        unknown_device = DeviceContext(device_id="unknown-phone", biometric_score=0.0, threat_level=0.5)

        trusted = Intent6DVector.from_context(gps, trusted_device)
        current = Intent6DVector.from_context(gps, unknown_device)

        d, H, decision = compute_location_risk(current, trusted)

        assert d > 0.5  # Significant deviation
        assert H > 1.5  # Amplified cost


# =============================================================================
# SECTION 6: CONTEXT-BOUND KEY DERIVATION TESTS
# =============================================================================

class TestContextBoundKeyDerivation:
    """Tests for context-bound cryptographic key derivation."""

    def test_same_context_same_key(self):
        """Same GPS/time/mission should derive the same key."""
        master_key = b"test_master_key_32_bytes_long!!"
        gps = GPSLocation(latitude=40.7128, longitude=-74.0060)
        mission_id = "mission-001"

        key1 = derive_context_key_argon2(master_key, gps, mission_id)
        key2 = derive_context_key_argon2(master_key, gps, mission_id)

        assert key1 == key2

    def test_different_location_different_key(self):
        """Different GPS should derive different key."""
        master_key = b"test_master_key_32_bytes_long!!"
        mission_id = "mission-001"

        gps_nyc = GPSLocation(latitude=40.7128, longitude=-74.0060)
        gps_london = GPSLocation(latitude=51.5074, longitude=-0.1278)

        key_nyc = derive_context_key_argon2(master_key, gps_nyc, mission_id)
        key_london = derive_context_key_argon2(master_key, gps_london, mission_id)

        assert key_nyc != key_london

    def test_different_mission_different_key(self):
        """Different mission ID should derive different key."""
        master_key = b"test_master_key_32_bytes_long!!"
        gps = GPSLocation(latitude=40.7128, longitude=-74.0060)

        key1 = derive_context_key_argon2(master_key, gps, "mission-001")
        key2 = derive_context_key_argon2(master_key, gps, "mission-002")

        assert key1 != key2

    def test_key_length_32_bytes(self):
        """Derived key should be exactly 32 bytes."""
        master_key = b"test_master_key_32_bytes_long!!"
        gps = GPSLocation(latitude=0.0, longitude=0.0)

        key = derive_context_key_argon2(master_key, gps, "test")

        assert len(key) == 32

    def test_simple_derivation_fallback(self):
        """Simple HMAC-based derivation should work as fallback."""
        master_key = b"test_master_key"
        gps = GPSLocation(latitude=40.7128, longitude=-74.0060)

        key = derive_context_key_simple(master_key, gps, "mission-001")

        assert len(key) == 32
        assert isinstance(key, bytes)


# =============================================================================
# SECTION 7: IMPOSSIBLE TRAVEL DETECTION TESTS
# =============================================================================

class TestImpossibleTravel:
    """Tests for teleportation attack detection."""

    def test_normal_travel_allowed(self):
        """Reasonable travel speed should not trigger detection."""
        t0 = time.time()

        # NYC to Times Square in 1 hour (~5km, ~5 km/h walking)
        loc1 = GPSLocation(40.7128, -74.0060, timestamp=t0)
        loc2 = GPSLocation(40.7580, -73.9855, timestamp=t0 + 3600)

        is_impossible, speed, _ = detect_impossible_travel(loc1, loc2)

        assert is_impossible == False
        assert speed < 100  # Walking/driving speed

    def test_impossible_travel_detected(self):
        """NYC to London in 1 hour should be detected as impossible."""
        t0 = time.time()

        loc_nyc = GPSLocation(40.7128, -74.0060, timestamp=t0)
        loc_london = GPSLocation(51.5074, -0.1278, timestamp=t0 + 3600)

        is_impossible, speed, reason = detect_impossible_travel(loc_nyc, loc_london)

        assert is_impossible == True
        assert speed > 5000  # > 5000 km/h required
        assert "Impossible" in reason

    def test_supersonic_flight_allowed(self):
        """Travel under 1200 km/h should be allowed (Concorde speed)."""
        t0 = time.time()

        # ~1000 km in 1 hour = 1000 km/h (fast but possible)
        loc1 = GPSLocation(40.7128, -74.0060, timestamp=t0)
        loc2 = GPSLocation(48.8566, -74.0060, timestamp=t0 + 3600)  # ~900km north

        is_impossible, speed, _ = detect_impossible_travel(loc1, loc2)

        assert is_impossible == False

    def test_negative_time_detected(self):
        """Time travel (negative delta) should be detected."""
        t0 = time.time()

        loc1 = GPSLocation(40.7128, -74.0060, timestamp=t0)
        loc2 = GPSLocation(40.7580, -73.9855, timestamp=t0 - 3600)  # 1 hour BEFORE

        is_impossible, _, reason = detect_impossible_travel(loc1, loc2)

        assert is_impossible == True
        assert "Time travel" in reason


# =============================================================================
# SECTION 8: GREAT CIRCLE DISTANCE TESTS
# =============================================================================

class TestGreatCircleDistance:
    """Tests for Haversine great-circle distance."""

    def test_same_point_zero_distance(self):
        """Distance from point to itself should be zero."""
        d = great_circle_distance_km(40.7128, -74.0060, 40.7128, -74.0060)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_nyc_to_london(self):
        """NYC to London should be approximately 5570 km."""
        d = great_circle_distance_km(40.7128, -74.0060, 51.5074, -0.1278)
        assert 5500 < d < 5650  # Known distance ~5570 km

    def test_antipodal_points(self):
        """Opposite sides of Earth should be ~20000 km apart."""
        d = great_circle_distance_km(0.0, 0.0, 0.0, 180.0)
        assert 20000 < d < 20100  # Half circumference


# =============================================================================
# SECTION 9: PROXIMITY PROTOCOL TESTS
# =============================================================================

class TestProximityProtocol:
    """Tests for space-based proximity protocol selection."""

    def test_close_proximity_light_protocol(self):
        """Close agents (<100km) should use LIGHT protocol."""
        prox = ProximityProtocol()
        proto = prox.select_protocol(50)

        assert proto["level"] == "LIGHT"
        assert len(proto["tongues"]) == 2  # KO, AV only
        assert proto["bandwidth_savings"] == 0.80

    def test_medium_distance_standard_protocol(self):
        """Medium distance should use STANDARD protocol."""
        prox = ProximityProtocol()
        proto = prox.select_protocol(500)

        assert proto["level"] == "STANDARD"
        assert len(proto["tongues"]) == 4

    def test_far_distance_full_protocol(self):
        """Far distance should use FULL 6-tongue protocol."""
        prox = ProximityProtocol()
        proto = prox.select_protocol(10000)

        assert proto["level"] == "FULL"
        assert len(proto["tongues"]) == 6

    def test_extreme_distance_pqc_required(self):
        """Extreme distance should require PQC."""
        prox = ProximityProtocol()
        proto = prox.select_protocol(50000)

        assert proto["level"] == "MAXIMUM"
        assert proto.get("pqc_required") == True


# =============================================================================
# SECTION 10: GEO VECTOR PROCESSOR TESTS
# =============================================================================

class TestGeoVectorProcessor:
    """Tests for the unified geo-vector processor."""

    def test_processor_initialization(self):
        """Processor should initialize with trusted realm."""
        home_gps = GPSLocation(40.7128, -74.0060)
        home_device = DeviceContext(device_id="home", biometric_score=0.9, threat_level=0.0)

        processor = GeoVectorProcessor()
        processor.set_trusted_realm(home_gps, home_device)

        assert processor.trusted_center is not None

    def test_process_returns_complete_result(self):
        """Process should return complete result dict."""
        home_gps = GPSLocation(40.7128, -74.0060)
        home_device = DeviceContext(device_id="home", biometric_score=0.9, threat_level=0.0)

        processor = GeoVectorProcessor()
        processor.set_trusted_realm(home_gps, home_device)

        result = processor.process(home_gps, home_device)

        assert "vector" in result
        assert "risk" in result
        assert "6d" in result["vector"]
        assert "poincare" in result["vector"]

    def test_process_with_key_derivation(self):
        """Process should include key derivation when master key provided."""
        home_gps = GPSLocation(40.7128, -74.0060)
        home_device = DeviceContext(device_id="home", biometric_score=0.9, threat_level=0.0)

        processor = GeoVectorProcessor()
        processor.set_trusted_realm(home_gps, home_device)

        master_key = b"test_master_key_32_bytes_long!!"
        result = processor.process(home_gps, home_device, master_key, "mission-001")

        assert "key_derivation" in result
        assert result["key_derivation"]["context_bound"] == True

    def test_anomaly_detection_after_history(self):
        """Anomaly detection should work after location history is built."""
        home_gps = GPSLocation(40.7128, -74.0060)
        home_device = DeviceContext(device_id="home", biometric_score=0.9, threat_level=0.0)

        processor = GeoVectorProcessor()
        processor.set_trusted_realm(home_gps, home_device)

        # First process (no history)
        result1 = processor.process(home_gps, home_device)
        assert "anomaly" not in result1  # No history yet

        # Second process (should have anomaly check)
        t = time.time()
        gps2 = GPSLocation(40.7580, -73.9855, timestamp=t + 3600)
        result2 = processor.process(gps2, home_device)

        assert "anomaly" in result2


# =============================================================================
# SECTION 11: ECI LOCATION TESTS
# =============================================================================

class TestECILocation:
    """Tests for Earth-Centered Inertial coordinates."""

    def test_eci_distance(self):
        """Distance between ECI points should be computed correctly."""
        loc1 = ECILocation(x=6371.0, y=0.0, z=0.0)  # On equator
        loc2 = ECILocation(x=6371.0 + 100.0, y=0.0, z=0.0)  # 100km above

        d = loc1.distance_to(loc2)
        assert d == pytest.approx(100.0, abs=0.1)

    def test_eci_normalization(self):
        """ECI coordinates should normalize to [-1, 1]."""
        loc = ECILocation(x=10000.0, y=20000.0, z=5000.0)
        x1, x2, x3 = loc.to_normalized(max_distance_km=50000.0)

        assert -1 <= x1 <= 1
        assert -1 <= x2 <= 1
        assert -1 <= x3 <= 1


# =============================================================================
# SECTION 12: ADVERSARIAL TESTS (MUST FAIL)
# =============================================================================

@pytest.mark.xfail(reason="SECURITY: Geographic spoofing must be detected", strict=True)
class TestGeographicSpoofingMustFail:
    """
    These tests attempt to bypass geographic security.
    ALL tests MUST FAIL (xfail). If any PASS, security is compromised.
    """

    def test_spoofed_location_low_risk(self):
        """MUST FAIL: Spoofed distant location should NOT have low risk."""
        home_gps = GPSLocation(40.7128, -74.0060)
        home_device = DeviceContext(device_id="home", biometric_score=0.9, threat_level=0.0)

        # Attacker claims to be in Sydney
        spoofed_gps = GPSLocation(-33.8688, 151.2093)

        home = Intent6DVector.from_context(home_gps, home_device)
        spoofed = Intent6DVector.from_context(spoofed_gps, home_device)

        d, H, decision = compute_location_risk(spoofed, home)

        # This SHOULD fail - spoofed location should have HIGH risk
        assert H < 1.5, f"Spoofed location correctly has high risk: H={H}"
        assert decision == "ALLOW", f"Spoofed location correctly blocked: {decision}"

    def test_teleportation_undetected(self):
        """MUST FAIL: Impossible travel must be detected."""
        t0 = time.time()

        loc_nyc = GPSLocation(40.7128, -74.0060, timestamp=t0)
        loc_tokyo = GPSLocation(35.6762, 139.6503, timestamp=t0 + 60)  # 1 minute later!

        is_impossible, _, _ = detect_impossible_travel(loc_nyc, loc_tokyo)

        # This SHOULD fail - teleportation must be detected
        assert is_impossible == False, "Teleportation was correctly detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
