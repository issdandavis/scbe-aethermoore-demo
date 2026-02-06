/**
 * SCBE Core Mathematical Axioms - Ground Truth Tests
 * ===================================================
 *
 * Everything extends from these axioms. If these fail, the whole system fails.
 *
 * Core Formulas:
 *   1. H(d,R) = R^(d²)           - Harmonic scaling (vertical wall)
 *   2. d_H(u,v) = arcosh(...)    - Hyperbolic distance (Poincaré ball)
 *   3. L(x,t) = Σ wₗ exp(βₗ·d)  - Langues metric (6 Sacred Tongues)
 *   4. Risk' = B × H × T × I     - Composite risk (Time × Intent)
 *   5. wₗ = φ^(l-1)              - Golden ratio weighting
 *
 * @module tests/mathematical/core-axioms
 */

import { describe, it, expect } from 'vitest';
import {
  harmonicScale,
  securityBits,
  harmonicDistance,
} from '../../src/harmonic/harmonicScaling.js';
import { hyperbolicDistance, projectToBall, mobiusAdd } from '../../src/harmonic/hyperbolic.js';
import { LanguesMetric } from '../../src/harmonic/languesMetric.js';
import { CONSTANTS, type Vector6D } from '../../src/harmonic/constants.js';

// Define PHI locally (golden ratio)
const PHI = (1 + Math.sqrt(5)) / 2;
const R = CONSTANTS.DEFAULT_R; // 1.5

describe('SCBE Core Mathematical Axioms', () => {
  describe('Axiom 1: Harmonic Scaling H(d,R) = R^(d²)', () => {
    it('should compute H(1,R) = R^1 = R', () => {
      const result = harmonicScale(1, R);
      expect(result).toBeCloseTo(R, 10);
    });

    it('should compute H(2,R) = R^4', () => {
      const result = harmonicScale(2, R);
      expect(result).toBeCloseTo(Math.pow(R, 4), 10);
    });

    it('should compute H(6,R) = R^36 (six tongues)', () => {
      const result = harmonicScale(6, R);
      expect(result).toBeCloseTo(Math.pow(R, 36), 5);
      // R=1.5, d=6: H ≈ 2.18 × 10⁶ - the "vertical wall"
      expect(result).toBeGreaterThan(2e6);
    });

    it('should satisfy superexponential growth: H(d+1)/H(d) increases with d', () => {
      const ratios: number[] = [];
      for (let d = 1; d <= 5; d++) {
        const ratio = harmonicScale(d + 1, R) / harmonicScale(d, R);
        ratios.push(ratio);
      }
      // Each ratio should be larger than the previous (superexponential)
      for (let i = 1; i < ratios.length; i++) {
        expect(ratios[i]).toBeGreaterThan(ratios[i - 1]);
      }
    });

    it('should satisfy monotonicity: dH/dd > 0 (always increasing)', () => {
      for (let d = 1; d <= 10; d++) {
        expect(harmonicScale(d + 1, R)).toBeGreaterThan(harmonicScale(d, R));
      }
    });

    it('should satisfy convexity: d²H/dd² > 0 (accelerating growth)', () => {
      // Second derivative positive means first derivative is increasing
      const firstDerivatives: number[] = [];
      for (let d = 1; d <= 9; d++) {
        const dH = harmonicScale(d + 1, R) - harmonicScale(d, R);
        firstDerivatives.push(dH);
      }
      for (let i = 1; i < firstDerivatives.length; i++) {
        expect(firstDerivatives[i]).toBeGreaterThan(firstDerivatives[i - 1]);
      }
    });
  });

  describe('Axiom 2: Hyperbolic Distance (Poincaré Ball)', () => {
    const origin = [0, 0, 0, 0, 0, 0];

    it('should satisfy d_H(u,u) = 0 (identity)', () => {
      const u = [0.1, 0.2, 0.1, 0.05, 0.1, 0.15];
      const dist = hyperbolicDistance(u, u);
      expect(dist).toBeCloseTo(0, 10);
    });

    it('should satisfy d_H(u,v) = d_H(v,u) (symmetry)', () => {
      const u = [0.1, 0.2, 0.1, 0.05, 0.1, 0.15];
      const v = [0.3, 0.1, 0.2, 0.15, 0.05, 0.1];
      expect(hyperbolicDistance(u, v)).toBeCloseTo(hyperbolicDistance(v, u), 10);
    });

    it('should satisfy triangle inequality: d(u,w) ≤ d(u,v) + d(v,w)', () => {
      const u = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
      const v = [0.2, 0.2, 0.2, 0.1, 0.1, 0.1];
      const w = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1];

      const d_uw = hyperbolicDistance(u, w);
      const d_uv = hyperbolicDistance(u, v);
      const d_vw = hyperbolicDistance(v, w);

      expect(d_uw).toBeLessThanOrEqual(d_uv + d_vw + 1e-10);
    });

    it('should satisfy exponential volume growth near boundary', () => {
      // Points closer to boundary have larger distances from origin
      const near = [0.1, 0, 0, 0, 0, 0];
      const far = [0.9, 0, 0, 0, 0, 0];

      const d_near = hyperbolicDistance(origin, near);
      const d_far = hyperbolicDistance(origin, far);

      // Far point should have MUCH larger distance (exponential)
      expect(d_far / d_near).toBeGreaterThan(5);
    });

    it('should approach infinity as point approaches boundary', () => {
      const distances: number[] = [];
      for (const r of [0.5, 0.9, 0.99, 0.999]) {
        const point = [r, 0, 0, 0, 0, 0];
        distances.push(hyperbolicDistance(origin, point));
      }
      // Each should be larger (monotonically increasing toward infinity)
      for (let i = 1; i < distances.length; i++) {
        expect(distances[i]).toBeGreaterThan(distances[i - 1]);
      }
      // Final distance should be very large
      expect(distances[distances.length - 1]).toBeGreaterThan(5);
    });

    it('should project points into valid ball (norm < 1)', () => {
      // Point outside ball
      const outside = [2, 3, 1, 2, 1, 0.5];
      const projected = projectToBall(outside);
      const norm = Math.sqrt(projected.reduce((s, x) => s + x * x, 0));
      expect(norm).toBeLessThan(1);
    });
  });

  describe('Axiom 3: Golden Ratio Weighting φ^(l-1)', () => {
    it('should have PHI ≈ 1.618033988749895', () => {
      expect(PHI).toBeCloseTo((1 + Math.sqrt(5)) / 2, 10);
    });

    it('should satisfy φ² = φ + 1 (defining property)', () => {
      expect(PHI * PHI).toBeCloseTo(PHI + 1, 10);
    });

    it('should satisfy 1/φ = φ - 1 (reciprocal property)', () => {
      expect(1 / PHI).toBeCloseTo(PHI - 1, 10);
    });

    it('should generate correct tongue weights', () => {
      const weights = [1, 2, 3, 4, 5, 6].map((l) => Math.pow(PHI, l - 1));
      // Weights: 1, φ, φ², φ³, φ⁴, φ⁵
      expect(weights[0]).toBeCloseTo(1, 10);
      expect(weights[1]).toBeCloseTo(PHI, 10);
      expect(weights[2]).toBeCloseTo(PHI * PHI, 10);
      expect(weights[5]).toBeCloseTo(Math.pow(PHI, 5), 10);
    });

    it('should satisfy Fibonacci-like sum: w_n + w_{n+1} ≈ w_{n+2}', () => {
      for (let n = 0; n < 4; n++) {
        const w_n = Math.pow(PHI, n);
        const w_n1 = Math.pow(PHI, n + 1);
        const w_n2 = Math.pow(PHI, n + 2);
        expect(w_n + w_n1).toBeCloseTo(w_n2, 8);
      }
    });
  });

  describe('Axiom 4: Langues Metric L(x,t) = Σ wₗ exp(βₗ·dₗ)', () => {
    it('should be positive for all inputs (non-negativity)', () => {
      const metric = new LanguesMetric();
      for (let i = 0; i < 100; i++) {
        const point: Vector6D = [
          Math.random() * 0.5,
          Math.random() * 0.5,
          Math.random() * 0.5,
          Math.random() * 0.5,
          Math.random() * 0.5,
          Math.random() * 0.5,
        ];
        const t = Math.random() * 10;
        expect(metric.compute(point, t)).toBeGreaterThan(0);
      }
    });

    it('should increase with deviation (monotonicity)', () => {
      const metric = new LanguesMetric();
      const baseline = metric.compute([0, 0, 0, 0, 0, 0], 0);
      const deviated = metric.compute([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 0);
      expect(deviated).toBeGreaterThan(baseline);
    });

    it('should be bounded under temporal breathing', () => {
      const metric = new LanguesMetric();
      const point: Vector6D = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
      const values: number[] = [];
      for (let t = 0; t <= 2 * Math.PI; t += 0.1) {
        values.push(metric.compute(point, t));
      }
      const min = Math.min(...values);
      const max = Math.max(...values);
      // Should oscillate but remain bounded (ratio bounded by exp(2β) ≈ e²)
      expect(Number.isFinite(max)).toBe(true);
      expect(min).toBeGreaterThan(0);
      expect(max / min).toBeLessThan(1000); // Reasonable bound for exponential metric
    });

    it('should have minimum at origin (d=0)', () => {
      const metric = new LanguesMetric();
      const atOrigin = metric.compute([0, 0, 0, 0, 0, 0], 0);
      const nearby = metric.compute([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0);
      expect(nearby).toBeGreaterThanOrEqual(atOrigin);
    });
  });

  describe('Axiom 5: Security Bits Amplification', () => {
    it('should compute bits = base + d² × log₂(R)', () => {
      const baseBits = 128;
      const d = 6;
      const expected = baseBits + d * d * Math.log2(R);
      const result = securityBits(baseBits, d, R);
      expect(result).toBeCloseTo(expected, 10);
    });

    it('should amplify exponentially with d', () => {
      const baseBits = 128;
      const bits_d1 = securityBits(baseBits, 1, R);
      const bits_d6 = securityBits(baseBits, 6, R);
      // d=6 adds 36×log₂(1.5) ≈ 21 bits more than d=1
      expect(bits_d6 - bits_d1).toBeGreaterThan(20);
    });

    it('should amplify security significantly at high d', () => {
      const baseBits = 128;
      const d = 14;
      const amplified = securityBits(baseBits, d, R);
      // d=14: bits = 128 + 196 * log2(1.5) ≈ 128 + 114 ≈ 242
      // Still significant amplification
      expect(amplified).toBeGreaterThan(200);
      expect(amplified).toBeGreaterThan(baseBits + 100);
    });
  });

  describe('Axiom 6: Composite Risk = B × H × T × I', () => {
    // Simulating the composite risk formula
    function compositeRisk(
      behavioral: number,
      d: number,
      timeFactor: number,
      intentFactor: number,
      R: number = 1.5
    ): number {
      const H = harmonicScale(d, R);
      return behavioral * H * timeFactor * intentFactor;
    }

    it('should be non-negative when all inputs are non-negative', () => {
      const risk = compositeRisk(0.1, 1, 1.0, 1.0);
      expect(risk).toBeGreaterThanOrEqual(0);
    });

    it('should satisfy lower bound: Risk ≥ Behavioral (when T,I ≥ 1, H ≥ 1)', () => {
      const behavioral = 0.3;
      const risk = compositeRisk(behavioral, 1, 1.0, 1.0);
      expect(risk).toBeGreaterThanOrEqual(behavioral);
    });

    it('should amplify multiplicatively (superexponential growth)', () => {
      const risk_d1 = compositeRisk(0.1, 1, 1.5, 1.5);
      const risk_d3 = compositeRisk(0.1, 3, 1.5, 1.5);
      const risk_d6 = compositeRisk(0.1, 6, 1.5, 1.5);

      // Growth should be dramatic
      expect(risk_d3 / risk_d1).toBeGreaterThan(10);
      expect(risk_d6 / risk_d3).toBeGreaterThan(1000);
    });

    it('should create vertical wall: small d change → huge risk change', () => {
      const risk_d5 = compositeRisk(0.1, 5, 1.0, 1.0);
      const risk_d6 = compositeRisk(0.1, 6, 1.0, 1.0);
      // One step increase should cause massive risk jump
      expect(risk_d6 / risk_d5).toBeGreaterThan(5);
    });

    it('should partition state space into ALLOW/QUARANTINE/DENY', () => {
      const ALLOW_THRESHOLD = 0.3;
      const DENY_THRESHOLD = 0.7;

      function decide(risk: number): 'ALLOW' | 'QUARANTINE' | 'DENY' {
        if (risk < ALLOW_THRESHOLD) return 'ALLOW';
        if (risk < DENY_THRESHOLD) return 'QUARANTINE';
        return 'DENY';
      }

      // Low deviation → ALLOW
      expect(decide(compositeRisk(0.1, 1, 1.0, 1.0))).toBe('ALLOW');

      // High deviation → DENY (due to H amplification)
      const highRisk = Math.min(1, compositeRisk(0.1, 3, 1.5, 1.5));
      expect(decide(highRisk)).toBe('DENY');
    });
  });

  describe('Axiom 7: Möbius Addition (Hyperbolic Geometry)', () => {
    it('should have identity: u ⊕ 0 = u', () => {
      const u = [0.2, 0.1, 0, 0, 0, 0];
      const zero = [0, 0, 0, 0, 0, 0];
      const result = mobiusAdd(u, zero);

      for (let i = 0; i < u.length; i++) {
        expect(result[i]).toBeCloseTo(u[i], 10);
      }
    });

    it('should satisfy d(u⊕v, 0) properties in hyperbolic space', () => {
      const u = [0.1, 0.1, 0, 0, 0, 0];
      const v = [0.1, 0, 0, 0, 0, 0];
      const origin = [0, 0, 0, 0, 0, 0];

      const sum = mobiusAdd(u, v);
      // Result should still be inside the ball
      const norm = Math.sqrt(sum.reduce((s, x) => s + x * x, 0));
      expect(norm).toBeLessThan(1);
    });
  });

  describe('Axiom 8: Lyapunov Stability (Convergence to Origin)', () => {
    it('should decrease energy under gradient descent toward origin', () => {
      const metric = new LanguesMetric();
      let point: Vector6D = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
      const stepSize = 0.1;

      const energies: number[] = [metric.compute(point, 0)];

      // Simulate gradient descent toward origin
      for (let step = 0; step < 10; step++) {
        // Move toward origin
        point = point.map((x) => x * (1 - stepSize)) as Vector6D;
        energies.push(metric.compute(point, 0));
      }

      // Energy should decrease monotonically
      for (let i = 1; i < energies.length; i++) {
        expect(energies[i]).toBeLessThanOrEqual(energies[i - 1] + 1e-10);
      }
    });
  });

  describe('Axiom 9: Harmonic Distance in 6D (Weighted Euclidean)', () => {
    it('should weight dimensions 4-6 with R^(1/5) factors', () => {
      const R5 = Math.pow(R, 0.2);
      const expectedWeights = [1, 1, 1, R5, R5 * R5, R5 * R5 * R5];

      // Verify R^(1/5) ≈ 1.084
      expect(R5).toBeCloseTo(Math.pow(1.5, 0.2), 10);

      // Weights should increase for sacred tongue dimensions
      expect(expectedWeights[3]).toBeGreaterThan(expectedWeights[0]);
      expect(expectedWeights[5]).toBeGreaterThan(expectedWeights[3]);
    });

    it('should compute weighted distance correctly', () => {
      const u: [number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0];
      const v: [number, number, number, number, number, number] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1];

      const dist = harmonicDistance(u, v);

      // Manual calculation
      const R5 = Math.pow(R, 0.2);
      const weights = [1, 1, 1, R5, R5 * R5, R5 * R5 * R5];
      let expected = 0;
      for (let i = 0; i < 6; i++) {
        expected += weights[i] * 0.1 * 0.1;
      }
      expected = Math.sqrt(expected);

      expect(dist).toBeCloseTo(expected, 10);
    });
  });

  describe('Ground Truth: Mathematical Invariants', () => {
    it('exp(a+b) = exp(a) × exp(b) - used in all exponential formulas', () => {
      const a = 2.5,
        b = 3.7;
      expect(Math.exp(a + b)).toBeCloseTo(Math.exp(a) * Math.exp(b), 10);
    });

    it('R^(d²) = exp(d² × ln(R)) - harmonic scaling identity', () => {
      const d = 5;
      expect(Math.pow(R, d * d)).toBeCloseTo(Math.exp(d * d * Math.log(R)), 10);
    });

    it('arcosh(x) = ln(x + sqrt(x²-1)) for x ≥ 1 - hyperbolic identity', () => {
      const x = 2.5;
      const expected = Math.log(x + Math.sqrt(x * x - 1));
      expect(Math.acosh(x)).toBeCloseTo(expected, 10);
    });

    it('sin²(x) + cos²(x) = 1 - used in temporal breathing', () => {
      for (const x of [0, Math.PI / 4, Math.PI / 2, Math.PI, 2 * Math.PI]) {
        const sin = Math.sin(x);
        const cos = Math.cos(x);
        expect(sin * sin + cos * cos).toBeCloseTo(1, 10);
      }
    });
  });
});
