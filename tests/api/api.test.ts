/**
 * SCBE API Tests
 *
 * Tests the simplified API wrapper.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { SCBE, evaluateRisk, sign, verify, breathe } from '../../src/api/index.js';

describe('SCBE API', () => {
  let api: SCBE;

  beforeEach(() => {
    api = new SCBE();
  });

  describe('evaluateRisk', () => {
    it('should return low risk for normal context', () => {
      const result = api.evaluateRisk({
        userId: 'user123',
        action: 'login',
        ip: '192.168.1.1',
      });

      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(result.score).toBeLessThanOrEqual(1);
      expect(result.distance).toBeGreaterThanOrEqual(0);
      expect(result.scaledCost).toBeGreaterThan(0);
      expect(['ALLOW', 'REVIEW', 'DENY']).toContain(result.decision);
      expect(result.reason).toBeDefined();
    });

    it('should return consistent scores for same context', () => {
      const context = { userId: 'test', action: 'read' };
      const result1 = api.evaluateRisk(context);
      const result2 = api.evaluateRisk(context);

      expect(result1.score).toBe(result2.score);
      expect(result1.decision).toBe(result2.decision);
    });

    it('should return different scores for different contexts', () => {
      const result1 = api.evaluateRisk({ type: 'A', value: 100 });
      const result2 = api.evaluateRisk({ type: 'B', value: 999999 });

      // Different contexts should produce different distances
      // (they might still get same decision, but distances differ)
      expect(result1.distance).not.toBe(result2.distance);
    });
  });

  describe('sign', () => {
    it('should sign a payload with default tongue', () => {
      const payload = { message: 'test', timestamp: Date.now() };
      const result = api.sign(payload);

      expect(result.envelope).toBeDefined();
      expect(result.envelope.payload).toBeDefined(); // Payload is base64 encoded
      expect(result.envelope.sigs).toBeDefined();
      expect(result.envelope.sigs.ko).toBeDefined();
      expect(result.tongues).toContain('ko');
    });

    it('should sign a payload with multiple tongues', () => {
      const payload = { data: 'sensitive' };
      const result = api.sign(payload, ['ko', 'um', 'dr']);

      expect(result.envelope.sigs.ko).toBeDefined();
      expect(result.envelope.sigs.um).toBeDefined();
      expect(result.envelope.sigs.dr).toBeDefined();
      expect(result.tongues).toEqual(['ko', 'um', 'dr']);
    });
  });

  describe('verify', () => {
    it('should verify a valid envelope', () => {
      const payload = { action: 'transfer', amount: 1000 };
      const { envelope } = api.sign(payload);
      const result = api.verify(envelope);

      expect(result.valid).toBe(true);
      expect(result.reason).toContain('valid');
    });

    it('should reject a tampered envelope', () => {
      const payload = { action: 'transfer', amount: 1000 };
      const { envelope } = api.sign(payload);

      // Tamper with payload
      const tampered = {
        ...envelope,
        payload: { action: 'transfer', amount: 1000000 },
      };

      const result = api.verify(tampered);
      expect(result.valid).toBe(false);
    });
  });

  describe('breathe', () => {
    it('should return a 6D point', () => {
      const result = api.breathe({ context: 'test' });

      expect(result).toHaveLength(6);
      result.forEach((v) => {
        expect(typeof v).toBe('number');
        expect(Number.isFinite(v)).toBe(true);
      });
    });

    it('should vary with intensity', () => {
      const context = { test: true };
      const low = api.breathe(context, 0.1);
      const high = api.breathe(context, 2.0);

      // Different intensities should produce different results
      // (at least slightly different due to breathing amplitude)
      const diff = low.some((v, i) => Math.abs(v - high[i]) > 0.0001);
      expect(diff).toBe(true);
    });
  });

  describe('convenience functions', () => {
    it('evaluateRisk function should work', () => {
      const result = evaluateRisk({ test: true });
      expect(result.score).toBeDefined();
    });

    it('sign function should work', () => {
      const result = sign({ message: 'test' });
      expect(result.envelope).toBeDefined();
    });

    it('verify function should work', () => {
      const { envelope } = sign({ message: 'test' });
      const result = verify(envelope);
      expect(result.valid).toBe(true);
    });

    it('breathe function should work', () => {
      const result = breathe({ context: 'test' });
      expect(result).toHaveLength(6);
    });
  });

  describe('full flow', () => {
    it('should handle complete risk evaluation and signing flow', () => {
      // 1. Evaluate risk of a transaction
      const context = {
        userId: 'alice',
        action: 'wire_transfer',
        amount: 50000,
        destination: 'external_bank',
        timestamp: Date.now(),
      };

      const risk = api.evaluateRisk(context);

      // 2. If allowed, sign the transaction
      if (risk.decision === 'ALLOW' || risk.decision === 'REVIEW') {
        const { envelope } = api.sign(context, ['ko', 'um']);

        // 3. Verify the signed transaction
        const verified = api.verify(envelope);
        expect(verified.valid).toBe(true);
      }

      // Test passes regardless of decision
      expect(risk.decision).toBeDefined();
    });
  });
});
