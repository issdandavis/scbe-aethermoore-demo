/**
 * SCBE-AETHERMOORE Simple API
 *
 * This is the "steering wheel" - a simple interface to the physics engine.
 *
 * Usage:
 *   import { SCBE } from './api';
 *
 *   // Evaluate risk of a context
 *   const risk = SCBE.evaluateRisk({ userId: '123', action: 'transfer', amount: 10000 });
 *
 *   // Sign a payload
 *   const envelope = SCBE.sign(payload);
 *
 *   // Verify an envelope
 *   const valid = SCBE.verify(envelope);
 */

import { randomBytes } from 'crypto';
import {
  projectToBall,
  hyperbolicDistance,
  breathTransform,
  harmonicScale,
  LanguesMetric,
  type BreathConfig,
} from '../harmonic/index.js';

import {
  signRoundtable,
  verifyRoundtable,
  clearNonceCache,
  type Keyring,
  type RoundtableEnvelope,
} from '../spiralverse/index.js';

// ============================================================
// Types
// ============================================================

export interface Context {
  [key: string]: unknown;
}

export interface RiskResult {
  score: number;           // 0-1, higher = more risky
  distance: number;        // Hyperbolic distance from safe center
  scaledCost: number;      // Exponential cost to attack
  decision: 'ALLOW' | 'REVIEW' | 'DENY';
  reason: string;
}

export interface SignResult {
  envelope: RoundtableEnvelope;
  tongues: string[];
}

export interface VerifyResult {
  valid: boolean;
  reason?: string;
}

// ============================================================
// Configuration
// ============================================================

const DEFAULT_KEYRING: Keyring = {
  ko: randomBytes(32),
  av: randomBytes(32),
  ru: randomBytes(32),
  ca: randomBytes(32),
  um: randomBytes(32),
  dr: randomBytes(32),
};

const SAFE_CENTER = [0, 0, 0, 0, 0, 0];
const RISK_THRESHOLDS = {
  ALLOW: 0.3,
  REVIEW: 0.7,
};

// ============================================================
// Core API
// ============================================================

export class SCBE {
  private keyring: Keyring;
  private metric: LanguesMetric;

  constructor(keyring?: Keyring) {
    this.keyring = keyring || DEFAULT_KEYRING;
    this.metric = new LanguesMetric();
  }

  /**
   * Evaluate the risk of a context/action.
   * Returns a risk score and decision.
   */
  evaluateRisk(context: Context): RiskResult {
    // Convert context to 6D point
    const point = this.contextToPoint(context);

    // Project to Poincaré ball (ensures point is valid)
    const projected = projectToBall(point);

    // Compute hyperbolic distance from safe center
    const distance = hyperbolicDistance(projected, SAFE_CENTER);

    // Apply harmonic scaling (exponential cost)
    const d = Math.max(1, Math.ceil(distance));
    const scaledCost = harmonicScale(d, 1.5);

    // Normalize to 0-1 risk score
    const score = Math.min(1, distance / 5);

    // Make decision
    let decision: 'ALLOW' | 'REVIEW' | 'DENY';
    let reason: string;

    if (score < RISK_THRESHOLDS.ALLOW) {
      decision = 'ALLOW';
      reason = 'Context within safe zone';
    } else if (score < RISK_THRESHOLDS.REVIEW) {
      decision = 'REVIEW';
      reason = 'Context requires review - moderate deviation';
    } else {
      decision = 'DENY';
      reason = 'Context exceeds safe threshold - high risk';
    }

    return {
      score,
      distance,
      scaledCost,
      decision,
      reason,
    };
  }

  /**
   * Sign a payload using RWP multi-signature envelope.
   */
  sign(payload: unknown, tongues: string[] = ['ko']): SignResult {
    clearNonceCache();
    const envelope = signRoundtable(
      payload,
      tongues[0] as 'ko' | 'av' | 'ru' | 'ca' | 'um' | 'dr',
      'scbe-api',
      this.keyring,
      tongues as ('ko' | 'av' | 'ru' | 'ca' | 'um' | 'dr')[]
    );
    return { envelope, tongues };
  }

  /**
   * Verify an envelope signature.
   */
  verify(envelope: RoundtableEnvelope): VerifyResult {
    const result = verifyRoundtable(envelope, this.keyring);
    return {
      valid: result.valid,
      reason: result.valid ? 'Signature valid' : 'Signature invalid or tampered',
    };
  }

  /**
   * Apply breathing transform to a context point.
   * Used for dynamic security adaptation.
   */
  breathe(context: Context, intensity: number = 1.0): number[] {
    const point = this.contextToPoint(context);
    const projected = projectToBall(point);
    const config: BreathConfig = { amplitude: 0.1 * intensity, omega: 1.0 };
    return breathTransform(projected, Date.now() / 1000, config);
  }

  /**
   * Get the keyring (for advanced usage).
   */
  getKeyring(): Keyring {
    return this.keyring;
  }

  /**
   * Set a custom keyring.
   */
  setKeyring(keyring: Keyring): void {
    this.keyring = keyring;
  }

  // ============================================================
  // Private Methods
  // ============================================================

  private contextToPoint(context: Context): number[] {
    // Convert arbitrary context to 6D point using hash-based mapping
    const str = JSON.stringify(context);
    const hash = this.simpleHash(str);

    // Map hash to 6 dimensions, scaled to reasonable range
    const point: number[] = [];
    for (let i = 0; i < 6; i++) {
      // Extract 4 characters at a time, convert to number, scale to [-2, 2]
      const slice = hash.slice(i * 4, i * 4 + 4);
      const num = parseInt(slice, 16) / 65535; // 0 to 1
      point.push((num - 0.5) * 4); // -2 to 2
    }

    return point;
  }

  private simpleHash(str: string): string {
    // Simple hash for context-to-point mapping
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    // Pad to 24 hex chars (6 dimensions * 4 chars each)
    const hex = Math.abs(hash).toString(16).padStart(8, '0');
    return (hex + hex + hex).slice(0, 24);
  }
}

// ============================================================
// Singleton Export for Simple Usage
// ============================================================

export const scbe = new SCBE();

// ============================================================
// Convenience Functions
// ============================================================

export function evaluateRisk(context: Context): RiskResult {
  return scbe.evaluateRisk(context);
}

export function sign(payload: unknown, tongues?: string[]): SignResult {
  return scbe.sign(payload, tongues);
}

export function verify(envelope: RoundtableEnvelope): VerifyResult {
  return scbe.verify(envelope);
}

export function breathe(context: Context, intensity?: number): number[] {
  return scbe.breathe(context, intensity);
}
