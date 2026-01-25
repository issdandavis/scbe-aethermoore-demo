/**
 * SCBE Gateway - Integration Layer for AI Workflow Architect
 *
 * Wires SCBE's 14-layer security pipeline into agent request flow.
 *
 * Flow:
 *   Agent Request → SCBE Pipeline → Decision
 *     ├─ ALLOW → Proceed to action
 *     ├─ QUARANTINE → Hold for Sacred Tongue consensus
 *     └─ DENY → Redirect to Physics Trap (hostile environment)
 *
 * @module integration/scbe-gateway
 */

import { hyperbolicDistance, projectToBall } from '../harmonic/index.js';

/**
 * Harmonic amplification for continuous distance values
 * H(d, R) = R^(d²) where d is the scaled hyperbolic distance
 *
 * Unlike the discrete harmonicScale, this works with floating-point distances.
 */
function continuousHarmonicScale(distance: number, R: number): number {
  // Scale distance: trusted agents are near 0, hostile agents are near boundary (>1)
  // Use the square of distance for exponential amplification
  const d2 = distance * distance;
  return Math.pow(R, d2);
}

// Physics Simulator API endpoint
const PHYSICS_TRAP_API = process.env.PHYSICS_TRAP_API ||
  'https://ldrsy9yqs7.execute-api.us-west-2.amazonaws.com/';

// Decision types
export type SecurityDecision = 'ALLOW' | 'QUARANTINE' | 'DENY';

// Agent request structure
export interface AgentRequest {
  agentId: string;
  action: string;
  payload: unknown;
  position?: number[];  // 6D position in hyperbolic space
  timestamp?: number;
}

// Gateway response
export interface GatewayResponse {
  decision: SecurityDecision;
  riskScore: number;
  harmonicFactor: number;
  allowed: boolean;
  trapActivated: boolean;
  trapEnvironment?: PhysicsTrap;
  reason: string;
}

// Physics trap configuration for attackers
export interface PhysicsTrap {
  simulationType: 'quantum' | 'relativity' | 'thermodynamics' | 'electromagnetism';
  hostileParameters: {
    gravity_multiplier: number;
    time_dilation: number;
    entropy_injection: number;
    turbulence_factor: number;
  };
  duration_ms: number;
}

/**
 * Default agent positions in 6D hyperbolic space
 * Legitimate agents cluster near origin (low distance)
 * Attackers are pushed to boundary (high distance)
 */
const TRUSTED_AGENT_POSITIONS: Record<string, number[]> = {
  'captain': [0.1, 0.05, 0.08, 0.12, 0.06, 0.09],
  'architect': [0.15, 0.1, 0.12, 0.08, 0.11, 0.07],
  'security': [0.05, 0.03, 0.04, 0.06, 0.02, 0.05],
  'analyst': [0.2, 0.15, 0.18, 0.14, 0.16, 0.12],
};

// Thresholds
const ALLOW_THRESHOLD = 0.3;
const QUARANTINE_THRESHOLD = 0.7;

/**
 * SCBE Gateway - Main entry point
 *
 * Evaluates agent requests through the 14-layer security pipeline
 */
export class SCBEGateway {
  private originPosition = [0, 0, 0, 0, 0, 0];
  private baseRadius = 10.0;

  /**
   * Process an agent request through SCBE security
   */
  async evaluateRequest(request: AgentRequest): Promise<GatewayResponse> {
    const position = request.position || this.getAgentPosition(request.agentId);

    // Project position into Poincaré ball
    const projectedPosition = projectToBall(position);

    // Calculate hyperbolic distance from origin (trusted center)
    const distance = hyperbolicDistance(projectedPosition, this.originPosition);

    // Apply continuous harmonic scaling: H(d,R) = R^(d²)
    // Uses continuous distance instead of discrete dimension
    const H = continuousHarmonicScale(distance, this.baseRadius);

    // Base risk from action type
    const baseRisk = this.calculateBaseRisk(request.action);

    // Final risk with harmonic amplification
    const riskScore = Math.min(1.0, baseRisk * H);

    // Make decision
    const decision = this.makeDecision(riskScore);

    // Build response
    const response: GatewayResponse = {
      decision,
      riskScore,
      harmonicFactor: H,
      allowed: decision === 'ALLOW',
      trapActivated: decision === 'DENY',
      reason: this.getDecisionReason(decision, riskScore, H),
    };

    // If DENY, activate physics trap
    if (decision === 'DENY') {
      response.trapEnvironment = await this.activatePhysicsTrap(riskScore);
    }

    return response;
  }

  /**
   * Get known position for agent, or generate hostile position for unknown
   */
  private getAgentPosition(agentId: string): number[] {
    const known = TRUSTED_AGENT_POSITIONS[agentId.toLowerCase()];
    if (known) return known;

    // Unknown agent = hostile position near boundary
    return [0.85, 0.82, 0.88, 0.79, 0.84, 0.81];
  }

  /**
   * Calculate base risk from action type
   */
  private calculateBaseRisk(action: string): number {
    const riskMap: Record<string, number> = {
      'read': 0.1,
      'query': 0.1,
      'list': 0.1,
      'write': 0.3,
      'update': 0.3,
      'create': 0.3,
      'delete': 0.6,
      'execute': 0.5,
      'deploy': 0.7,
      'admin': 0.8,
      'sudo': 0.9,
      'root': 0.95,
    };

    const lowerAction = action.toLowerCase();
    for (const [key, risk] of Object.entries(riskMap)) {
      if (lowerAction.includes(key)) return risk;
    }

    return 0.5; // Unknown action = medium risk
  }

  /**
   * Make security decision based on risk score
   */
  private makeDecision(riskScore: number): SecurityDecision {
    if (riskScore < ALLOW_THRESHOLD) return 'ALLOW';
    if (riskScore < QUARANTINE_THRESHOLD) return 'QUARANTINE';
    return 'DENY';
  }

  /**
   * Generate human-readable reason for decision
   */
  private getDecisionReason(decision: SecurityDecision, risk: number, H: number): string {
    switch (decision) {
      case 'ALLOW':
        return `Risk ${(risk * 100).toFixed(1)}% within acceptable threshold. H(d,R)=${H.toFixed(3)}`;
      case 'QUARANTINE':
        return `Elevated risk ${(risk * 100).toFixed(1)}%. Requires Sacred Tongue consensus. H(d,R)=${H.toFixed(3)}`;
      case 'DENY':
        return `BLOCKED: Risk ${(risk * 100).toFixed(1)}% exceeds threshold. Harmonic amplification ${H.toFixed(1)}x. Activating physics trap.`;
    }
  }

  /**
   * Activate hostile physics environment for attacker
   */
  private async activatePhysicsTrap(riskScore: number): Promise<PhysicsTrap> {
    // Higher risk = more hostile environment
    const hostility = Math.min(10, 1 + riskScore * 9);

    const trap: PhysicsTrap = {
      simulationType: this.selectTrapType(riskScore),
      hostileParameters: {
        gravity_multiplier: hostility * 2,
        time_dilation: 1 / hostility,  // Slow their time
        entropy_injection: hostility * 0.5,
        turbulence_factor: hostility * 3,
      },
      duration_ms: 30000 + Math.floor(riskScore * 60000), // 30-90 seconds
    };

    // Fire and forget - send attacker to physics trap
    this.sendToPhysicsTrap(trap).catch(() => {
      // Trap activation is best-effort
    });

    return trap;
  }

  /**
   * Select trap type based on risk profile
   */
  private selectTrapType(riskScore: number): PhysicsTrap['simulationType'] {
    if (riskScore > 0.9) return 'relativity';      // Extreme time dilation
    if (riskScore > 0.8) return 'quantum';         // Uncertainty principle chaos
    if (riskScore > 0.7) return 'thermodynamics';  // Entropy nightmare
    return 'electromagnetism';                      // EM interference
  }

  /**
   * Send attacker to physics trap API
   */
  private async sendToPhysicsTrap(trap: PhysicsTrap): Promise<void> {
    try {
      await fetch(PHYSICS_TRAP_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          simulation_type: trap.simulationType,
          parameters: trap.hostileParameters,
          trap_mode: true,
        }),
      });
    } catch {
      // Trap is secondary - don't fail the main flow
    }
  }
}

/**
 * Middleware for Express/Fastify integration
 */
export function scbeMiddleware(gateway: SCBEGateway) {
  return async (req: { body?: AgentRequest; headers: Record<string, string> }, res: { status: (code: number) => { json: (data: unknown) => void } }, next: () => void) => {
    const agentId = req.headers['x-agent-id'] || 'unknown';
    const action = req.headers['x-action'] || 'unknown';

    const request: AgentRequest = {
      agentId,
      action,
      payload: req.body,
      timestamp: Date.now(),
    };

    const result = await gateway.evaluateRequest(request);

    if (result.decision === 'DENY') {
      return res.status(403).json({
        error: 'Access Denied',
        reason: result.reason,
        trap: result.trapEnvironment,
      });
    }

    if (result.decision === 'QUARANTINE') {
      return res.status(202).json({
        status: 'pending',
        reason: result.reason,
        requiresConsensus: true,
      });
    }

    // ALLOW - proceed
    next();
  };
}

// Export singleton instance
export const scbeGateway = new SCBEGateway();
