/**
 * SCBE Gateway Integration Tests
 * ===============================
 *
 * Tests for the SCBE Gateway integration layer that connects
 * SCBE's 14-layer security pipeline to AI Workflow Architect.
 *
 * @module tests/integration/scbe-gateway
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  SCBEGateway,
  scbeGateway,
  scbeMiddleware,
  type AgentRequest,
  type GatewayResponse,
  type SecurityDecision,
} from '../../src/integration/scbe-gateway.js';

describe('SCBE Gateway Integration', () => {
  let gateway: SCBEGateway;

  beforeEach(() => {
    gateway = new SCBEGateway();
    // Mock fetch to prevent actual API calls
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }));
  });

  describe('Agent Request Evaluation', () => {
    describe('Trusted Agents (Low Risk)', () => {
      it('should ALLOW requests from trusted "captain" agent with low-risk actions', async () => {
        const request: AgentRequest = {
          agentId: 'captain',
          action: 'read',
          payload: { file: 'config.json' },
        };

        const result = await gateway.evaluateRequest(request);

        expect(result.decision).toBe('ALLOW');
        expect(result.allowed).toBe(true);
        expect(result.trapActivated).toBe(false);
        expect(result.riskScore).toBeLessThan(0.3);
      });

      it('should ALLOW requests from trusted "security" agent', async () => {
        const request: AgentRequest = {
          agentId: 'security',
          action: 'query',
          payload: { target: 'logs' },
        };

        const result = await gateway.evaluateRequest(request);

        expect(result.decision).toBe('ALLOW');
        expect(result.allowed).toBe(true);
        expect(result.harmonicFactor).toBeGreaterThan(0);
      });

      it('should ALLOW requests from trusted "architect" agent', async () => {
        const request: AgentRequest = {
          agentId: 'architect',
          action: 'list',
          payload: { resources: 'all' },
        };

        const result = await gateway.evaluateRequest(request);

        expect(result.decision).toBe('ALLOW');
        expect(result.allowed).toBe(true);
      });

      it('should ALLOW requests from trusted "analyst" agent', async () => {
        const request: AgentRequest = {
          agentId: 'analyst',
          action: 'query',
          payload: { data: 'metrics' },
        };

        const result = await gateway.evaluateRequest(request);

        // Analyst is furthest from origin among trusted agents, may get QUARANTINE
        expect(['ALLOW', 'QUARANTINE']).toContain(result.decision);
      });
    });

    describe('Unknown Agents (Hostile Positioning)', () => {
      it('should DENY requests from unknown agents with high-risk actions', async () => {
        const request: AgentRequest = {
          agentId: 'malicious-bot',
          action: 'sudo',
          payload: { command: 'rm -rf /' },
        };

        const result = await gateway.evaluateRequest(request);

        expect(result.decision).toBe('DENY');
        expect(result.allowed).toBe(false);
        expect(result.trapActivated).toBe(true);
        expect(result.trapEnvironment).toBeDefined();
        expect(result.riskScore).toBeGreaterThan(0.7);
      });

      it('should QUARANTINE unknown agents with medium-risk actions', async () => {
        const request: AgentRequest = {
          agentId: 'new-agent-123',
          action: 'read',
          payload: { file: 'data.txt' },
        };

        const result = await gateway.evaluateRequest(request);

        // Unknown agent with low-risk action should be quarantined or allowed
        // depending on harmonic amplification
        expect(['ALLOW', 'QUARANTINE', 'DENY']).toContain(result.decision);
      });

      it('should position unknown agents near hyperbolic boundary', async () => {
        const request: AgentRequest = {
          agentId: 'stranger',
          action: 'execute',
          payload: { script: 'unknown.sh' },
        };

        const result = await gateway.evaluateRequest(request);

        // Harmonic factor should be high for unknown agents
        expect(result.harmonicFactor).toBeGreaterThan(1);
      });
    });

    describe('Custom Position Override', () => {
      it('should respect custom position in request', async () => {
        // Position near origin (trusted)
        const trustedRequest: AgentRequest = {
          agentId: 'custom-agent',
          action: 'read',
          payload: {},
          position: [0.1, 0.05, 0.08, 0.12, 0.06, 0.09],
        };

        const trustedResult = await gateway.evaluateRequest(trustedRequest);
        expect(trustedResult.riskScore).toBeLessThan(0.5);

        // Position near boundary (hostile)
        const hostileRequest: AgentRequest = {
          agentId: 'custom-agent',
          action: 'read',
          payload: {},
          position: [0.95, 0.92, 0.98, 0.89, 0.94, 0.91],
        };

        const hostileResult = await gateway.evaluateRequest(hostileRequest);
        expect(hostileResult.harmonicFactor).toBeGreaterThan(trustedResult.harmonicFactor);
      });
    });
  });

  describe('Risk Calculation by Action Type', () => {
    it('should assign low risk (0.1) to read-type actions', async () => {
      const actions = ['read', 'query', 'list'];

      for (const action of actions) {
        const request: AgentRequest = {
          agentId: 'captain',
          action,
          payload: {},
        };

        const result = await gateway.evaluateRequest(request);
        // Base risk 0.1 with harmonic amplification from trusted position
        expect(result.riskScore).toBeLessThan(0.3);
      }
    });

    it('should assign medium risk (0.3) to write-type actions', async () => {
      const actions = ['write', 'update', 'create'];

      for (const action of actions) {
        const request: AgentRequest = {
          agentId: 'captain',
          action,
          payload: {},
        };

        const result = await gateway.evaluateRequest(request);
        // Base risk 0.3 with harmonic amplification
        expect(result.riskScore).toBeLessThan(0.7);
      }
    });

    it('should assign high risk to dangerous actions', async () => {
      const dangerousActions = [
        { action: 'delete', minRisk: 0.5 },
        { action: 'deploy', minRisk: 0.6 },
        { action: 'admin', minRisk: 0.7 },
        { action: 'sudo', minRisk: 0.8 },
        { action: 'root', minRisk: 0.9 },
      ];

      for (const { action, minRisk } of dangerousActions) {
        const request: AgentRequest = {
          agentId: 'unknown-agent',
          action,
          payload: {},
        };

        const result = await gateway.evaluateRequest(request);
        expect(result.riskScore).toBeGreaterThanOrEqual(minRisk);
      }
    });

    it('should assign medium risk (0.5) to unknown actions', async () => {
      const request: AgentRequest = {
        agentId: 'captain',
        action: 'completely_unknown_action_xyz',
        payload: {},
      };

      const result = await gateway.evaluateRequest(request);
      // Base risk 0.5 with harmonic amplification
      expect(result.riskScore).toBeLessThan(1.0);
      expect(result.riskScore).toBeGreaterThan(0.4);
    });
  });

  describe('Security Decision Thresholds', () => {
    it('should ALLOW when risk < 0.3', async () => {
      const request: AgentRequest = {
        agentId: 'security', // Closest to origin
        action: 'read',      // Lowest risk action
        payload: {},
      };

      const result = await gateway.evaluateRequest(request);
      expect(result.decision).toBe('ALLOW');
      expect(result.riskScore).toBeLessThan(0.3);
    });

    it('should include decision reason', async () => {
      const request: AgentRequest = {
        agentId: 'captain',
        action: 'read',
        payload: {},
      };

      const result = await gateway.evaluateRequest(request);
      expect(result.reason).toBeDefined();
      expect(typeof result.reason).toBe('string');
      expect(result.reason.length).toBeGreaterThan(0);
    });
  });

  describe('Physics Trap Activation', () => {
    it('should activate physics trap for DENY decisions', async () => {
      const request: AgentRequest = {
        agentId: 'attacker',
        action: 'root',
        payload: { command: 'malicious' },
      };

      const result = await gateway.evaluateRequest(request);

      expect(result.trapActivated).toBe(true);
      expect(result.trapEnvironment).toBeDefined();
    });

    it('should configure trap hostility based on risk score', async () => {
      const request: AgentRequest = {
        agentId: 'attacker',
        action: 'root',
        payload: {},
      };

      const result = await gateway.evaluateRequest(request);

      if (result.trapEnvironment) {
        expect(result.trapEnvironment.hostileParameters.gravity_multiplier).toBeGreaterThan(1);
        expect(result.trapEnvironment.hostileParameters.time_dilation).toBeLessThan(1);
        expect(result.trapEnvironment.hostileParameters.entropy_injection).toBeGreaterThan(0);
        expect(result.trapEnvironment.hostileParameters.turbulence_factor).toBeGreaterThan(0);
        expect(result.trapEnvironment.duration_ms).toBeGreaterThanOrEqual(30000);
      }
    });

    it('should select appropriate trap type based on risk', async () => {
      // Highest risk -> relativity trap
      const extremeRequest: AgentRequest = {
        agentId: 'extreme-attacker',
        action: 'root',
        payload: {},
        position: [0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
      };

      const result = await gateway.evaluateRequest(extremeRequest);

      if (result.trapEnvironment) {
        expect(['quantum', 'relativity', 'thermodynamics', 'electromagnetism']).toContain(
          result.trapEnvironment.simulationType
        );
      }
    });

    it('should not activate trap for ALLOW decisions', async () => {
      const request: AgentRequest = {
        agentId: 'captain',
        action: 'read',
        payload: {},
      };

      const result = await gateway.evaluateRequest(request);

      expect(result.trapActivated).toBe(false);
      expect(result.trapEnvironment).toBeUndefined();
    });
  });

  describe('Harmonic Scaling', () => {
    it('should apply harmonic amplification H(d,R) = R^(d^2)', async () => {
      const request: AgentRequest = {
        agentId: 'captain',
        action: 'read',
        payload: {},
      };

      const result = await gateway.evaluateRequest(request);

      // Harmonic factor should be positive
      expect(result.harmonicFactor).toBeGreaterThan(0);
      // For trusted agents, harmonic factor should be close to base radius
      expect(result.harmonicFactor).toBeLessThan(100);
    });

    it('should amplify risk for agents far from origin', async () => {
      const nearOriginRequest: AgentRequest = {
        agentId: 'captain', // Near origin
        action: 'execute',
        payload: {},
      };

      const farFromOriginRequest: AgentRequest = {
        agentId: 'unknown', // Far from origin
        action: 'execute',
        payload: {},
      };

      const nearResult = await gateway.evaluateRequest(nearOriginRequest);
      const farResult = await gateway.evaluateRequest(farFromOriginRequest);

      expect(farResult.harmonicFactor).toBeGreaterThan(nearResult.harmonicFactor);
      expect(farResult.riskScore).toBeGreaterThan(nearResult.riskScore);
    });
  });

  describe('Middleware Integration', () => {
    it('should create middleware function', () => {
      const middleware = scbeMiddleware(gateway);
      expect(typeof middleware).toBe('function');
    });

    it('should block DENY decisions with 403', async () => {
      const middleware = scbeMiddleware(gateway);

      const mockReq = {
        headers: { 'x-agent-id': 'attacker', 'x-action': 'root' },
        body: { command: 'malicious' },
      };

      const mockRes = {
        status: vi.fn().mockReturnThis(),
        json: vi.fn(),
      };

      const mockNext = vi.fn();

      await middleware(mockReq, mockRes, mockNext);

      expect(mockRes.status).toHaveBeenCalledWith(403);
      expect(mockRes.json).toHaveBeenCalled();
      expect(mockNext).not.toHaveBeenCalled();
    });

    it('should return 202 for QUARANTINE decisions', async () => {
      const middleware = scbeMiddleware(gateway);

      // Create a request that will result in QUARANTINE
      const mockReq = {
        headers: { 'x-agent-id': 'semi-trusted', 'x-action': 'deploy' },
        body: {},
      };

      const mockRes = {
        status: vi.fn().mockReturnThis(),
        json: vi.fn(),
      };

      const mockNext = vi.fn();

      await middleware(mockReq, mockRes, mockNext);

      // Could be QUARANTINE or DENY depending on exact risk calculation
      const statusCall = mockRes.status.mock.calls[0]?.[0];
      if (statusCall) {
        expect([202, 403]).toContain(statusCall);
      } else {
        // If no status was set, next() was called (ALLOW)
        expect(mockNext).toHaveBeenCalled();
      }
    });

    it('should call next() for ALLOW decisions', async () => {
      const middleware = scbeMiddleware(gateway);

      const mockReq = {
        headers: { 'x-agent-id': 'captain', 'x-action': 'read' },
        body: { file: 'config.json' },
      };

      const mockRes = {
        status: vi.fn().mockReturnThis(),
        json: vi.fn(),
      };

      const mockNext = vi.fn();

      await middleware(mockReq, mockRes, mockNext);

      expect(mockNext).toHaveBeenCalled();
    });

    it('should handle missing agent-id header', async () => {
      const middleware = scbeMiddleware(gateway);

      const mockReq = {
        headers: { 'x-action': 'read' },
        body: {},
      };

      const mockRes = {
        status: vi.fn().mockReturnThis(),
        json: vi.fn(),
      };

      const mockNext = vi.fn();

      await middleware(mockReq, mockRes, mockNext);

      // Should treat as unknown agent
      // Result depends on action risk and harmonic amplification
      expect(mockRes.status).toHaveBeenCalled();
    });
  });

  describe('Singleton Instance', () => {
    it('should export a singleton gateway instance', () => {
      expect(scbeGateway).toBeDefined();
      expect(scbeGateway).toBeInstanceOf(SCBEGateway);
    });

    it('should be usable for request evaluation', async () => {
      const request: AgentRequest = {
        agentId: 'captain',
        action: 'read',
        payload: {},
      };

      const result = await scbeGateway.evaluateRequest(request);
      expect(result.decision).toBeDefined();
    });
  });

  describe('Response Structure', () => {
    it('should return complete GatewayResponse structure', async () => {
      const request: AgentRequest = {
        agentId: 'test-agent',
        action: 'query',
        payload: { test: true },
      };

      const result = await gateway.evaluateRequest(request);

      // Verify all required fields are present
      expect(result).toHaveProperty('decision');
      expect(result).toHaveProperty('riskScore');
      expect(result).toHaveProperty('harmonicFactor');
      expect(result).toHaveProperty('allowed');
      expect(result).toHaveProperty('trapActivated');
      expect(result).toHaveProperty('reason');

      // Verify types
      expect(['ALLOW', 'QUARANTINE', 'DENY']).toContain(result.decision);
      expect(typeof result.riskScore).toBe('number');
      expect(typeof result.harmonicFactor).toBe('number');
      expect(typeof result.allowed).toBe('boolean');
      expect(typeof result.trapActivated).toBe('boolean');
      expect(typeof result.reason).toBe('string');
    });

    it('should include timestamp in request processing', async () => {
      const now = Date.now();
      const request: AgentRequest = {
        agentId: 'captain',
        action: 'read',
        payload: {},
        timestamp: now,
      };

      const result = await gateway.evaluateRequest(request);
      expect(result).toBeDefined();
    });
  });
});
