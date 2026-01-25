/**
 * SCBE Integration Module
 *
 * Connects SCBE security to AI Workflow Architect and other systems.
 *
 * @module integration
 */

export {
  SCBEGateway,
  scbeGateway,
  scbeMiddleware,
  type AgentRequest,
  type GatewayResponse,
  type PhysicsTrap,
  type SecurityDecision,
} from './scbe-gateway.js';
