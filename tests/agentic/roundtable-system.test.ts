/**
 * SCBE-AETHERMOORE Roundtable System Tests
 * =========================================
 *
 * Tests for the agent fleet and governance system.
 *
 * @module tests/agentic/roundtable-system
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  AgentRegistry,
  TaskDispatcher,
  GovernanceManager,
  FleetManager,
  GOVERNANCE_TIERS,
  type GovernanceTier,
} from '../../src/fleet/index.js';

describe('SCBE-AETHERMOORE Roundtable System', () => {
  describe('Agent Registration & Trust Vectors', () => {
    let registry: AgentRegistry;

    beforeEach(() => {
      registry = new AgentRegistry();
    });

    it('should register agent with 6D trust vector', () => {
      const agent = registry.registerAgent({
        name: 'TestAgent',
        description: 'Test agent for trust vectors',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['code_generation', 'security_scan'],
      });

      expect(agent.id).toBeDefined();
      expect(agent.trustVector).toHaveLength(6);
      agent.trustVector.forEach(v => {
        expect(v).toBeCloseTo(0.5, 1);
      });
    });

    it('should compute initial trust score', () => {
      const agent = registry.registerAgent({
        name: 'TrustedAgent',
        description: 'Trusted agent with high initial trust',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['orchestration'],
        initialTrustVector: [0.8, 0.85, 0.9, 0.75, 0.88, 0.82],
      });

      expect(agent.trustScore).toBeDefined();
      // Trust score is an object with normalized value
      if (typeof agent.trustScore === 'object' && agent.trustScore !== null) {
        expect(agent.trustScore.normalized).toBeDefined();
      }
    });

    it('should generate unique spectral identity', () => {
      const agent1 = registry.registerAgent({
        name: 'Agent1',
        description: 'First agent for spectral test',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['code_generation'],
      });

      const agent2 = registry.registerAgent({
        name: 'Agent2',
        description: 'Second agent for spectral test',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['code_generation'],
      });

      expect(agent1.spectralIdentity).toBeDefined();
      expect(agent2.spectralIdentity).toBeDefined();
      expect(agent1.spectralIdentity).not.toBe(agent2.spectralIdentity);
    });

    it('should assign governance tier based on trust', () => {
      const agent = registry.registerAgent({
        name: 'NewAgent',
        description: 'New agent for governance test',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['documentation'],
      });

      expect(agent.maxGovernanceTier).toBeDefined();
    });

    it('should filter agents by capability', () => {
      registry.registerAgent({
        name: 'Coder',
        description: 'Code generation specialist',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['code_generation', 'code_review'],
      });

      registry.registerAgent({
        name: 'SecOps',
        description: 'Security operations specialist',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['security_scan', 'monitoring'],
      });

      const coders = registry.getAgentsByCapability('code_generation');
      expect(coders).toHaveLength(1);
      expect(coders[0].name).toBe('Coder');
    });

    it('should get all registered agents', () => {
      registry.registerAgent({
        name: 'Agent1',
        description: 'First test agent',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['code_generation'],
      });

      registry.registerAgent({
        name: 'Agent2',
        description: 'Second test agent',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['code_review'],
      });

      const allAgents = registry.getAllAgents();
      expect(allAgents).toHaveLength(2);
    });
  });

  describe('Sacred Tongue Governance Tiers', () => {
    it('should define all 6 Sacred Tongues', () => {
      const tongues: GovernanceTier[] = ['KO', 'AV', 'RU', 'CA', 'UM', 'DR'];

      tongues.forEach(tongue => {
        expect(GOVERNANCE_TIERS[tongue]).toBeDefined();
      });
    });

    it('should have increasing trust requirements', () => {
      const tiers = ['KO', 'AV', 'RU', 'CA', 'UM', 'DR'] as GovernanceTier[];
      let prevMinTrust = 0;

      tiers.forEach(tier => {
        const config = GOVERNANCE_TIERS[tier];
        expect(config.minTrustScore).toBeGreaterThanOrEqual(prevMinTrust);
        prevMinTrust = config.minTrustScore;
      });
    });

    it('should have increasing tongue requirements', () => {
      const tiers = ['KO', 'AV', 'RU', 'CA', 'UM', 'DR'] as GovernanceTier[];
      let prevRequired = 0;

      tiers.forEach(tier => {
        const config = GOVERNANCE_TIERS[tier];
        expect(config.requiredTongues).toBeGreaterThanOrEqual(prevRequired);
        prevRequired = config.requiredTongues;
      });
    });

    it('should have 6 tongues required for DR (critical) tier', () => {
      expect(GOVERNANCE_TIERS.DR.requiredTongues).toBe(6);
    });

    it('should have 1 tongue required for KO (read) tier', () => {
      expect(GOVERNANCE_TIERS.KO.requiredTongues).toBe(1);
    });

    it('should have descriptions for all tiers', () => {
      const tiers = ['KO', 'AV', 'RU', 'CA', 'UM', 'DR'] as GovernanceTier[];
      tiers.forEach(tier => {
        expect(GOVERNANCE_TIERS[tier].description).toBeDefined();
        expect(GOVERNANCE_TIERS[tier].description.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Roundtable Consensus Protocol', () => {
    let governance: GovernanceManager;
    let registry: AgentRegistry;

    beforeEach(() => {
      registry = new AgentRegistry();
      governance = new GovernanceManager(registry);

      // Register agents with varying trust levels
      registry.registerAgent({
        name: 'Leader',
        description: 'Roundtable leader agent',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['orchestration'],
        initialTrustVector: [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
      });

      registry.registerAgent({
        name: 'Senior1',
        description: 'Senior code generation agent',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['code_generation'],
        initialTrustVector: [0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
      });
    });

    it('should create roundtable session', () => {
      const session = governance.createRoundtable({
        topic: 'Grant elevated permissions',
        requiredTier: 'KO',  // Use KO tier since we only have 2 agents
      });

      expect(session.id).toBeDefined();
      expect(session.status).toBe('active');
    });

    it('should allow voting on session', () => {
      const allAgents = registry.getAllAgents();
      const leader = allAgents.find(a => a.name === 'Leader');

      const session = governance.createRoundtable({
        topic: 'Test operation',
        requiredTier: 'KO',
        specificParticipants: [leader!.id],
      });

      // Cast a vote
      governance.castVote(session.id, leader!.id, 'approve');

      const updated = governance.getSession(session.id);
      expect(updated).toBeDefined();
      expect(updated?.votes.size).toBeGreaterThan(0);
    });

    it('should get required tier for actions', () => {
      const readTier = governance.getRequiredTier('read');
      expect(readTier).toBe('KO');

      const adminTier = governance.getRequiredTier('admin');
      expect(adminTier).toBe('UM');
    });

    it('should check if agent can perform action', () => {
      const leader = registry.getAllAgents().find(a => a.name === 'Leader');
      expect(leader).toBeDefined();

      const result = governance.canPerformAction(leader!.id, 'read');
      expect(result.allowed).toBeDefined();
      expect(typeof result.allowed).toBe('boolean');
    });
  });

  describe('Task Dispatching', () => {
    let dispatcher: TaskDispatcher;
    let registry: AgentRegistry;

    beforeEach(() => {
      registry = new AgentRegistry();
      dispatcher = new TaskDispatcher(registry);

      registry.registerAgent({
        name: 'Coder',
        description: 'Code generation and testing agent',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['code_generation', 'testing'],
        initialTrustVector: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
      });
    });

    it('should create task', () => {
      const task = dispatcher.createTask({
        name: 'Generate auth module',
        description: 'Create authentication system',
        requiredCapability: 'code_generation',
        requiredTier: 'AV',
        input: { requirements: 'OAuth2 implementation' },
      });

      expect(task.id).toBeDefined();
      expect(task.status).toBe('pending');
    });

    it('should assign task to capable agent', () => {
      const task = dispatcher.createTask({
        name: 'Write tests',
        description: 'Create unit tests',
        requiredCapability: 'testing',
        requiredTier: 'AV',
        input: { target: 'auth module' },
      });

      const result = dispatcher.assignTask(task.id);
      expect(result.success).toBe(true);

      const updatedTask = dispatcher.getTask(task.id);
      expect(updatedTask?.assignedAgentId).toBeDefined();
    });

    it('should fail assignment when no capable agent', () => {
      const task = dispatcher.createTask({
        name: 'Deploy to prod',
        description: 'Production deployment',
        requiredCapability: 'deployment',
        requiredTier: 'CA',
        input: {},
      });

      const result = dispatcher.assignTask(task.id);
      expect(result.success).toBe(false);
    });

    it('should get pending tasks', () => {
      dispatcher.createTask({
        name: 'Task 1',
        description: 'Test task',
        requiredCapability: 'code_generation',
        requiredTier: 'AV',
        input: {},
      });

      const pending = dispatcher.getPendingTasks();
      expect(pending.length).toBeGreaterThan(0);
    });
  });

  describe('Fleet Manager', () => {
    let fleet: FleetManager;

    beforeEach(() => {
      fleet = new FleetManager();
    });

    it('should register agents through fleet', () => {
      const agent = fleet.registerAgent({
        name: 'FleetAgent',
        description: 'Test fleet agent',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['code_generation'],
      });

      expect(agent.id).toBeDefined();
    });

    it('should get fleet statistics', () => {
      fleet.registerAgent({
        name: 'Agent1',
        description: 'Test agent 1',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['code_generation'],
      });

      fleet.registerAgent({
        name: 'Agent2',
        description: 'Test agent 2',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['code_review'],
      });

      const stats = fleet.getStatistics();
      expect(stats.totalAgents).toBe(2);
    });

    it('should create and manage tasks', () => {
      fleet.registerAgent({
        name: 'Coder',
        description: 'Code generation agent',
        provider: 'openai',
        model: 'gpt-4',
        capabilities: ['code_generation'],
        initialTrustVector: [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
      });

      const task = fleet.createTask({
        name: 'Generate feature',
        description: 'New feature',
        requiredCapability: 'code_generation',
        requiredTier: 'AV',
        input: {},
      });

      expect(task).toBeDefined();
      expect(task.id).toBeDefined();
      const retrievedTask = fleet.getTask(task.id);
      expect(retrievedTask).toBeDefined();
    });

    it('should initiate roundtable', () => {
      const agent = fleet.registerAgent({
        name: 'Leader',
        description: 'Fleet leader agent',
        provider: 'anthropic',
        model: 'claude-3',
        capabilities: ['orchestration'],
        initialTrustVector: [0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
      });

      const session = fleet.createRoundtable({
        topic: 'Elevate permissions',
        requiredTier: 'KO',
        specificParticipants: [agent.id],
      });

      expect(session.id).toBeDefined();
    });
  });
});
