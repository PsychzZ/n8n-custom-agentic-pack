import {
  AgentState,
  AgentStep,
  AgentStateNode,
  initializeState,
  appendStep,
  isMaxStepsReached,
  resetState,
  serializeState,
  deserializeState,
} from '../AgentStateNode.node';
import type { IExecuteFunctions } from 'n8n-workflow';

// ---------------------------------------------------------------------------
// n8n IExecuteFunctions mock factory
// ---------------------------------------------------------------------------

function makeExecuteFunctions(
  params: Record<string, unknown>,
  continueOnFail = false,
): IExecuteFunctions {
  const state = initializeState('mock context', 5);
  const serialized = JSON.stringify(serializeState(state));

  return {
    getInputData: () => [{ json: {} }],
    getNodeParameter: (name: string, _index: number) => {
      if (name in params) return params[name];
      if (name === 'currentState') return serialized;
      return undefined;
    },
    continueOnFail: () => continueOnFail,
    getNode: () => ({ name: 'AgentStateNode', type: 'agentStateNode' } as never),
  } as unknown as IExecuteFunctions;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeStep(overrides: Partial<AgentStep> = {}): AgentStep {
  return {
    step: 1,
    thought: 'I need to query the database.',
    action: 'sql_query',
    actionInput: { query: 'SELECT 1' },
    observation: '1 row returned.',
    timestamp: '2025-01-01T00:00:00.000Z',
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// initializeState
// ---------------------------------------------------------------------------

describe('initializeState', () => {
  it('creates a state with an empty history', () => {
    const state = initializeState('Test task');
    expect(state.history).toEqual([]);
    expect(state.stepCount).toBe(0);
  });

  it('assigns a non-empty UUID as sessionId', () => {
    const state = initializeState('Test task');
    expect(state.sessionId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
  });

  it('stores the provided context verbatim', () => {
    const ctx = 'Retrieve Q2 sales data from ERP system.';
    const state = initializeState(ctx);
    expect(state.context).toBe(ctx);
  });

  it('uses default maxSteps of 10 when not provided', () => {
    const state = initializeState('ctx');
    expect(state.maxSteps).toBe(10);
  });

  it('accepts a custom maxSteps value', () => {
    const state = initializeState('ctx', 25);
    expect(state.maxSteps).toBe(25);
  });

  it('stores provided metadata', () => {
    const meta = { projectId: 'proj-42', priority: 'high' };
    const state = initializeState('ctx', 10, meta);
    expect(state.metadata).toEqual(meta);
  });

  it('uses empty metadata by default', () => {
    const state = initializeState('ctx');
    expect(state.metadata).toEqual({});
  });

  it('generates unique sessionIds for different calls', () => {
    const s1 = initializeState('ctx');
    const s2 = initializeState('ctx');
    expect(s1.sessionId).not.toBe(s2.sessionId);
  });
});

// ---------------------------------------------------------------------------
// appendStep
// ---------------------------------------------------------------------------

describe('appendStep', () => {
  let baseState: AgentState;

  beforeEach(() => {
    baseState = initializeState('Base context', 10);
  });

  it('increments stepCount by 1', () => {
    const updated = appendStep(baseState, makeStep());
    expect(updated.stepCount).toBe(1);
  });

  it('appends step to history', () => {
    const step = makeStep({ step: 1, action: 'tool_a' });
    const updated = appendStep(baseState, step);
    expect(updated.history).toHaveLength(1);
    expect(updated.history[0].action).toBe('tool_a');
  });

  it('does not mutate the original state', () => {
    appendStep(baseState, makeStep());
    expect(baseState.history).toHaveLength(0);
    expect(baseState.stepCount).toBe(0);
  });

  it('accumulates multiple steps correctly', () => {
    const s1 = appendStep(baseState, makeStep({ step: 1, action: 'a' }));
    const s2 = appendStep(s1, makeStep({ step: 2, action: 'b' }));
    const s3 = appendStep(s2, makeStep({ step: 3, action: 'c' }));
    expect(s3.stepCount).toBe(3);
    expect(s3.history.map((h) => h.action)).toEqual(['a', 'b', 'c']);
  });

  it('preserves sessionId and context', () => {
    const updated = appendStep(baseState, makeStep());
    expect(updated.sessionId).toBe(baseState.sessionId);
    expect(updated.context).toBe(baseState.context);
  });

  it('preserves metadata', () => {
    const stateWithMeta = initializeState('ctx', 10, { key: 'val' });
    const updated = appendStep(stateWithMeta, makeStep());
    expect(updated.metadata).toEqual({ key: 'val' });
  });
});

// ---------------------------------------------------------------------------
// isMaxStepsReached
// ---------------------------------------------------------------------------

describe('isMaxStepsReached', () => {
  it('returns false when stepCount is below maxSteps', () => {
    const state = initializeState('ctx', 5);
    expect(isMaxStepsReached(state)).toBe(false);
  });

  it('returns true when stepCount equals maxSteps', () => {
    let state = initializeState('ctx', 2);
    state = appendStep(state, makeStep({ step: 1 }));
    state = appendStep(state, makeStep({ step: 2 }));
    expect(isMaxStepsReached(state)).toBe(true);
  });

  it('returns true when stepCount exceeds maxSteps', () => {
    let state = initializeState('ctx', 1);
    state = appendStep(state, makeStep({ step: 1 }));
    // Manually bump stepCount beyond limit
    state = { ...state, stepCount: 5 };
    expect(isMaxStepsReached(state)).toBe(true);
  });

  it('returns false on freshly initialized state', () => {
    const state = initializeState('ctx', 10);
    expect(isMaxStepsReached(state)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// resetState
// ---------------------------------------------------------------------------

describe('resetState', () => {
  it('clears history', () => {
    let state = initializeState('ctx', 10);
    state = appendStep(state, makeStep());
    const fresh = resetState(state);
    expect(fresh.history).toHaveLength(0);
  });

  it('resets stepCount to 0', () => {
    let state = initializeState('ctx', 10);
    state = appendStep(state, makeStep());
    const fresh = resetState(state);
    expect(fresh.stepCount).toBe(0);
  });

  it('preserves sessionId', () => {
    let state = initializeState('ctx', 10);
    state = appendStep(state, makeStep());
    const fresh = resetState(state);
    expect(fresh.sessionId).toBe(state.sessionId);
  });

  it('preserves context', () => {
    const state = initializeState('Important task', 10);
    const fresh = resetState(state);
    expect(fresh.context).toBe('Important task');
  });

  it('preserves maxSteps', () => {
    const state = initializeState('ctx', 42);
    const fresh = resetState(state);
    expect(fresh.maxSteps).toBe(42);
  });

  it('preserves metadata', () => {
    const state = initializeState('ctx', 10, { project: 'thesis' });
    const fresh = resetState(state);
    expect(fresh.metadata).toEqual({ project: 'thesis' });
  });
});

// ---------------------------------------------------------------------------
// serialize / deserialize round-trip
// ---------------------------------------------------------------------------

describe('serializeState / deserializeState', () => {
  it('round-trips a fresh state without data loss', () => {
    const original = initializeState('round-trip test', 7, { k: 'v' });
    const serialized = serializeState(original);
    const restored = deserializeState(serialized);
    expect(restored.sessionId).toBe(original.sessionId);
    expect(restored.context).toBe(original.context);
    expect(restored.maxSteps).toBe(7);
    expect(restored.metadata).toEqual({ k: 'v' });
  });

  it('round-trips a state with history', () => {
    let state = initializeState('ctx', 10);
    state = appendStep(state, makeStep({ action: 'tool_x', observation: 'result_x' }));
    const restored = deserializeState(serializeState(state));
    expect(restored.history).toHaveLength(1);
    expect(restored.history[0].action).toBe('tool_x');
  });

  it('deserializeState throws on missing sessionId', () => {
    expect(() => deserializeState({ context: 'ctx', history: [] })).toThrow();
  });

  it('deserializeState throws on missing context', () => {
    expect(() =>
      deserializeState({ sessionId: 'id', history: [] }),
    ).toThrow();
  });

  it('deserializeState throws on missing history', () => {
    expect(() =>
      deserializeState({ sessionId: 'id', context: 'ctx' }),
    ).toThrow();
  });

  it('deserializeState infers stepCount from history length when missing', () => {
    const raw = {
      sessionId: 'abc',
      context: 'ctx',
      history: [makeStep(), makeStep()],
    };
    const state = deserializeState(raw as unknown as Record<string, unknown>);
    expect(state.stepCount).toBe(2);
  });

  it('serializeState produces JSON-safe output', () => {
    const state = initializeState('ctx');
    const serialized = serializeState(state);
    expect(() => JSON.stringify(serialized)).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// AgentStateNode.execute() — integration tests via mocked IExecuteFunctions
// ---------------------------------------------------------------------------

describe('AgentStateNode.execute()', () => {
  const node = new AgentStateNode();

  it('initialize: creates a new AgentState with correct context', async () => {
    const ctx = makeExecuteFunctions({
      operation: 'initialize',
      context: 'My init context',
      maxSteps: 7,
      metadata: '{"env":"test"}',
    });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.context).toBe('My init context');
    expect(result.json.stepCount).toBe(0);
    expect(result.json.maxSteps).toBe(7);
  });

  it('initialize: uses empty metadata when "{}" provided', async () => {
    const ctx = makeExecuteFunctions({
      operation: 'initialize',
      context: 'ctx',
      maxSteps: 10,
      metadata: '{}',
    });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.metadata).toEqual({});
  });

  it('get: passes state through with validation', async () => {
    const state = initializeState('pass-through', 5);
    const serial = JSON.stringify(serializeState(state));
    const ctx = makeExecuteFunctions({ operation: 'get', currentState: serial });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.context).toBe('pass-through');
  });

  it('reset: clears history while preserving context', async () => {
    let state = initializeState('reset-ctx', 5);
    state = appendStep(state, makeStep());
    const serial = JSON.stringify(serializeState(state));
    const ctx = makeExecuteFunctions({ operation: 'reset', currentState: serial });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.stepCount).toBe(0);
    expect((result.json.history as AgentStep[]).length).toBe(0);
    expect(result.json.context).toBe('reset-ctx');
  });

  it('update: appends a step and increments stepCount', async () => {
    const state = initializeState('update-ctx', 5);
    const serial = JSON.stringify(serializeState(state));
    const ctx = makeExecuteFunctions({
      operation: 'update',
      currentState: serial,
      thought: 'Thinking...',
      action: 'sql_query',
      actionInput: '{"q":"SELECT 1"}',
      observation: 'Got result',
    });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.stepCount).toBe(1);
    expect((result.json.history as AgentStep[])[0].action).toBe('sql_query');
  });

  it('update: throws when maxSteps reached', async () => {
    let state = initializeState('limit-ctx', 1);
    state = appendStep(state, makeStep({ step: 1 }));
    const serial = JSON.stringify(serializeState(state));
    const ctx = makeExecuteFunctions({
      operation: 'update',
      currentState: serial,
      thought: 't',
      action: 'a',
      actionInput: '{}',
      observation: 'o',
    });
    await expect(node.execute.call(ctx)).rejects.toThrow();
  });

  it('unknown operation: throws without continueOnFail', async () => {
    const ctx = makeExecuteFunctions({ operation: 'nonexistent' });
    await expect(node.execute.call(ctx)).rejects.toThrow();
  });

  it('error with continueOnFail: returns error object instead of throwing', async () => {
    const ctx = makeExecuteFunctions(
      { operation: 'get', currentState: '{"invalid":true}' },
      true,
    );
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.error).toBeDefined();
  });
});
